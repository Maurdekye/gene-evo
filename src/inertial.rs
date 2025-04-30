//! Intertial Training.
//! See the documentation for [`InertialTrainer`] for more details.

use std::{
    ops::{Add, AddAssign, Mul},
    random::RandomSource,
    sync::{Arc, RwLock, mpmc, mpsc},
    thread::{self, ScopedJoinHandle},
};

#[allow(unused_imports)]
use crate::continuous;
use crate::{Gate, num_cpus, random_choice_weighted_mapped_by_key};

/// This is a unique genetic algorithm strategy, the "inertial" strategy.
///
/// This is a modification of the [`continuous::ContinuousTrainer`] that implements a further extension of the
/// genetic algorithm. instead of mutations and crossovers, this trainer makes exclusive use of a concept referred
/// to as "mutation vectors". Borrowing from the more well known machine learning technique of gradient descent,
/// mutations are characterized not as random adjustemnts to a given gene, but as mutation vectors, which can
/// represent a given gene's velocity making mutations in a certain direction. in this way, a gene accrues
/// "mutation inertia" over time as it mutates in positive ways, gaining momentum towards mutation directions that
/// improve its fitness.
pub struct InertialTrainer<'scope, G>
where
    G: InertialGenome,
{
    pub gene_pool: Arc<RwLock<Vec<RankedGenome<G>>>>,
    pub children_created: usize,
    pub mutation_rate: f32,
    pub inertia: f32,

    work_submission: mpmc::Sender<RankedGenome<G>>,
    worker_pool: Vec<ScopedJoinHandle<'scope, ()>>,
    receiver_thread: ScopedJoinHandle<'scope, ()>,
    population_size: usize,
    in_flight: Gate<usize>,
}

impl<'scope, G> InertialTrainer<'scope, G>
where
    G: InertialGenome,
{
    pub fn new(
        population_size: usize,
        mutation_rate: f32,
        inertia: f32,
        scope: &'scope thread::Scope<'scope, '_>,
    ) -> Self
    where
        G: 'scope + Send + Sync,
    {
        let in_flight = Gate::new(0);
        let (work_submission, inbox) = mpmc::channel();
        let (outbox, work_reception) = mpsc::channel();
        let gene_pool = Arc::new(RwLock::new(Vec::new()));
        let worker_pool = (0..num_cpus())
            .map(|_| {
                let inbox = inbox.clone();
                let outbox = outbox.clone();
                scope.spawn(move || Self::worker_thread(inbox, outbox))
            })
            .collect();
        let receiver_thread = {
            let gene_pool = gene_pool.clone();
            let in_flight = in_flight.clone();
            scope.spawn(move || {
                Self::work_receiver_thread(work_reception, gene_pool, population_size, in_flight)
            })
        };
        Self {
            gene_pool,
            children_created: 0,
            mutation_rate,
            inertia,
            work_submission,
            worker_pool,
            receiver_thread,
            population_size,
            in_flight,
        }
    }

    fn worker_thread(
        inbox: mpmc::Receiver<RankedGenome<G>>,
        outbox: mpsc::Sender<RankedGenome<G>>,
    ) {
        for mut ranked_gene in inbox {
            ranked_gene.eval();
            outbox.send(ranked_gene).unwrap();
        }
    }

    fn work_receiver_thread(
        work_reception: mpsc::Receiver<RankedGenome<G>>,
        gene_pool: Arc<RwLock<Vec<RankedGenome<G>>>>,
        max_population_size: usize,
        in_flight: Gate<usize>,
    ) {
        for ranked_gene in work_reception {
            let mut gene_pool = gene_pool.write().unwrap();
            let insert_index =
                gene_pool.binary_search_by(|rg| ranked_gene.fitness.total_cmp(&rg.fitness));
            let insert_index = match insert_index {
                Ok(i) => i,
                Err(i) => i,
            };
            gene_pool.insert(insert_index, ranked_gene);
            if gene_pool.len() > max_population_size {
                gene_pool.drain(max_population_size..);
            }
            in_flight.update(|x| *x = x.saturating_sub(1));
        }
    }

    pub fn submit_job(&mut self, ranked_gene: RankedGenome<G>) {
        self.children_created += 1;
        self.in_flight.update(|x| x.add_assign(1));
        self.work_submission.send(ranked_gene).unwrap();
    }

    pub fn seed<R>(&mut self, rng: &mut R)
    where
        R: RandomSource,
    {
        let current_gene_pool_size = self.gene_pool.read().unwrap().len();
        for _ in current_gene_pool_size..self.population_size {
            self.submit_job(G::generate(rng).into());
        }
    }

    pub fn train_custom<R>(&mut self, num_children: usize, rng: &mut R) -> G
    where
        R: RandomSource,
        G: Clone,
    {
        self.seed(rng);
        self.in_flight.wait_while(|x| *x > 0);
        loop {
            let new_parent = {
                let gene_pool = self.gene_pool.read().unwrap();
                let min_fitness = gene_pool
                    .iter()
                    .map(|x| x.fitness)
                    .min_by(|a, b| a.total_cmp(b))
                    .unwrap();
                random_choice_weighted_mapped_by_key(
                    &gene_pool,
                    rng,
                    |x| x - min_fitness,
                    |x| x.fitness,
                )
                .clone()
            };
            // todo: mutation algorithm
            self.submit_job(new_parent);
            if self.children_created >= num_children {
                break;
            }
        }
        self.in_flight.wait_while(|x| *x > 0);
        self.gene_pool.read().unwrap().first().unwrap().gene.clone()
    }
}

#[derive(Clone, Debug)]
pub struct RankedGenome<G>
where
    G: InertialGenome,
{
    gene: G,
    fitness: f32,
    parent_fitness: Option<f32>,
    inertia: G::MutationVector,
}

impl<G> RankedGenome<G>
where
    G: InertialGenome,
{
    fn eval(&mut self) {
        self.fitness = self.gene.fitness();
    }
}

impl<G> From<G> for RankedGenome<G>
where
    G: InertialGenome,
{
    fn from(gene: G) -> Self {
        RankedGenome {
            gene,
            fitness: 0.0,
            parent_fitness: None,
            inertia: G::MutationVector::default(),
        }
    }
}

/// Represents a mutation vector. It must obey basic vector space
/// operations, such as vector addition and scalar multiplication.
/// The default value should be a "zero" vector, which performs no
/// mutations if applied.
pub trait MutationVector:
    Clone + Sized + Default + Send + Sync + Add<Self, Output = Self> + Mul<f32, Output = Self>
{
}

/// Represents a single Genome in the inertial genetic algorithm.
///
/// The unique functionality of the intertial trainer requires a unique
/// genome implementation that operates with mutation vectors as opposed to just
/// applying mutations statically.
/// Implement this trait for your type to evolve it using the [`InertialTrainer`].
pub trait InertialGenome {
    /// The type that this gene uses to represent its mutation vectors. See
    /// the documentation on [`MutationVector`] for more information.
    type MutationVector: MutationVector;

    /// Generate a new instance of this genome from the given random source.
    fn generate<R>(rng: &mut R) -> Self
    where
        R: RandomSource;

    /// Generate a random mutation vector that can be used to
    /// mutate this genome.
    fn create_mutation<R>(&self, rng: &mut R) -> Self::MutationVector
    where
        R: RandomSource;

    /// Apply the mutations encapsulated by a mutation vector to this genome.
    fn apply_mutation(&mut self, mutation: Self::MutationVector);

    /// Evaluate this genome for its 'fitness' score. Higher fitness
    /// scores will lead to a higher survival rate.
    ///
    /// Negative fitness scores are valid.
    fn fitness(&self) -> f32;
}
