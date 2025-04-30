//! Intertial Training.
//! See the documentation for [`InertialTrainer`] for more details.

use core::fmt;
use std::{
    ops::{Add, AddAssign, Mul},
    random::RandomSource,
    sync::{Arc, RwLock, mpmc, mpsc},
    thread::{self, ScopedJoinHandle},
};

#[allow(unused_imports)]
use crate::continuous::ContinuousTrainer;
use crate::{
    Gate, PopulationStats, TrainingReportStrategy, num_cpus, random_choice_weighted_mapped_by_key,
};

/// This is a unique and novel genetic algorithm strategy, the "inertial" strategy.
///
/// This is a modification of the [`ContinuousTrainer`] that implements a further novel extension of the
/// genetic algorithm. Instead of mutations and crossovers, this trainer makes exclusive use of a concept referred
/// to as "mutation vectors". Borrowing from the more widely used machine learning technique of gradient descent,
/// mutations are characterized not as random adjustemnts to a given gene, but as mutation vectors, which can
/// represent an abstract "direction" of mutational progress. In this way, a gene accrues
/// "mutation velocity" over time as it mutates in positive ways, gaining momentum towards mutation directions that
/// improve its fitness.
///
/// As it's a modification of the [`ContinuousTrainer`], many of the same structures and method are shared between the
/// two trainers.
pub struct InertialTrainer<'scope, G>
where
    G: InertialGenome,
{
    /// A collection of all the genes in the population,
    /// sorted descending by fitness.
    ///
    /// Includes additional relevant information
    /// necessary to perform the inertia based genetic algorithm.
    ///
    /// Because of the continuous nature of the trainer,
    /// the collection is behind an `Arc<RwLock<_>>` combo.
    pub gene_pool: Arc<RwLock<Vec<RankedGenome<G>>>>,

    /// Count of the total number of children reproduced.
    pub children_created: usize,

    /// The mutation rate of newly reproduced children.
    pub mutation_rate: f32,

    /// The impact of inertia on a gene's mutation. A
    /// value of 0 disables inertia entirely, effectively
    /// making this trainer identical to the [`ContinuousTrainer`],
    /// but with double the performance cost. A positive value close to zero
    /// is typically desired, between 0.1-0.0001.
    ///
    /// Contextualizing this optimizer as a form of PID controller,
    /// this parameter is akin to the "proportional" component.
    pub inertia: f32,

    /// The amount of damping to apply to a gene's inertia over time;
    /// set to between 0-1. 1 means no damping, and 0 will completely cancel
    /// out all impacts from velocity entirely, effectively making
    /// this trainer identical to the [`ContinuousTrainer`], but
    /// with double the performance cost. A value below but close
    /// to 1 is typically desired, between 0.9-0.999.
    ///
    /// Contextualizing this optimizer as a form of PID controller,
    /// this parameter is akin to the "derivative" component.
    pub damping: f32,

    work_submission: mpmc::Sender<FirstStageJob<G>>,
    #[allow(unused)]
    first_stage_worker_pool: Vec<ScopedJoinHandle<'scope, ()>>,
    #[allow(unused)]
    second_stage_worker_pool: Vec<ScopedJoinHandle<'scope, ()>>,
    #[allow(unused)]
    receiver_thread: ScopedJoinHandle<'scope, ()>,
    population_size: usize,
    in_flight: Gate<usize>,
}

impl<'scope, G> InertialTrainer<'scope, G>
where
    G: InertialGenome,
{
    /// Construct a new trainer with given population size, 
    /// mutation rate, inertia, and damping parameters.
    ///
    /// A reference to a [`thread::Scope`] must be passed in order
    /// to spawn the child worker threads for the lifetime of the trainer.
    pub fn new(
        population_size: usize,
        mutation_rate: f32,
        inertia: f32,
        damping: f32,
        scope: &'scope thread::Scope<'scope, '_>,
    ) -> Self
    where
        G: 'scope + Send + Sync,
    {
        let in_flight = Gate::new(0);
        let (work_submission, inbox) = mpmc::channel();
        let (first_stage_outbox, second_stage_inbox) = mpmc::channel();
        let (outbox, work_reception) = mpsc::channel();
        let gene_pool = Arc::new(RwLock::new(Vec::new()));
        let first_stage_worker_pool = (0..num_cpus())
            .map(|_| {
                let inbox = inbox.clone();
                let outbox = outbox.clone();
                let first_stage_outbox = first_stage_outbox.clone();
                scope.spawn(move || Self::first_stage_worker(inbox, outbox, first_stage_outbox))
            })
            .collect();
        let second_stage_worker_pool = (0..num_cpus())
            .map(|_| {
                let second_stage_inbox = second_stage_inbox.clone();
                let outbox = outbox.clone();
                scope.spawn(move || Self::second_stage_worker(second_stage_inbox, outbox))
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
            damping,
            work_submission,
            first_stage_worker_pool,
            second_stage_worker_pool,
            receiver_thread,
            population_size,
            in_flight,
        }
    }

    fn first_stage_worker(
        inbox: mpmc::Receiver<FirstStageJob<G>>,
        outbox: mpsc::Sender<RankedGenome<G>>,
        first_stage_outbox: mpmc::Sender<SecondStageJob<G>>,
    ) {
        for job in inbox {
            match job {
                FirstStageJob::NewGene(gene) => {
                    let mut ranked_gene: RankedGenome<G> = gene.into();
                    ranked_gene.eval();
                    outbox.send(ranked_gene).unwrap();
                }
                FirstStageJob::FirstStageAncestorEvaluation {
                    mut gene,
                    mutation_to_apply,
                    velocity,
                } => {
                    let prior_fitness = gene.fitness();
                    gene.apply_mutation(&mutation_to_apply);
                    first_stage_outbox
                        .send(SecondStageJob {
                            gene,
                            prior_fitness,
                            recent_mutation: mutation_to_apply,
                            velocity,
                        })
                        .unwrap();
                }
            }
        }
    }

    fn second_stage_worker(
        second_stage_inbox: mpmc::Receiver<SecondStageJob<G>>,
        outbox: mpsc::Sender<RankedGenome<G>>,
    ) {
        for SecondStageJob {
            gene,
            prior_fitness,
            recent_mutation,
            velocity,
        } in second_stage_inbox
        {
            let current_fitness = gene.fitness();
            let ranked_gene = RankedGenome {
                gene,
                current_fitness,
                prior_fitness,
                velocity,
                recent_mutation,
            };
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
            let insert_index = gene_pool
                .binary_search_by(|rg| ranked_gene.current_fitness.total_cmp(&rg.current_fitness));
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

    fn submit_job(&mut self, ranked_gene: FirstStageJob<G>) {
        self.children_created += 1;
        self.in_flight.update(|x| x.add_assign(1));
        self.work_submission.send(ranked_gene).unwrap();
    }

    /// Seed the population with new genes up to the current population cap.
    ///
    /// This is called automatically at the start of training, so should typically
    /// not need to be called directly.
    ///
    /// A [`RandomSource`] must be passed as a source of randomness
    /// for generating the initial population.
    pub fn seed<R>(&mut self, rng: &mut R)
    where
        R: RandomSource,
    {
        let current_gene_pool_size = self.gene_pool.read().unwrap().len();
        for _ in current_gene_pool_size..self.population_size {
            self.submit_job(FirstStageJob::NewGene(G::generate(rng)));
        }
    }

    /// Begin training, finishing once `num_children` children have been
    /// reproduced, ranked for fitness, and introduced into the population.
    ///
    /// A [`RandomSource`] must be passed as a source of randomness
    /// for mutating genes to produce new offspring.
    pub fn train<R>(&mut self, num_children: usize, rng: &mut R) -> G
    where
        R: RandomSource,
        G: Clone + Send + Sync + 'scope,
    {
        self.train_custom(
            |x| x.child_count <= num_children,
            Some(default_reporting_strategy(self.population_size)),
            rng,
        )
    }

    /// Begin training with detailed custom parameters.
    ///
    /// Instead of a specific child
    /// count cutoff point, a function `train_criteria` is passed in, which takes in an
    /// instance of [`TrainingCriteriaMetrics`] and outputs a `bool`. This allows greater
    /// control over exactly what criteria to finish training under.
    ///
    /// Additionally, the user may pass a `reporting_strategy`, which determines the conditions
    /// and method under which periodic statistical reporting of the population is performed.
    /// Pass `None` to disable reporting entirely, otherwise pass `Some` with an instance of a
    /// [`TrainingReportStrategy`] to define the two methods necessary to manage reporting.
    /// To mimic the default reporting strategy, pass the result of [`default_reporting_strategy()`] wrapped
    /// in `Some()`.
    /// 
    /// A [`RandomSource`] must be passed as a source of randomness
    /// for mutating genes to produce new offspring.
    pub fn train_custom<R>(
        &mut self,
        mut train_criteria: impl FnMut(TrainingCriteriaMetrics) -> bool,
        mut reporting_strategy: Option<
            TrainingReportStrategy<
                impl FnMut(TrainingCriteriaMetrics) -> bool,
                impl FnMut(TrainingStats),
            >,
        >,
        rng: &mut R,
    ) -> G
    where
        R: RandomSource,
        G: Clone,
    {
        self.seed(rng);
        self.in_flight.wait_while(|x| *x > 0);
        loop {
            let parent = {
                let gene_pool = self.gene_pool.read().unwrap();
                let min_fitness = gene_pool
                    .iter()
                    .map(|x| x.current_fitness)
                    .min_by(|a, b| a.total_cmp(b))
                    .unwrap();
                random_choice_weighted_mapped_by_key(
                    &gene_pool,
                    rng,
                    |x| x - min_fitness,
                    |x| x.current_fitness,
                )
                .clone()
            };

            let delta_fitness = (parent.current_fitness - parent.prior_fitness) * self.inertia;
            let acceleration = parent.recent_mutation * delta_fitness;
            let velocity = (parent.velocity + acceleration) * self.damping;
            let mut gene = parent.gene;
            gene.apply_mutation(&velocity);
            let mutation_to_apply = G::create_mutation(rng);
            let new_job = FirstStageJob::FirstStageAncestorEvaluation {
                gene,
                mutation_to_apply,
                velocity,
            };
            self.submit_job(new_job);

            let metrics = self.metrics();
            if let Some(reporting_strategy) = &mut reporting_strategy {
                if (reporting_strategy.should_report)(metrics) {
                    (reporting_strategy.report_callback)(self.stats())
                }
            }
            if !(train_criteria)(metrics) {
                break;
            }
        }
        self.in_flight.wait_while(|x| *x > 0);
        self.gene_pool.read().unwrap().first().unwrap().gene.clone()
    }

    /// Generate training criteria metrics for the current state of this trainer.
    ///
    /// This is a strict subset of the data available in an instance of [`TrainingStats`]
    /// returned from calling [`ContinuousTrainer::stats`]. However, these
    /// metrics were chosen specifically for their computation efficiency, and thus can be
    /// re-evaluated frequently with minimal cost. These metrics are used both to determine
    /// whether or not to continue training, and whether or not to display a report about
    /// training progress.
    pub fn metrics(&self) -> TrainingCriteriaMetrics {
        let gene_pool = self.gene_pool.read().unwrap();
        TrainingCriteriaMetrics {
            max_fitness: gene_pool.first().unwrap().current_fitness,
            min_fitness: gene_pool.last().unwrap().current_fitness,
            median_fitness: gene_pool[gene_pool.len() / 2].current_fitness,
            child_count: self.children_created,
        }
    }

    /// Generate population stats for the current state of this trainer.
    ///
    /// This function is called whenever the reporting strategy is asked
    /// to produce a report about the current population, but it may also be called
    /// manually here.
    pub fn stats(&self) -> TrainingStats {
        TrainingStats {
            population_stats: self
                .gene_pool
                .read()
                .unwrap()
                .iter()
                .map(|x| x.current_fitness)
                .collect(),
            child_count: self.children_created,
        }
    }
}

/// A collection of relevant & quick to compute metrics that
/// can be used to inform whether or not to continue training.
#[derive(Clone, Copy, Debug)]
pub struct TrainingCriteriaMetrics {
    /// Maximum fitness of the population.
    pub max_fitness: f32,

    /// Minimum fitness of the population.
    pub min_fitness: f32,

    /// Median fitness of the population.
    pub median_fitness: f32,

    /// Total number of children that have been
    /// reproduced and introduced into the population,
    /// including the initial seed population count.
    pub child_count: usize,
}

/// A collection of statistics about the population as a whole.
///
/// Relatively more expensive to compute than training metrics, so
/// should be computed infrequently.
#[derive(Clone, Copy, Debug)]
pub struct TrainingStats {
    /// A collection of standard population stats: see [`PopulationStats`]
    /// for more information
    pub population_stats: PopulationStats,

    /// Total number of children that have been
    /// reproduced and introduced into the population,
    /// including the initial seed population count.
    ///
    /// Same as [`TrainingCriteriaMetrics::child_count`].
    pub child_count: usize,
}

impl fmt::Display for TrainingStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "child #{} {}", self.child_count, self.population_stats)
    }
}

/// Returns a default reporting strategy which logs population
/// statistics to the console every `n` children reproduced.
///
/// Used by [`InertialTrainer::train`].
pub fn default_reporting_strategy(
    n: usize,
) -> TrainingReportStrategy<impl FnMut(TrainingCriteriaMetrics) -> bool, impl FnMut(TrainingStats)>
{
    TrainingReportStrategy {
        should_report: move |m: TrainingCriteriaMetrics| m.child_count % n == 0,
        report_callback: |s| println!("{s}"),
    }
}

enum FirstStageJob<G>
where
    G: InertialGenome,
{
    NewGene(G),
    FirstStageAncestorEvaluation {
        gene: G,
        mutation_to_apply: G::MutationVector,
        velocity: G::MutationVector,
    },
}

struct SecondStageJob<G>
where
    G: InertialGenome,
{
    gene: G,
    prior_fitness: f32,
    recent_mutation: G::MutationVector,
    velocity: G::MutationVector,
}

#[derive(Clone, Debug)]
pub struct RankedGenome<G>
where
    G: InertialGenome,
{
    gene: G,
    current_fitness: f32,
    prior_fitness: f32,
    velocity: G::MutationVector,
    recent_mutation: G::MutationVector,
}

impl<G> RankedGenome<G>
where
    G: InertialGenome,
{
    fn eval(&mut self) {
        self.current_fitness = self.gene.fitness();
    }
}

impl<G> From<G> for RankedGenome<G>
where
    G: InertialGenome,
{
    fn from(gene: G) -> Self {
        RankedGenome {
            gene,
            current_fitness: 0.0,
            prior_fitness: 0.0,
            velocity: G::MutationVector::default(),
            recent_mutation: G::MutationVector::default(),
        }
    }
}


/// Represents a single Genome in the inertial genetic algorithm.
///
/// The unique functionality of the intertial trainer requires a unique
/// genome implementation that operates with mutation vectors, as opposed to just
/// applying mutations statically.
/// Implement this trait for your type to evolve it using the [`InertialTrainer`].
pub trait InertialGenome {
    /// Represents a mutation vector.
    ///
    /// It must obey basic vector space
    /// operations, such as vector addition and scalar multiplication.
    /// The default value should be a "zero" vector, which performs no
    /// mutations if applied.
    type MutationVector: Clone
        + Sized
        + Default
        + Send
        + Sync
        + Add<Self::MutationVector, Output = Self::MutationVector>
        + Mul<f32, Output = Self::MutationVector>;

    /// Generate a new instance of this genome from the given random source.
    fn generate<R>(rng: &mut R) -> Self
    where
        R: RandomSource;

    /// Generate a random mutation vector that can be used to
    /// mutate this genome.
    fn create_mutation<R>(rng: &mut R) -> Self::MutationVector
    where
        R: RandomSource;

    /// Apply the mutations encapsulated by a mutation vector to this genome.
    fn apply_mutation(&mut self, mutation: &Self::MutationVector);

    /// Evaluate this genome for its 'fitness' score.
    ///
    /// Higher fitness scores will lead to a higher survival rate.
    ///
    /// Negative fitness scores are valid.
    fn fitness(&self) -> f32;
}
