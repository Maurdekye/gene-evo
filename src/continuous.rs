use std::{
    ops::AddAssign,
    random::RandomSource,
    sync::{Arc, RwLock, mpmc, mpsc},
    thread::{self, ScopedJoinHandle},
};

use crate::{
    Gate, GeneticTrainer, Genome, PopulationStats, num_cpus, random_choice_weighted_mapped,
};

#[allow(unused)]
pub struct ContinuousTrainer<'scope, G> {
    pub gene_pool: Arc<RwLock<Vec<(G, f32)>>>,
    work_submission: mpmc::Sender<G>,
    worker_pool: Vec<ScopedJoinHandle<'scope, ()>>,
    receiver_thread: ScopedJoinHandle<'scope, ()>,
    mutation_rate: f32,
    population_size: usize,
    in_flight: Gate<usize>,
}

impl<'scope, G> ContinuousTrainer<'scope, G> {
    pub fn new(
        population_size: usize,
        mutation_rate: f32,
        scope: &'scope thread::Scope<'scope, '_>,
    ) -> Self
    where
        G: Genome + 'scope + Send + Sync,
    {
        let in_flight = Gate::new(0);
        let (work_submission, inbox) = mpmc::sync_channel(0);
        let (outbox, work_reception) = mpsc::channel();
        let gene_pool = Arc::new(RwLock::new(Vec::new()));
        let worker_pool = (0..num_cpus())
            .map(|_| {
                let inbox = inbox.clone();
                let outbox = outbox.clone();
                scope.spawn(move || ContinuousTrainer::<G>::worker_thread(inbox, outbox))
            })
            .collect();
        let receiver_thread = {
            let gene_pool = gene_pool.clone();
            let in_flight = in_flight.clone();
            scope.spawn(move || {
                ContinuousTrainer::<G>::work_receiver_thread(
                    work_reception,
                    gene_pool,
                    population_size,
                    in_flight,
                )
            })
        };
        Self {
            gene_pool,
            work_submission,
            worker_pool,
            receiver_thread,
            mutation_rate,
            population_size,
            in_flight,
        }
    }

    fn worker_thread(inbox: mpmc::Receiver<G>, outbox: mpsc::Sender<(G, f32)>)
    where
        G: Genome,
    {
        for gene in inbox {
            let fitness = gene.fitness();
            outbox.send((gene, fitness)).unwrap();
        }
    }

    fn work_receiver_thread(
        work_reception: mpsc::Receiver<(G, f32)>,
        gene_pool: Arc<RwLock<Vec<(G, f32)>>>,
        max_population_size: usize,
        in_flight: Gate<usize>,
    ) {
        for (gene, score) in work_reception {
            let mut gene_pool = gene_pool.write().unwrap();
            let insert_index = gene_pool.binary_search_by(|x| score.total_cmp(&x.1));
            let insert_index = match insert_index {
                Ok(i) => i,
                Err(i) => i,
            };
            gene_pool.insert(insert_index, (gene, score));
            if gene_pool.len() > max_population_size {
                gene_pool.drain(max_population_size..);
            }
            in_flight.update(|x| *x = x.saturating_sub(1));
        }
    }

    pub fn submit_job(&mut self, gene: G) {
        self.in_flight.update(|x| x.add_assign(1));
        self.work_submission.send(gene).unwrap();
    }

    pub fn seed<R>(&mut self, rng: &mut R)
    where
        R: RandomSource,
        G: Genome,
    {
        let current_gene_pool_size = self.gene_pool.read().unwrap().len();
        for _ in current_gene_pool_size..self.population_size {
            self.submit_job(G::generate(rng));
        }
    }

    pub fn train<R>(&mut self, num_children: usize, rng: &mut R) -> G
    where
        R: RandomSource,
        G: Clone + Genome + Send + Sync + 'scope,
    {
        self.seed(rng);
        for i in 0..num_children {
            let mut new_child = {
                let gene_pool = self.gene_pool.read().unwrap();
                let min_fitness = gene_pool
                    .iter()
                    .map(|x| x.1)
                    .min_by(|a, b| a.total_cmp(b))
                    .unwrap();
                random_choice_weighted_mapped(&gene_pool, rng, |x| x - min_fitness).clone()
            };
            new_child.mutate(self.mutation_rate, rng);
            self.submit_job(new_child);
            if i % self.population_size == 0 {
                println!("child {i}, {}", self.population_stats());
            }
        }
        self.in_flight.wait_while(|x| *x > 0);
        self.gene_pool.read().unwrap().first().unwrap().0.clone()
    }

    pub fn population_stats(&self) -> PopulationStats {
        self.gene_pool.read().unwrap().iter().map(|x| x.1).collect()
    }
}

pub struct ContinuousTrainerParams {
    pub num_children: usize,
}

impl<'scope, G> GeneticTrainer<G> for ContinuousTrainer<'scope, G>
where
    G: Clone + Genome + Send + Sync + 'scope,
{
    type TrainingParams = ContinuousTrainerParams;

    fn train<R: RandomSource>(&mut self, rng: &mut R, params: Self::TrainingParams) -> G {
        ContinuousTrainer::train(self, params.num_children, rng)
    }
}
