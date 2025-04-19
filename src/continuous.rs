use std::{
    random::RandomSource,
    sync::{Arc, RwLock, mpmc, mpsc},
    thread::{self, ScopedJoinHandle},
};

use crate::{Genome, PopulationStats, num_cpus, random_choice_weighted};

#[allow(unused)]
pub struct ContinuousTrainer<'scope, G> {
    pub gene_pool: Arc<RwLock<Vec<(G, f32)>>>,
    work_submission: mpmc::Sender<G>,
    worker_pool: Vec<ScopedJoinHandle<'scope, ()>>,
    receiver_thread: ScopedJoinHandle<'scope, ()>,
    mutation_rate: f32,
    population_size: usize,
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
            scope.spawn(move || {
                ContinuousTrainer::<G>::work_receiver_thread(
                    work_reception,
                    gene_pool,
                    population_size,
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
        }
    }

    pub fn seed<R>(&mut self, rng: &mut R)
    where
        R: RandomSource,
        G: Genome,
    {
        let current_gene_pool_size = self.gene_pool.read().unwrap().len();
        for _ in current_gene_pool_size..self.population_size {
            self.work_submission.send(G::generate(rng)).unwrap();
        }
    }

    pub fn train<R>(&mut self, num_children: usize, rng: &mut R)
    where
        R: RandomSource,
        G: Clone + Genome + Send + Sync + 'scope,
    {
        for _ in 0..num_children {
            let mut new_child = {
                let gene_pool = self.gene_pool.read().unwrap();
                random_choice_weighted(&gene_pool, rng).clone()
            };
            new_child.mutate(self.mutation_rate, rng);
            self.work_submission.send(new_child).unwrap();
        }
    }

    pub fn population_stats(&self) -> PopulationStats {
        self.gene_pool.read().unwrap().iter().map(|x| x.1).collect()
    }
}
