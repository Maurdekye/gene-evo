use std::{
    random::RandomSource,
    sync::{mpmc, mpsc},
    thread::{self, ScopedJoinHandle},
};

use crate::{Genome, PopulationStats, SelectionStrategy, num_cpus, random_choice_weighted};

#[allow(unused)]
pub struct StochasticTrainer<'scope, G> {
    pub gene_pool: Vec<(G, Option<f32>)>,
    worker_pool: Vec<ScopedJoinHandle<'scope, ()>>,
    work_submission: mpmc::Sender<(usize, G)>,
    work_reception: mpsc::Receiver<(usize, f32)>,
    pub epoch: usize,
    population_size: usize,
    mutation_rate: f32,
}

impl<'scope, G> StochasticTrainer<'scope, G> {
    pub fn new<R>(
        population_size: usize,
        mutation_rate: f32,
        rng: &mut R,
        scope: &'scope thread::Scope<'scope, '_>,
    ) -> Self
    where
        G: Genome + Send + 'scope,
        R: RandomSource,
    {
        let (work_submission, inbox) = mpmc::channel();
        let (outbox, work_reception) = mpsc::channel();
        let gene_pool = (0..population_size)
            .map(|_| (G::generate(rng), None))
            .collect();
        let worker_pool = (0..num_cpus())
            .map(|_| {
                let inbox = inbox.clone();
                let outbox = outbox.clone();
                scope.spawn(move || StochasticTrainer::<G>::worker_thread(inbox, outbox))
            })
            .collect();
        Self {
            gene_pool,
            worker_pool,
            work_submission,
            work_reception,
            epoch: 0,
            population_size,
            mutation_rate,
        }
    }

    fn worker_thread<I>(inbox: mpmc::Receiver<(I, G)>, outbox: mpsc::Sender<(I, f32)>)
    where
        G: Genome,
    {
        for (id, gene) in inbox {
            let result = gene.fitness();
            outbox.send((id, result)).unwrap();
        }
    }

    pub fn eval(&mut self)
    where
        G: Clone,
    {
        let to_eval = self
            .gene_pool
            .iter()
            .enumerate()
            .filter_map(|(i, (gene, score))| score.is_none().then(|| (i, gene.clone())))
            .map(|p| self.work_submission.send(p).unwrap())
            .count();
        for (i, score) in self.work_reception.iter().take(to_eval) {
            self.gene_pool[i].1 = Some(score);
        }
    }

    pub fn scores(&self) -> impl Iterator<Item = f32> {
        self.gene_pool.iter().filter_map(|(_, s)| *s)
    }

    pub fn score_bounds(&self) -> Option<(f32, f32)> {
        let mut minmax = None;
        for score in self.scores() {
            minmax = match (minmax, score) {
                (None, score) => Some((score, score)),
                (Some((min, max)), score) if score < min => Some((score, max)),
                (Some((min, max)), score) if score > max => Some((min, score)),
                (minmax, _) => minmax,
            };
        }
        minmax
    }

    pub fn prune<S, R>(&mut self, rng: &mut R)
    where
        S: SelectionStrategy,
        R: RandomSource,
    {
        let (min_score, max_score) = self.score_bounds().unwrap();
        let score_range = max_score - min_score;
        self.gene_pool.retain(|(_, score)| {
            let Some(score) = score else {
                return true;
            };
            let percentile = (*score - min_score) / score_range;
            S::select(percentile, rng)
        });
    }

    pub fn reproduce<R>(&mut self, rng: &mut R)
    where
        R: RandomSource,
        G: Genome + Clone,
    {
        let (min_score, max_score) = self.score_bounds().unwrap();
        let score_range = max_score - min_score;
        let mut total_score = 0.0;
        let percentile_pairs: Vec<_> = self
            .gene_pool
            .iter()
            .enumerate()
            .map(|(i, (_, score))| {
                let Some(score) = score else {
                    return (i, 0.0);
                };
                let percentile = (*score - min_score) / score_range;
                total_score += percentile;
                (i, percentile)
            })
            .collect();
        while self.gene_pool.len() < self.population_size {
            let &parent_index = random_choice_weighted(&percentile_pairs, rng);
            let mut new_child = self.gene_pool[parent_index].0.clone();
            new_child.mutate(self.mutation_rate, rng);
            self.gene_pool.push((new_child, None));
        }
    }

    pub fn step<S, R>(&mut self, rng: &mut R) -> PopulationStats
    where
        G: Clone + Genome,
        R: RandomSource,
        S: SelectionStrategy,
    {
        self.epoch += 1;
        self.eval();
        let results: PopulationStats = self.scores().collect();
        self.prune::<S, R>(rng);
        self.reproduce(rng);
        results
    }
}
