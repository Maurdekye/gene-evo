//! Stochastic Training
//! See the documentation for [`StochasticTrainer`] for more details.

use std::{
    fmt,
    random::RandomSource,
    sync::{mpmc, mpsc},
    thread::{self, ScopedJoinHandle},
};

use crate::{
    Genome, PopulationStats, TrainingReportStrategy, num_cpus, random_choice_weighted, random_f32,
};

/// This is an implementation of the familiar and well known stochastic training strategy, whereby
/// the full population is evaluated for fitness all at once, some portion is culled based on
/// their fitness score, and then the remaining empty spaces are repopulated from the surviving
/// genes, with choice likelihood based on their relative fitness scores. Each iteration of this
/// process is a single "generation" of the algorithm.
pub struct StochasticTrainer<'scope, G> {
    /// A collection of all genes in the population, alongside
    /// their associated fitness scores, if computed at this point in the
    /// training process. Note that members of the population
    /// in this trainer are unsorted, unlike in [`crate::continuous::ContinuousTrainer`]
    pub gene_pool: Vec<(G, Option<f32>)>,

    /// The current number of generations the trainer has iterated.
    pub generation: usize,

    /// The mutation rate of newly reproduced children.
    pub mutation_rate: f32,
    #[allow(unused)]
    worker_pool: Vec<ScopedJoinHandle<'scope, ()>>,
    work_submission: mpmc::Sender<(usize, G)>,
    work_reception: mpsc::Receiver<(usize, f32)>,
    population_size: usize,
}

impl<'scope, G> StochasticTrainer<'scope, G> {
    /// Construct a new trainer with a given population size and mutation rate.
    /// A [`std::random::RandomSource`] must be passed in order to seed the
    /// initial population.
    /// A reference to a [`thread::Scope`] must be passed as well in order
    /// to spawn the child worker threads for the lifetime of the trainer.
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
            generation: 0,
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

    /// Evaluate the fitness of all genes in the population without fitness
    /// scores in parallel. All genes with a `None` as the second value in
    /// their pair in the [`StochasticTrainer::gene_pool`] will have their
    /// fitness evaluated and defined.
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

    /// Retrieve an iterator over all the individual fitness scores
    /// in the population.
    pub fn scores(&self) -> impl Iterator<Item = f32> {
        self.gene_pool.iter().filter_map(|(_, s)| *s)
    }

    /// Compute the minimum and maximum fitness scores of
    /// the population in a single pass.
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

    /// Prune the population using the given `selection_strategy`. The
    /// strategy should take a float between 0-1 representing the percentile of the
    /// fitness score associated (higher is better), and return
    /// a bool representing whether or not to retain the gene associated with this score
    /// in the population.
    pub fn prune<R>(&mut self, mut selection_strategy: impl FnMut(f32, &mut R) -> bool, rng: &mut R)
    where
        R: RandomSource,
    {
        let (min_score, max_score) = self.score_bounds().unwrap();
        let score_range = max_score - min_score;
        self.gene_pool.retain(|(_, score)| {
            let Some(score) = score else {
                return true;
            };
            let percentile = (*score - min_score) / score_range;
            (selection_strategy)(percentile, rng)
        });
    }

    /// Reproduce new children into the population up to the population cap.
    /// Chooses parent genes preferentially from the population weighted by fitness
    /// score.
    /// Requires a [`std::random::RandomSource`] used to perform
    /// mutations on the new children.
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

    /// Perform one iteration of the genetic evolution process:
    /// 1. increment the generation
    /// 2. evaluate the fitness for all genes in the current population
    /// 3. compute statistics about the current population's fitness scores, if necessary
    /// 4. prune the population using the passed `selection_strategy` (see [`StochasticTrainer::prune`] for a description of what the selection strategy is)
    /// 5. reproduce the population to the current cap using the passed [`std::random::RandomSource`]
    /// 6. generate a report about the current generation's statistics, if necessary
    /// This function is used internally by the training process, should
    /// typically not be called directly unless the user knows what they're doing.
    pub fn step<R>(
        &mut self,
        selection_strategy: impl FnMut(f32, &mut R) -> bool,
        mut reporting_strategy: Option<
            &mut TrainingReportStrategy<
                impl FnMut(StochasticTrainingCriteriaMetrics) -> bool,
                impl FnMut(StochasticTrainingStats),
            >,
        >,
        rng: &mut R,
    ) where
        G: Clone + Genome,
        R: RandomSource,
    {
        self.generation += 1;
        self.eval();

        let stats = reporting_strategy
            .as_mut()
            .and_then(|s| (s.should_report)(self.metrics()).then(|| self.stats()));
        self.prune(selection_strategy, rng);
        self.reproduce(rng);
        if let (Some(stats), Some(reporting_strategy)) = (stats, &mut reporting_strategy) {
            (reporting_strategy.report_callback)(stats);
        }
    }

    /// Begin training, finishing once the number of generations passed
    /// have been reached.
    /// A [`std::random::RandomSource`] must be passed as a source of randomness
    /// for mutating genes to produce new offspring, and to be passed to the selection strategy when
    /// deciding to prune a given gene from the gene pool.
    pub fn train<R>(&mut self, generations: usize, rng: &mut R) -> G
    where
        G: Clone + Genome,
        R: RandomSource,
    {
        self.train_custom(
            |score, rng| score > random_f32(rng),
            |m| m.generation <= generations,
            Some(default_report_strategy()),
            rng,
        )
    }

    /// Begin training with more detailed custom parameters.
    /// A manual `selection_strategy` can be passed, which determines which genes to prune.
    /// (see [`StochasticTrainer::prune`] for a description of what a selection strategy is)
    /// 
    /// Instead of a specific generation count cutoff point, a function `train_criteria` is passed in,
    /// which takes in an instance of [`StochasticTrainingCriteriaMetrics`] and outputs a [`bool`].
    /// This allows greater control over exactly what criteria to finish training under.
    ///
    /// Additionally, the user may pass a `reporting_strategy`, which determines the conditions
    /// and method under which periodic statistical reporting of the population is performed.
    /// Pass `None` to disable reporting entirely, otherwise pass `Some` with an instance of
    /// [`TrainingReportStrategy`] to define the two methods necessary to manage reporting.
    /// A [`std::random::RandomSource`] must be passed as a source of randomness
    /// for mutating genes to produce new offspring, and to be passed to the selection strategy when
    /// deciding to prune a given gene from the gene pool.
    pub fn train_custom<R>(
        &mut self,
        mut selection_strategy: impl FnMut(f32, &mut R) -> bool,
        mut training_criteria: impl FnMut(StochasticTrainingCriteriaMetrics) -> bool,
        mut reporting_strategy: Option<
            TrainingReportStrategy<
                impl FnMut(StochasticTrainingCriteriaMetrics) -> bool,
                impl FnMut(StochasticTrainingStats),
            >,
        >,
        rng: &mut R,
    ) -> G
    where
        G: Clone + Genome,
        R: RandomSource,
    {
        loop {
            self.step::<R>(&mut selection_strategy, reporting_strategy.as_mut(), rng);
            if !(training_criteria)(self.metrics()) {
                break;
            }
        }
        self.gene_pool
            .iter()
            .filter_map(|(g, score)| score.map(|s| (g, s)))
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap()
            .0
            .clone()
    }

    /// Generate population stats for the current state of this trainer.
    /// This function is called whenever the reporting strategy is asked
    /// to produce a report about the current population, but it may also be called
    /// manually here.
    pub fn stats(&self) -> StochasticTrainingStats {
        StochasticTrainingStats {
            population_stats: self.scores().collect(),
            generation: self.generation,
        }
    }

    /// Generate training criteria metrics for the current state of this trainer.
    /// This is a strict subset of the data available in an instance of [`StochasticTrainingStats`]
    /// returned from calling [`StochasticTrainer::stats`]. However, these
    /// metrics were chosen specifically for their computation efficiency, and thus can be
    /// re-evaluated frequently with minimal cost. These metrics are used both to determine
    /// whether or not to continue training, and whether or not to display a report about
    /// training progress.
    /// Unfortunately, due to the unsorted nature of the gene pool in efficient stochastic training,
    /// these statistics are much more minimal than the ones available in
    /// [`crate::continuous::ContinuousTrainingCriteriaMetrics`]
    pub fn metrics(&self) -> StochasticTrainingCriteriaMetrics {
        StochasticTrainingCriteriaMetrics {
            generation: self.generation,
        }
    }
}

/// A collection of relevant & quick to compute metrics that
/// can be used to inform whether or not to continue training.
/// Unfortunately, because the gene population isn't inherently
/// ordered like in [`crate::continuous::ContinuousTrainer`],
/// the only metric that can be computed quickly for this purpose
/// is the current `generation` of the training process.
pub struct StochasticTrainingCriteriaMetrics {
    /// Total number of generations that have elapsed.
    pub generation: usize,
}

/// A collection of statistics about the population as a whole
/// Relatively more expensive to compute than training metrics, so
/// should be computed infrequently.
#[derive(Clone, Copy, Debug)]
pub struct StochasticTrainingStats {
    /// A collection of standard population stats: see [`PopulationStats`]
    /// for more information
    pub population_stats: PopulationStats,

    /// Total number of generations that have elapsed.
    pub generation: usize,
}

impl fmt::Display for StochasticTrainingStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "generation {} {}",
            self.generation, self.population_stats
        )
    }
}

/// Returns a default reporting strategy which logs population
/// statistics to the console after every generation.
/// Used by [`StochasticTrainer::train`].
pub fn default_report_strategy() -> TrainingReportStrategy<
    impl FnMut(StochasticTrainingCriteriaMetrics) -> bool,
    impl FnMut(StochasticTrainingStats),
> {
    TrainingReportStrategy {
        should_report: |_| true,
        report_callback: |s| println!("{s}"),
    }
}
