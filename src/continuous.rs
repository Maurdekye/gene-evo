//! Continuous Training.
//! See the documentation for [`ContinuousTrainer`] for more details.

use std::{
    fmt,
    ops::AddAssign,
    random::RandomSource,
    sync::{Arc, RwLock, mpmc, mpsc},
    thread::{self, ScopedJoinHandle},
};

use crate::{
    Gate, Genome, PopulationStats, TrainingReportStrategy, num_cpus, random_choice_weighted_mapped,
};

/// This is an implementation of a continuous training strategy, an "alternative" evolutionary
/// training strategy to the standard generation-based strategy. This trainer runs its training
/// continuously, nonstop, with no definable "break" in between generations. New genes are constantly
/// being reproduced, mutated, evaluated, and ranked while the trainer runs, using a multithreaded
/// pool of workers. As it's a more nonstandard training strategy, the allowed criteria for
/// determining when to print population reports and when to end training are more flexible than in the
/// typical evolutionary trainer, allowing the user to define exactly what criteria they want to
/// pay attention to during the training process. You can access these more detailed and complex
/// controls by calling the [`ContinuousTrainer::train_custom`] function. If you would like a simpler
/// interface with simple, default training criteria and reporting, just call [`ContinuousTrainer::train`].
pub struct ContinuousTrainer<'scope, G> {
    /// A collection of all the genes in the population and
    /// their fitness score, sorted descending by fitness.
    /// Because of the continuous nature of the trainer,
    /// the collection is behind an `Arc<RwLock<>>` combo.
    pub gene_pool: Arc<RwLock<Vec<(G, f32)>>>,

    /// The mutation rate of newly reproduced children.
    pub mutation_rate: f32,

    /// Count of the total number of children reproduced.
    pub children_created: usize,
    work_submission: mpmc::Sender<G>,
    #[allow(unused)]
    worker_pool: Vec<ScopedJoinHandle<'scope, ()>>,
    #[allow(unused)]
    receiver_thread: ScopedJoinHandle<'scope, ()>,
    population_size: usize,
    in_flight: Gate<usize>,
}

impl<'scope, G> ContinuousTrainer<'scope, G> {
    /// Construct a new trainer with a given population size and mutation rate.
    /// A reference to a [`thread::Scope`] must be passed as well in order
    /// to spawn the child worker threads for the lifetime of the trainer.
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
            children_created: 0,
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

    /// Submit a new genome to the worker pool to be evaluated for its fitness and
    /// ranked among the population. Used internally by the training process, should
    /// typically not be called directly unless the users knows what they're doing.
    pub fn submit_job(&mut self, gene: G) {
        self.children_created += 1;
        self.in_flight.update(|x| x.add_assign(1));
        self.work_submission.send(gene).unwrap();
    }

    /// Seed the population with new genes up to the current population cap.
    /// Is called automatically at the start of training, so should typically
    /// not need to be called directly.
    /// A [`std::random::RandomSource`] must be passed as a source of randomness
    /// for generating the initial population.
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

    /// Begin training, finishing once `num_children` children have been
    /// bred and reproduced.
    /// A [`std::random::RandomSource`] must be passed as a source of randomness
    /// for mutating genes to produce new offspring.
    pub fn train<R>(&mut self, num_children: usize, rng: &mut R) -> G
    where
        R: RandomSource,
        G: Clone + Genome + Send + Sync + 'scope,
    {
        self.train_custom(
            |x| x.child_count <= num_children,
            Some(default_reporting_strategy(self.population_size)),
            rng,
        )
    }

    /// Begin training with detailed custom parameters. Instead of a specific child
    /// count cutoff point, a function `train_criteria` is passed in, which takes in an
    /// instance of [`ContinuousTrainingCriteriaMetrics`] and outputs a [`bool`]. This allows greater
    /// control over exactly what criteria to finish training under.
    ///
    /// Additionally, the user may pass a `reporting_strategy`, which determines the conditions
    /// and method under which periodic statistical reporting of the population is performed.
    /// Pass `None` to disable reporting entirely, otherwise pass `Some` with an instance of
    /// [`TrainingReportStrategy`] to define the two methods necessary to manage reporting.
    /// A [`std::random::RandomSource`] must be passed as a source of randomness
    /// for mutating genes to produce new offspring.
    pub fn train_custom<R>(
        &mut self,
        mut train_criteria: impl FnMut(ContinuousTrainingCriteriaMetrics) -> bool,
        mut reporting_strategy: Option<
            TrainingReportStrategy<
                impl FnMut(ContinuousTrainingCriteriaMetrics) -> bool,
                impl FnMut(ContinuousTrainingStats),
            >,
        >,
        rng: &mut R,
    ) -> G
    where
        R: RandomSource,
        G: Clone + Genome + Send + Sync + 'scope,
    {
        self.seed(rng);
        loop {
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
        self.gene_pool.read().unwrap().first().unwrap().0.clone()
    }

    /// Generate training criteria metrics for the current state of this trainer.
    /// This is a strict subset of the data available in an instance of [`ContinuousTrainingStats`]
    /// returned from calling [`ContinuousTrainer::stats`]. However, these
    /// metrics were chosen specifically for their computation efficiency, and thus can be
    /// re-evaluated frequently with minimal cost. These metrics are used both to determine
    /// whether or not to continue training, and whether or not to display a report about
    /// training progress.
    pub fn metrics(&self) -> ContinuousTrainingCriteriaMetrics {
        let gene_pool = self.gene_pool.read().unwrap();
        ContinuousTrainingCriteriaMetrics {
            max_fitness: gene_pool.first().unwrap().1,
            min_fitness: gene_pool.last().unwrap().1,
            median_fitness: gene_pool[gene_pool.len() / 2].1,
            child_count: self.children_created,
        }
    }

    /// Generate population stats for the current state of this trainer.
    /// This function is called whenever the reporting strategy is asked
    /// to produce a report about the current population, but it may also be called
    /// manually here.
    pub fn stats(&self) -> ContinuousTrainingStats {
        ContinuousTrainingStats {
            population_stats: self.gene_pool.read().unwrap().iter().map(|x| x.1).collect(),
            child_count: self.children_created,
        }
    }
}

/// A collection of relevant & quick to compute metrics that
/// can be used to inform whether or not to continue training
#[derive(Clone, Copy, Debug)]
pub struct ContinuousTrainingCriteriaMetrics {
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

/// A collection of statistics about the population as a whole
/// Relatively more expensive to compute than training metrics, so
/// should be computed infrequently.
#[derive(Clone, Copy, Debug)]
pub struct ContinuousTrainingStats {
    /// A collection of standard population stats: see [`PopulationStats`]
    /// for more information
    pub population_stats: PopulationStats,

    /// Total number of children that have been
    /// reproduced and introduced into the population,
    /// including the initial seed population count.
    /// Same as [`ContinuousTrainingCriteriaMetrics::child_count`].
    pub child_count: usize,
}

impl fmt::Display for ContinuousTrainingStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "child #{} {}", self.child_count, self.population_stats)
    }
}

/// Returns a default reporting strategy which logs population
/// statistics to the console every `n` children reproduced.
/// Used by [`ContinuousTrainer::train`].
pub fn default_reporting_strategy(
    n: usize,
) -> TrainingReportStrategy<
    impl FnMut(ContinuousTrainingCriteriaMetrics) -> bool,
    impl FnMut(ContinuousTrainingStats),
> {
    TrainingReportStrategy {
        should_report: move |m: ContinuousTrainingCriteriaMetrics| m.child_count % n == 0,
        report_callback: |s| println!("{s}"),
    }
}
