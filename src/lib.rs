#![doc = include_str!("../README.md")]
#![feature(random)]
#![feature(mpmc_channel)]

use std::{
    fmt,
    random::{Random, RandomSource},
    sync::{Arc, Condvar, Mutex},
    thread::available_parallelism,
};

pub mod continuous;
pub mod stochastic;

/// Represents a single Genome in a genetic evolution simulation.
/// Implement this trait for your type to evolve it using genetic evolution.
pub trait Genome {
    /// Generate a new instance of this genome from the given random source.
    fn generate<R: RandomSource>(rng: &mut R) -> Self;

    /// Mutate this genome by an amount using the given random source.
    /// The `mutation_rate` parameter should influence how much mutation to
    /// perform on the genome, with 0 meaning no mutation, ie. the genome is unchanged.
    fn mutate<R: RandomSource>(&mut self, mutation_rate: f32, rng: &mut R);

    /// Evaluate this genome for its 'fitness' score. Higher fitness
    /// scores will lead to a higher survival rate.
    fn fitness(&self) -> f32;
}

/// A collection of standard population statistics that can be used
/// for progress reporting.
#[derive(Clone, Copy, Debug)]
pub struct PopulationStats {
    
    /// Maximum fitness of the population.
    pub max_fitness: f32,
    
    /// Minimum fitness of the population.
    pub min_fitness: f32,
    
    /// Median fitness of the population.
    pub mean_fitness: f32,

    /// Median fitness of the population.
    pub median_fitness: f32,
}

impl FromIterator<f32> for PopulationStats {
    fn from_iter<T: IntoIterator<Item = f32>>(iter: T) -> Self {
        let mut scores: Vec<_> = iter.into_iter().collect();
        scores.sort_by(|a, b| a.total_cmp(b));
        let &min_fitness = scores.first().unwrap();
        let &max_fitness = scores.last().unwrap();
        let mean_fitness = scores.iter().sum::<f32>() / scores.len() as f32;
        let median_fitness = scores[scores.len() / 2];
        PopulationStats {
            min_fitness,
            max_fitness,
            mean_fitness,
            median_fitness,
        }
    }
}

impl fmt::Display for PopulationStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "min={:.4} max={:.4} mean={:.4} median={:.4}",
            self.min_fitness, self.max_fitness, self.mean_fitness, self.median_fitness
        )
    }
}

/// A struct that encompasses a two-part reporting
/// strategy to use when performing periodic progress
/// updates. [`TrainingReportStrategy::should_report`]
/// represents whether or not a report should be generated
/// at this moment, and [`TrainingReportStrategy::report_callback`]
/// performs the actual reporting.
pub struct TrainingReportStrategy<F, C> {
    /// A callback used to determine if a report
    /// should be generated at this moment.
    pub should_report: F,

    /// A callback used to when a report should be displayed 
    /// or logged in some form. This callback is only called if
    /// [`TrainingReportStrategy::should_report`] returns `true`.
    pub report_callback: C,
}

fn random_f32<R>(rng: &mut R) -> f32
where
    R: RandomSource,
{
    u32::random(rng) as f32 / u32::MAX as f32
}

fn num_cpus() -> usize {
    available_parallelism().map(usize::from).unwrap_or(1)
}

fn random_choice_weighted<'a, T, R>(weights: &'a [(T, f32)], rng: &mut R) -> &'a T
where
    R: RandomSource,
{
    random_choice_weighted_mapped(weights, rng, |x| x)
}

fn random_choice_weighted_mapped<'a, T, R>(
    weights: &'a [(T, f32)],
    rng: &mut R,
    weight_map: impl Fn(f32) -> f32,
) -> &'a T
where
    R: RandomSource,
{
    let total: f32 = weights.iter().map(|x| (weight_map)(x.1)).sum();
    let mut n = random_f32(rng) * total;
    for (value, weight) in weights {
        let weight = (weight_map)(*weight);
        if n <= weight {
            return value;
        }
        n -= weight;
    }
    &weights.last().unwrap().0
}

#[derive(Clone)]
struct Gate<S>(Arc<(Mutex<S>, Condvar)>);

impl<S> Gate<S> {
    fn new(initial_state: S) -> Self {
        Self(Arc::new((Mutex::new(initial_state), Condvar::new())))
    }

    fn update(&self, updater: impl Fn(&mut S)) {
        let mut state = self.0.0.lock().unwrap();
        (updater)(&mut state);
        self.0.1.notify_all();
    }

    fn wait_while(&self, condition: impl Fn(&S) -> bool) {
        let mut state = self.0.0.lock().unwrap();
        while (condition)(&state) {
            state = self.0.1.wait(state).unwrap();
        }
    }
}
