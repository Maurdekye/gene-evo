#![feature(random)]
#![feature(mpmc_channel)]
use std::{
    fmt,
    random::{Random, RandomSource},
    thread::available_parallelism,
};

pub mod continuous;
pub mod stochastic;

pub trait Genome {
    /// Generate a new instance of this genome from the given random source.
    fn generate<R: RandomSource>(rng: &mut R) -> Self;

    /// Mutate this genome by an amount using the given random source.
    /// The [`radical`] parameter should influence how much mutation to
    /// perform on the genome, with 0 meaning no mutation, ie. the genome is unchanged.
    fn mutate<R: RandomSource>(&mut self, radical: f32, rng: &mut R);

    /// Evaluate this genome for its 'fitness' score. Higher fitness
    /// scores will lead to a higher survival rate.
    fn fitness(&self) -> f32;
}

pub trait SelectionStrategy {
    /// A strategy to determine if a given fitness percentile from 0-1 should
    /// be selected to proceed to the next epoch, or if it should be
    /// removed from the gene pool
    fn select<R: RandomSource>(percentile: f32, rng: &mut R) -> bool;
}

pub struct Top50th;

impl SelectionStrategy for Top50th {
    fn select<R: RandomSource>(percentile: f32, _rng: &mut R) -> bool {
        percentile > 0.5
    }
}

pub struct LinearLikelihood;

impl SelectionStrategy for LinearLikelihood {
    fn select<R: RandomSource>(percentile: f32, rng: &mut R) -> bool {
        percentile > random_f32(rng)
    }
}

pub struct PopulationStats {
    min_fitness: f32,
    max_fitness: f32,
    mean_fitness: f32,
    median_fitness: f32,
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
    let total: f32 = weights.iter().map(|x| x.1).sum();
    let mut n = random_f32(rng) * total;
    for (value, weight) in weights {
        if n <= *weight {
            return value;
        }
        n -= weight;
    }
    unreachable!("Should always eventually choose an item")
}
