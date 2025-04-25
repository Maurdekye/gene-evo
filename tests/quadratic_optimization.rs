//! simple tests that determine the roots of a simple quadratic equation
#![feature(random)]

use std::{
    array,
    random::{DefaultRandomSource, Random, RandomSource},
    thread::scope,
};

use gene_evo::{
    Genome,
    continuous::ContinuousTrainer,
    stochastic::{LinearLikelihood, StochasticTrainer},
};

#[derive(Debug, Clone)]
pub struct PsuedoRandomSource {
    state: u64,
}

impl PsuedoRandomSource {
    pub fn new() -> Self {
        let mut seed_bytes = [0; 8];
        DefaultRandomSource.fill_bytes(&mut seed_bytes);
        let state = u64::from_ne_bytes(seed_bytes);
        let state = if state == 0 {
            0x9e3779b97f4a7c15
        } else {
            state
        };

        PsuedoRandomSource::new_from_seed(state)
    }

    pub fn new_from_seed(seed: u64) -> Self {
        PsuedoRandomSource { state: seed }
    }

    /// use xorshift algorithm for psuedorandom generation
    #[inline]
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
}

impl RandomSource for PsuedoRandomSource {
    fn fill_bytes(&mut self, bytes: &mut [u8]) {
        for i in (0..bytes.len()).step_by(8) {
            let rand_bytes = self.next_u64().to_ne_bytes();
            let end = (i + 8).min(bytes.len());
            bytes[i..end].copy_from_slice(&rand_bytes[..end - i]);
        }
    }
}

fn random_f32<R>(rng: &mut R) -> f32
where
    R: RandomSource,
{
    u32::random(rng) as f32 / u32::MAX as f32
}

fn hump(x: f32) -> f32 {
    if x.abs() > 10.0 {
        0.0
    } else {
        2.0 / (x.exp() + (-x).exp())
    }
}

#[derive(Clone, Debug)]
struct QuadraticZerosFinder {
    zeroes: [f32; 4],
}

impl Genome for QuadraticZerosFinder {
    fn generate<R: RandomSource>(rng: &mut R) -> Self {
        Self {
            zeroes: array::from_fn(|_| (random_f32(rng) - 0.5) * 20.0),
        }
    }

    fn mutate<R: RandomSource>(&mut self, mutation_rate: f32, rng: &mut R) {
        self.zeroes = self
            .zeroes
            .map(|z| z + (random_f32(rng) - 0.5) * mutation_rate);
    }

    fn fitness(&self) -> f32 {
        fn quadratic(x: f32) -> f32 {
            x.powi(4) + 4.0 * x.powi(3) - 7.0 * x.powi(2) - 22.0 * x + 24.0
        }

        let closeness_to_zero_score: f32 = self
            .zeroes
            .iter()
            .map(|&z| hump(quadratic(z).abs()))
            .sum();

        let proximity_penalty: f32 = self
            .zeroes
            .iter()
            .enumerate()
            .map(|(i, z)| {
                self.zeroes
                    .iter()
                    .enumerate()
                    .filter_map(|(j, &w)| (i != j).then_some(w))
                    .map(|w| hump((z - w).abs() * 10.0))
                    .sum::<f32>()
            })
            .sum();

        closeness_to_zero_score - proximity_penalty
    }
}

#[test]
fn check_zeros_finder_validity() {
    let gene = QuadraticZerosFinder::generate(&mut DefaultRandomSource);
    println!("Gene: {gene:?}");
    println!("Fitness: {}", gene.fitness());
}

#[test]
fn stochastic() {
    scope(|scope| {
        let mut rng = PsuedoRandomSource::new();
        let mut trainer = StochasticTrainer::new(50000, 0.05, &mut rng, scope);
        let final_genome: QuadraticZerosFinder =
            trainer.train::<LinearLikelihood, _>(100, &mut rng);
        println!("Final genome: {final_genome:?}");
        println!("Fitness: {}", final_genome.fitness());
    });
}

#[test]
fn continuous() {
    scope(|scope| {
        let mut trainer = ContinuousTrainer::new(10000, 0.05, scope);
        let final_genome: QuadraticZerosFinder = trainer.train(
            |m| m.child_count < 100000,
            &mut PsuedoRandomSource::new(),
        );
        println!("Final genome: {final_genome:?}");
        println!("Fitness: {}", final_genome.fitness());
    });
}
