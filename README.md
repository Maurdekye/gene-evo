# Genetic algorithm evolution

Three fully parallel [genetic algorithm](https://www.mathworks.com/help/gads/what-is-the-genetic-algorithm.html) implementations in Rust, with zero external dependencies.

## Overview

Comes with three genetic evolution drivers: [`StochasticTrainer`], [`ContinuousTrainer`], and [`InertialTrainer`].

* [`StochasticTrainer`] trains using the typical discrete generational strategy: a random
    population of genomes is seeded to begin the process. They are individually evaluated for fitness, and then based on a given selection strategy, are either retained for the next generation or culled. The remaining genes reproduce using either crossover or minor mutations to determine the population of the next generation.
* [`ContinuousTrainer`] works using a more novel strategy: evolution occurs not in discrete 
    steps, but instead continuously, constantly, without stopping. All genes are ranked from least to most fit in the population. At all times, random genes from the population weighted by their fitness are selected and chosen to be mutated / crossbred, modified, and re-inserted back into the population in their appropriate ranking based on their new fitness level. This continues constantly without stopping, until a set number of new children have been repopulated.
* [`InertialTrainer`] is a modification of the [`ContinuousTrainer`] that uses an even more novel algorithm. 
    It borrows concepts from the widely used gradient descent algorithm, common in modern machine learning. Instead of taking random walk steps within the latent space of possible genes, the trainer operates on mutation vectors, accruing velocity in the direction of greatest improvement in fitness, and allowing that velocity to contribute to the gene's learning progress. It does not perform crossover, and instead uses this inertial mutation system as its sole form of evolution. Because of its unique requirements and implementation, this trainer also requires that your genes implement the unique [`InertialGenome`] trait, which has different specifications thant the standard [`Genome`] trait the two primary trainers use.

## Usage

Using the trainers is simple. First, implement the [`Genome`] trait with your custom type (or the [`InertialGenome`] trait if you're using the [`InertialTrainer`]). Then, construct a `new` trainer, and finally run its `train` method with the appropriate parameters. Training will begin automatically, and end once the ending criteria have been met. The trainers will report statistics about the population to the console on a periodic basis. Ending criteria, as well as reporting frequency, conditions, and methods, can all be configured in more detail using the `train_custom` method available on each trainer.

See `tests/` for an example of how to use the trainers in this library.