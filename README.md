# Genetic algorithm evolution

Comes with two drivers: [`continuous::ContinuousTrainer`] and [`stochastic::StochasticTrainer`]
[`stochastic::StochasticTrainer`] trains using the typical discrete generational strategy: a random
population of genomes is seeded to begin the process. They are individually evaluated for fitness,
and then based on a given selection strategy, are either retained for the next generation or die off.
The remaining genes reproduce with minor mutations to determine the population of the next generation.
[`continuous::ContinuousTrainer`] works using a more novel strategy: evolution occurs not in discrete
steps, but instead continuously, constantly, without stopping. All genes are ranked from least to most
fit, and at all times, random genes from the population, weighted by their fitness, are selected and chosen to
be mutated, modified, and re-inserted back into the population in their appropriate ranking based on their
new fitness level. This continues constantly without stopping, until a set number of new children have
been repopulated.

Using the trainers is simple. Simply implement the [`Genome`] trait with your custom type, construct a `new`
trainer, and then run `train` with the appropriate parameters. Training will begin automatically, and end
once the ending criteria have been met. By default, the stochastic trainer will print out statistics about the population
once per generation, and the continuous trainer will print those statistics once every n children have been reproduced,
where n is the overall population size.

See `tests/` for examples of how to use the trainers in this library.