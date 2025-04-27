# Genetic algorithm evolution

Two fully parallel [genetic algorithm](https://www.mathworks.com/help/gads/what-is-the-genetic-algorithm.html) implementations in Rust, with zero external dependencies. The algorithm only performs mutation based exploration, and does not utilize any crossover.

Comes with two genetic evolution drivers: [`continuous::ContinuousTrainer`] and [`stochastic::StochasticTrainer`]

* [`stochastic::StochasticTrainer`] trains using the typical discrete generational strategy: a random
    population of genomes is seeded to begin the process. They are individually evaluated for fitness,
    and then based on a given selection strategy, are either retained for the next generation or culled.
    The remaining genes reproduce with minor mutations to determine the population of the next generation.
* [`continuous::ContinuousTrainer`] works using a more novel strategy: evolution occurs not in discrete
    steps, but instead continuously, constantly, without stopping. All genes are ranked from least to most
    fit in the population. At all times, random genes from the population weighted by their fitness are selected and chosen to be mutated, modified, and re-inserted back into the population in their appropriate ranking based on their new fitness level. This continues constantly without stopping, until a set number of new children have
    been repopulated.

Using the trainers is simple. First, implement the [`Genome`] trait with your custom type. Then, construct a `new`
trainer, and finally run its `train` method with the appropriate parameters. Training will begin automatically, and end
once the ending criteria have been met. The trainers will report statistics about the population to the console on a periodic basis. Ending criteria, as well as reporting frequency, conditions, and methods, can all be configured in more detail using the `train_custom` method available on each trainer.

See `tests/` for an example of how to use the trainers in this library.