# Class: `GeneticAlgorithmResult`

Represents the result of a genetic algorithm run.

## Attributes

* `algo_parameters`, type: `GeneticAlgorithmParams` <br>
* `final_population`, type: `Population` <br>
* `lowest_costs`, type: `ndarray` <br>
    - Lowest cost values across generations.
* `avg_parent_costs`, type: `ndarray` <br>
    - Average cost of the top-performing parents across generations.
* `optimization_params`, type: `OptimizationParams`