# Class: `GeneticAlgorithm`

A class used only for running the genetic algorithm optimization. <br>
Built on the `Pydantic BaseModel`.

## Attributes

None.

---
## Class Methods

* `run`
    - Executes the Genetic Algorithm (GA) optimization process.
    - Initializes a population, evaluates costs, and iteratively evolves the population through breeding and selection to minimize the cost function over multiple generations. The best and average costs for each generation are tracked, and the final population is returned alongside optimization results.
    - **Args:**
        + user_inputs (UserInput)
        + ga_algo_params (GeneticAlgorithmParams, optional)
        + gen_counter (bool, optional)
    - **Returns:**
        + GeneticAlgorithmResult
