# Class: `OptimizationResultVisualizer`

Visualization class for plotting results of genetic algorithm optimization. <br>
Built on the `Pydantic BaseModel`.

## Attributes

* `ga_params`, type: `GeneticAlgorithmParams` <br>
* `opt_params`, type: `OptimizationParams` <br>
* `ga_result`, type: `GeneticAlgorithmResult` <br>

---
## Class Methods

* `get_table_of_best_designs`
    - Retrieves a table of the top-performing designs from the final population.
    - **Args:**
        + rows (int, optional)
            - The number of top designs to retrieve.
    - **Returns:** table_of_best_designs (ndarray)

* `print_table_of_best_designs`
    - Generates and displays a formatted table of the top-performing designs.
    - **Args:**
        + rows (int, optional)
            - The number of top designs to retrieve.
    - **Returns:** 
        + plotly.graph_objects.Figure
            - A table of the best designs with `rows` rows

* `plot_optimization_results`
    - Generates a plot visualizing the optimization convergence over generations.
    - **Returns:** 
        + plotly.graph_objects.Figure

* `plot_cost_func_contribs`
    - Generates a pie chart visualizing the contributions of different factors to the cost function for the best design found by the genetic algorithm.
    - **Returns:** 
        + plotly.graph_objects.Figure
