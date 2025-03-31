# Class: `MatchFinder`

A class which uses optimization results to find real material matches in the Materials Project databases, using the Materials Project API.<br>
Built on the `Pydantic BaseModel`.

## Attributes

* `optimization_params`, type: `OptimizationParams` <br>
* `optimized_population`, type: `Population` <br>
* `ga_params`, type: `GeneticAlgorithmParams` <br>

---
## Class Methods

* `get_dict_of_best_designs`
    - Constructs a dictionary containing the best designs found.
    - **Returns:** best_designs_dict (dict)

* `get_material_matches`
    - Identifies materials in the MP database which match those recommended by the optimization.
    - **Args:** 
        + overall_bounds_dict (dict, optional)
        + consolidated_dict (dict, optional)
        + threshold (float, optional)
            - Should be between 0 and 1, by default 1
    - **Returns:** best_designs_dict (dict)
        + final_matching_materials (dict)
            - Keys are fake materials recommended by the genetic algorithm
            - Values are mp-ids of real materials

* `get_all_possible_vol_frac_combos`
    - Computes the optimal volume fractions of known materials.
    - Once real materials have been identified, we must calculate which volume fraction combinations are *best* for the composite.
    - **Args:** num_fractions (int)
    - **Returns:** all_vol_frac_combos (list)

* `generate_consolidated_dict`
    - Main function used to generate material property dictionary depending on user request
    - **Args:** 
        + overall_bounds_dict (dict)
            - - Dictionary of bounds that encompass all the search bounds individually defined for constituent materials.
    - **Returns:**
        + consolidated_dict (dict)
            - Keyed by constituent materials
            - Items are lists of dictionaries keyed by [`mp-ids`](https://docs.materialsproject.org/downloading-data/using-the-api/querying-data) with properties which match the bounds criteria

* `get_material_match_costs`
    - Evaluates the *real* candidate composites with the same cost function used for optimization.
    - **Args:** 
        + matches_dict (dict)
        + consolidated_dict (dict)
    - **Returns:**
        + plotly.graph_objects.Figure
            - A table of the matches and their costs as evaluated by
              the genetic algorithm
