# Class: `CompositePropertyPlotter`

Visualization class for phase diagrams. <br>
Built on the `Pydantic BaseModel`.

## Attributes

* `ga_params`, type: `GeneticAlgorithmParams` <br>
* `opt_params`, type: `OptimizationParams` <br>
* `ga_result`, type: `GeneticAlgorithmResult` <br>

---
## Class Methods

* `get_all_possible_vol_frac_combos`
    - Generates all possible unique volume fraction combinations for a given number of materials, ensuring that the sum of fractions equals 1. 
    - **Args:** num_fractions (int, optional)
    - **Returns:** all_vol_frac_combos (list of list of float)

* `visualize_composite_eff_props`
- Generates visualizations of effective properties for composite materials. 
- **Args:**
    + match
        - A list of material identifiers used to construct the composite.
    + consolidated_dict (dict)
        - A dictionary containing material property values with keys as property names and values as lists of corresponding data.
    + num_fractions (int, optional)
        - The number of discrete volume fraction values to consider.
- **Returns:**
    + Nothing, but calls other methods which eventually output `plotly` figures
- **Notes:**
    + Does not support phase diagram visualization for single-phase or $\geq$5-phase composites.

* `visualize_composite_eff_props_2_phase`
    - Generates a 2D line plot to visualize the effective properties of a two-phase composite. 
    - **Args:**
        + match (list)
        + property (str)
        + units (str)
        + volume_fractions (ndarray)
        + effective_properties (ndarray)
    - **Returns:**
        + fig (plotly.graph_objects.Figure)

* `visualize_composite_eff_props_3_phase`
    - Generates a 3D surface plot to visualize the effective properties of a three-phase composite. 
    - **Args:**
        + match (list)
        + property (str)
        + units (str)
        + volume_fractions (ndarray)
        + effective_properties (ndarray)
    - **Returns:**
        + fig (plotly.graph_objects.Figure)

* `visualize_composite_eff_props_4_phase`
    - Generates a 3D scatter plot to visualize the effective properties of a four-phase composite. 
    - **Args:**
        + match (list)
        + property (str)
        + units (str)
        + volume_fractions (ndarray)
        + effective_properties (ndarray)
    - **Returns:**
        + fig (plotly.graph_objects.Figure)
