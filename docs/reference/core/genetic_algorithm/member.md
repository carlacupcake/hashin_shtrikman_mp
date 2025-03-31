# Class: `Member`

Class to represent a member of the population in genetic algorithm optimization. <br>
Stores the functions for cost function calculation. <br>
Built on the `Pydantic BaseModel`.

## Attributes

* `ga_params`, type: `GeneticAlgorithmParams` <br>
* `optimization_params`, type: `OptimizationParams` <br>
* `values`, type: `np.ndarray = None`
---
## Class Methods

* `get_cost`
    - Calculates the total cost for the current member based on effective properties and concentration factors, incorporating domain-specific weights and tolerances.
    - The cost function evaluates the deviation of effective properties and concentration factors from desired values, penalizing larger deviations while accounting for the relative importance of different property domains.
    - **Notes**
        + There is one effective property per property
        + (Non-modulus) There are two concentration factors per property per material
        + (Modulus) Bulk and shear moduli collectively have two concentration factors per material (instead of what would be four)
    - **Args:** include_cost_breakdown (bool, optional)
    - **Returns:** cost (float)

* `get_general_eff_prop_and_cfs`
    - Compute the effective non-modulus properties and concentration factors of a composite material using the Hashin-Shtrikman bounds.
    - **Notes**
        + `idx` is the index in self.values where category properties begin
    - **Args:** idx (int, optional)
    - **Returns:**
        + A tuple containing:
            - effective_properties (list)
            - concentration_factors (list)

* `get_elastic_eff_props_and_cfs`
    - Compute the effective modulus properties and concentration factors of a composite material using the Hashin-Shtrikman bounds.
    - **Notes**
        + `idx` is the index in self.values where category properties begin
    - **Args:** idx (int, optional)
    - **Returns:**
        + A tuple containing:
            - effective_properties (list)
            - concentration_factors (list)

* `get_effective_properties`
    - Computes the effective properties of a material system based on the Hashin-Shtrikman bounds for various property categories.
    - **Returns:** effective properties (ndarray)
