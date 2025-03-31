# Class: `Population`

Class to hold the population of members.<br>
The class also implements methods to generate the initial population and sort the members based on their costs. <br>
Built on the `Pydantic BaseModel`.

## Attributes

* `optimization_params`, type: `GeneticAlgorithmParams` <br>
* `ga_params`, type: `PositiveInt` <br>
* `values`, type: `np.ndarray = None` <br>
* `costs`, type: `np.ndarray = None)` <br>

---
## Class Methods

* `get_unique_designs`
    - Retrieves the unique designs from the population based on their costs.
    - **Returns:**
        + A list containing:
            - unique_members (ndarray), unique population members corresponding to unique_costs.
            - uniqie_costs (ndarray)

* `get_effective_properties`
    - Calculates the effective properties for each member in the population.
    - **Returns:**
        + all_effective_properties (ndarray)

* `set_random_values`
    - Sets random values for the properties of each member in the population.
    - **Args:**
        + lower_bounds (ndarray, optional)
        + upper_bounds (ndarray, optional)
        + start_member (int, optional): index of the first member to update in the population
        + indices_elastic_moduli (list, optional): ensures proper ordering.
    - **Returns:**
        + self (Population)

* `set_costs`
    - Calculates the costs for each member in the population.
    - **Returns:**
        + self (Population)

* `set_order_by_costs`
    - Reorders the population based on the sorted indices of costs.
    - **Args:** sorted_indices (ndarray)
    - **Returns:**
        + self (Population)

* `sort_costs`
    - Sorts the costs and returns the sorted values along with their corresponding indices.
    - **Returns:**
        + A list containing two arrays:
            - sorted costs (ndarray)
            - sorted_indices (ndarray), indices that would sort the original `self.costs`.
