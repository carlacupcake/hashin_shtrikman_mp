# Class: `OptimizationParams`

Hashin-Shtrikman optimization class.

Class to integrate Hashin-Shtrikman (HS) bounds with a genetic algorithm
and find optimal material properties for each composite constituent to achieve
desired properties.

Built on the `Pydantic BaseModel`.

## Attributes

* `property_categories`, `list[str]` <br>
    List of property categories considered for optimization.

* `property_docs`, `dict[str, dict[str, Any]]` <br>
    A hard coded yaml file containing property categories and their individual properties.

* `lower_bounds`, `dict[str, Any]` <br>
    Lower bounds for properties of materials considered in the optimization.

* `upper_bounds`, `dict[str, Any]` <br>
    Upper bounds for properties of materials considered in the optimization.

* `desired_props`, `dict[str, list[float]]` <br>
    Dictionary mapping individual properties to their desired properties.
    
* `num_materials`, `int` <br>
    Number of materials to comprise the composite.

* `num_properties`, `int` <br>
    Number of properties being optimized.

* `indices_elastic_moduli`, `list[Any]` <br>
    For handling coupling between bulk & shear moduli. List of length 2, first element is index of bulk modulus in list version of the bounds, second element is the shear modulus.

---
## Class Methods

* `from_user_input`
    - Initializes the `OptimizationParams` class from the user input dictionary.

* `get_num_properties_from_desired_props`
    - Calculates the total number of properties from the desired properties dictionary.
    - **Args:** desired_props (dict)
    - **Returns:** num_properties (int)

* `get_bounds_from_user_input`
    - Extracts bounds (upper or lower) for material properties from the user input.
    - **Args:**
        + user_input (dict)
        + bound_key (str)
        + property_docs (dict[str, list[str]])
        + num_materials (int)
    - **Returns:**
        + bounds (dict)

* `get_desired_props_from_user_input`
    - Extracts the desired values for each property category.
    - **Args:**
        + user_input (dict)
        + property_categories (list[str])
        + property_docs (dict)
    - **Returns:**
        + desired_props (dict)

* `get_elastic_idx_from_user_input`
    - Gets the indices of elastic properties from the upper bounds.
    - **Args:**
        + upper_bounds (dict)
        + property_categories (list[str])
    - Returns:
        + indices (list[int]): ensures proper ordering of elastic moduli
