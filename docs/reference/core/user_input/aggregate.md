# Class: `Aggregate`

Represents an aggregate of materials and mixtures. <br>
Needed in order to construct the dictionary of bounds that encompass all the search bounds individually defined for constituent materials. <br>
Built on the `Pydantic BaseModel`.

## Attributes

* `name`, type: `str` <br>
* `components`, type: `list[Material | Mixture]` <br>

---
## Class Methods

* `get_bounds_dict`
    - Computes the overall upper and lower bounds for each property.
    - **Returns:**
        + bounds_dict (dict)
