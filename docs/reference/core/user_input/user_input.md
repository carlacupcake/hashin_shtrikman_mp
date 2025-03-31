# Class: `UserInput`

Class to store user input needed to run an optimal composite optimization study. <br>

The genetic algorithm needs two things from the user: <br>
    1) Search bounds for each property of interest for each constituent phase <br>
    2) Desired values for each property of interest
    
This class stores such information. <br>
Built on the `Pydantic BaseModel`.

## Attributes

* `materials`, type: `list[Material]` <br>
* `mixtures`, type: `list[ Mixture]` <br>

---
## Class Methods

* `build_dict`
    - Builds the desired dictionary structure from the materials and mixtures.
    - **Returns:**
        + dict
* Various other functions that provide similar support for the `UserInput` class as the `Dict` class may have, including
    - `items`
    - `keys`
    - `values`
    - `__len__`
    - `__iter__`
    - `__getitem__`
    - `__rep__`
    - `__str__`
    - `__get__`
