# Class: `Mixture`

Represents an optimal mixture defined by a list of desired properties as `MixturePropertys`. <br>
Built on the `Pydantic BaseModel`.

## Attributes

* `name`, type: `str` <br>
* `properties`, type: `list[MixtureProperty]` <br>

---
## Class Methods

* `get_custom_dict`
    - Transforms the default Pydantic dict to the desired format
    - **Returns:**
        + dict
