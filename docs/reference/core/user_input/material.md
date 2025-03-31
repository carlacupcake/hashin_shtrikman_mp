# Class: `Material`

Represents a material with a list of property constraints represented by `MaterialProperty`s. <br>
Built on the `Pydantic BaseModel`.

## Attributes

* `name`, type: `str` <br>
* `properties`, type: `list[MaterialProperty]
` <br>

---
## Class Methods

* `get_custom_dict`
    - Transforms the default Pydantic dict to the desired format
    - **Returns:**
        + dict
