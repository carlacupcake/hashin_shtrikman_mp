"""material_property.py."""
from pydantic import BaseModel, model_validator


class MaterialProperty(BaseModel):
    """Represents a single property for a single material and its bounds."""

    prop: str
    upper_bound: float
    lower_bound: float

    @model_validator(mode="after")
    def check_bounds(self):
        if self.upper_bound <= self.lower_bound:
            raise ValueError("upper_bound must be greater than lower_bound")
        return self
