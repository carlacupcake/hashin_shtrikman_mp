"""material_property.py."""
from pydantic import BaseModel


class MaterialProperty(BaseModel):
    """Represents a single property and its bounds."""

    prop: str
    upper_bound: float
    lower_bound: float
