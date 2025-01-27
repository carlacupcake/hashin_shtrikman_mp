"""mixture_property.py."""
import sys
import yaml

from pathlib import Path
from pydantic import BaseModel


# Load valid properties from YAML file
sys.path.insert(1, "../io/inputs/")
PROPERTY_DOCS_YAML = "mp_property_docs.yaml"
MODULE_DIR = Path(__file__).resolve().parent
file_name = MODULE_DIR.joinpath("../../io/inputs/data", PROPERTY_DOCS_YAML).resolve()

with open(file_name) as stream:
    data = yaml.safe_load(stream)

# Flatten the nested YAML structure into a single list of valid property names
valid_properties = []
for category, props in data.items():
    if isinstance(props, dict):
        for sub_prop in props:
            valid_properties.append(sub_prop)
    else:
        valid_properties.append(category)


class MixtureProperty(BaseModel):
    """
    Represents a target property for an optimized mixture.
    """

    prop: str
    desired_prop: float

    def validate_prop(self, value):
        if value not in valid_properties:
            raise ValueError(f"Invalid prop: {value}. Must be one of {valid_properties}")
        return value
