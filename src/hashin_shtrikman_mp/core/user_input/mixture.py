"""mixtures.py."""
from pydantic import BaseModel
from .mixture_property import MixtureProperty


class Mixture(BaseModel):
    """Represents an optimal mixture defined by a list of desired properties as MixturePropertys."""

    name: str
    properties: list[MixtureProperty]

    def custom_dict(self) -> dict:
        """Transforms the default Pydantic dict to the desired format"""
        return {
            self.name: {
                p.prop: {"desired_prop": p.desired_prop} for p in self.properties
            }
        }
