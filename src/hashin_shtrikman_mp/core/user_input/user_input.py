from pydantic import BaseModel

from typing import Dict, Iterable, Any

from .mixture import Mixture
from .material import Material

class UserInput(BaseModel):
    """
    The definition for an optimization problem.

    This class allows the user to define an optimization problem in terms of the number
    of material components in the optimized composite (and the bounds on the properties
    for each of those components), and a list of optimized mixtures to produce from
    those individual components.
    """
    materials: list[Material]
    mixtures: list[Mixture]

    def build_dict(self) -> Dict:
        """Builds the desired dict structure"""
        result = {}
        for material in self.materials:
            result.update(material.custom_dict())

        for mixture in self.mixtures:
            result.update(mixture.custom_dict())

        return result

    def items(self) -> Iterable:
        """Support dict-like .items() method"""
        return self.build_dict().items()

    def keys(self) -> Iterable:
        """Support dict-like .keys() method"""
        return self.build_dict().keys()

    def values(self) -> Iterable:
        """Support dict-like .values() method"""
        return self.build_dict().values()

    def __len__(self) -> int:
        """"""
        return len(self.build_dict())

    def __iter__(self):
        return iter(self.build_dict().items())

    def __getitem__(self, key):
        return self.build_dict()[key]

    def __repr__(self) -> str:
        return str(self.build_dict())

    def __str__(self) -> str:
        return str(self.build_dict())

    def get(self, key: str, default=None) -> Any:
        """Dict-like get method implementation"""
        return self.build_dict().get(key, default)
