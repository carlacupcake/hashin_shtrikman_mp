"""user_input.py."""
from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel

from .material import Material
from .mixture import Mixture


class UserInput(BaseModel):
    """
    Class to store user input needed to run an optimal composite optimization study.

    This class allows the user to define an optimization problem in terms of the number
    of material components in the optimized composite (and the bounds on the properties
    for each of those components), and a list of optimized mixtures to produce from
    those individual components.
    """

    materials: list[Material]
    mixtures: list[Mixture]


    def build_dict(self) -> dict:
        """
        Builds the desired dictionary structure from the materials and mixtures.
        """
        result = {}
        for material in self.materials:
            result.update(material.custom_dict())

        for mixture in self.mixtures:
            result.update(mixture.custom_dict())

        return result


    def items(self) -> Iterable:
        """
        Support dict-like .items() method for iteration.
        """
        return self.build_dict().items()


    def keys(self) -> Iterable:
        """
        Support dict-like .keys() method for retrieving keys.
        """
        return self.build_dict().keys()


    def values(self) -> Iterable:
        """
        Support dict-like .values() method for retrieving values.
        """
        return self.build_dict().values()


    def __len__(self) -> int:
        """
        Returns the length of the dictionary representation of the object.
        """
        return len(self.build_dict())


    def __iter__(self) -> Iterable:
        """
        Returns an iterator over the items of the dictionary representation of the object.

        This method allows for iteration over key-value pairs
        in the dictionary representation of the object.
        """
        return iter(self.build_dict().items())


    def __getitem__(self, key) -> Any:
        """
        Accesses an item from the dictionary representation of the object using the provided key.
        """
        return self.build_dict()[key]


    def __repr__(self) -> str:
        """
        Returns a string representation of the UserInput object.
        """
        return str(self.build_dict())


    def __str__(self) -> str:
        """
        Returns a string version of the UserInput object.
        """
        return str(self.build_dict())


    def get(self, key: str, default=None) -> Any:
        """
        Retrieves a value from the dictionary representation of the object for the given key.
        """
        return self.build_dict().get(key, default)
