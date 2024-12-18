
from pydantic import BaseModel


class MaterialProperty(BaseModel):
    prop: str
    upper_bound: float
    lower_bound: float

class Material(BaseModel):
    name: str
    properties: list[MaterialProperty]

    def custom_dict(self):
        # Custom method to transform the default Pydantic dict to the desired format
        return {
            self.name: {
                p.prop: {"upper_bound": p.upper_bound, "lower_bound": p.lower_bound}
                for p in self.properties
            }
        }

class MixtureProperty(BaseModel):
    prop: str
    desired_prop: float

class Mixture(BaseModel):
    name: str
    properties: list[MixtureProperty]

    def custom_dict(self):
        # Custom method to transform the default Pydantic dict to the desired format
        return {
            self.name: {
                p.prop: {"desired_prop": p.desired_prop} for p in self.properties
            }
        }

class UserInput(BaseModel):
    materials: list[Material]
    mixtures: list[Mixture]

    def build_dict(self):
        # Builds the desired dict structure
        result = {}
        for material in self.materials:
            result.update(material.custom_dict())

        for mixture in self.mixtures:
            result.update(mixture.custom_dict())

        return result

    def items(self):
        # Support dict-like .items() method
        return self.build_dict().items()

    def keys(self):
        # Support dict-like .keys() method
        return self.build_dict().keys()

    def values(self):
        # Support dict-like .values() method
        return self.build_dict().values()

    def len(self):
        print(f"self.nuild_dict: {self.build_dict}")
        return len(self.build_dict())

    def __iter__(self):
        # Support dict-like iteration over key-value pairs
        return iter(self.build_dict().items())

    def __getitem__(self, key):
        return self.build_dict()[key]

    def __repr__(self) -> str:
        return str(self.build_dict())

    def __str__(self) -> str:
        return str(self.build_dict())

    def get(self, key, default=None):
        # Dict-like get method implementation
        return self.build_dict().get(key, default)
