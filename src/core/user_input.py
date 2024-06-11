# user_input.py
from pydantic import BaseModel
from typing import List

class MaterialProperty(BaseModel):
    prop: str
    upper_bound: float
    lower_bound: float

class Material(BaseModel):
    name: str
    properties: List[MaterialProperty]

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
    properties: List[MixtureProperty]

    def custom_dict(self):
        # Custom method to transform the default Pydantic dict to the desired format
        return {
            self.name: {
                p.prop: {"desired_prop": p.desired_prop} for p in self.properties
            }
        }