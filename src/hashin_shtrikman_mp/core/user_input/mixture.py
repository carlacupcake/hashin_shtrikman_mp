from pydantic import BaseModel

from .mixture_property import MixtureProperty

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