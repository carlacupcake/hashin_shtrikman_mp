from pydantic import BaseModel

from .material_property import MaterialProperty

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