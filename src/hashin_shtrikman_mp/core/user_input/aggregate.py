"""aggregate.py."""
from pydantic import BaseModel
from .material import Material
from .mixture import Mixture


class Aggregate(BaseModel):
    """
    Represents an aggregate of materials and mixtures.

    Needed in order to construct the dictionary of bounds that encompass
    all the search bounds individually defined for constituent materials.
    """

    name: str
    components: list[Material | Mixture]

    def get_bounds_dict(self) -> dict:
        """Computes the overall upper and lower bounds for each property."""
        bounds_dict = {}
        
        for entity in self.components:
            if isinstance(entity, Material):
                for property in entity.properties:
                    prop_name = property.prop
                    
                    if prop_name not in bounds_dict:
                        bounds_dict[prop_name] = {
                            'upper_bound': property.upper_bound,
                            'lower_bound': property.lower_bound
                        }
                    else:
                        bounds_dict[prop_name]['upper_bound'] = max(
                            bounds_dict[prop_name]['upper_bound'], property.upper_bound
                        )
                        bounds_dict[prop_name]['lower_bound'] = min(
                            bounds_dict[prop_name]['lower_bound'], property.lower_bound
                        )
        
        return bounds_dict
