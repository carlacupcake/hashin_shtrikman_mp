# member.py
import numpy as np
from genetic_algo import GAParams
import os
from pydantic import BaseModel, root_validator, Field, ConfigDict
import sys
from typing import List, Dict, Optional, Any

sys.path.append(os.path.join(os.path.dirname(__file__), 'cbuilds'))
from cmember import CMember

class Member(BaseModel):
    """
    Class to represent a member of the population in genetic algorithm optimization.
    Stores the properties and configuration for genetic algorithm operations.
    """

    num_materials: int = Field(
        default=0,
        description="Number of materials in the ultimate composite."
    )
    num_properties: int = Field(
        default=0,
        description="Number of properties that each member of the population has."
    )
    values: Optional[np.ndarray] = Field(
        default=None,
        description="Values array representing the member's properties."
    )
    property_categories: List[str] = Field(
        default=[],
        description="List of property categories considered for optimization."
    )
    property_docs: Dict[str, Dict[str, Any]] = Field(
        default={},
        description="A hard coded yaml file containing property categories "
                    "and their individual properties."
    )
    desired_props: Dict[str, Any] = Field(
        default={},
        description="Dictionary mapping individual properties to their desired "
                    "properties."
    )
    ga_params: Optional[GAParams] = Field(
        default=None,
        description="Parameter initilization class for the genetic algorithm."
    )
    calc_guide: Dict[str, Any] = Field(
        default={},
        description="Calculation guide for property evaluation. This is a "
                    "hard coded yaml file."
    )    

    # To use np.ndarray or other arbitrary types in your Pydantic models
    class Config:
        arbitrary_types_allowed = True

    @root_validator(pre=True)
    def check_and_initialize_arrays(cls, values):
        # Initialize 'values' with zeros if not provided or if it is np.empty
        if values.get('values') is None or (isinstance(values.get('values'), np.ndarray) and values.get('values').size == 0):
            num_properties = values.get('num_properties', 0)
            # Assuming you want a 2D array shape based on your original code
            values['values'] = np.zeros(shape=(num_properties, 1))  
        return values

    def get_cost(self):
        cmember = CMember(
            num_materials=self.num_materials,
            num_properties=self.num_properties,
            values=self.values,
            property_categories=self.property_categories,
            property_docs=self.property_docs,
            desired_props=self.desired_props,
            ga_params=self.ga_params,
            calc_guide=self.calc_guide
        )
        return cmember.get_cost()
            
    

        

    
