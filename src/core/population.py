import numpy as np
from core.member import Member
from core.genetic_algo import GAParams
from log.custom_logger import logger
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, root_validator

class Population(BaseModel):
    """
    Class to hold the population of members. The class also implements
    methods to generate the initial population, set the costs of the
    members, and sort the members based on their costs.
    """

    num_materials: int = Field(
        default=0,
        description="Number of materials in the ultimate composite."
    )
    num_properties: int = Field(
        default=0,
        description="Number of properties that each member of the population has."
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
    desired_props: Dict[str, List[float]] = Field(
        default={},
        description="Dictionary mapping individual properties to their desired "
                    "properties."
    )
    values: Optional[np.ndarray] = Field(
        default=None,
        description="Matrix of values representing the population's properties."
    )
    costs: Optional[np.ndarray] = Field(
        default=None,
        description="Array of costs associated with each member of the population."
    )
    ga_params: GAParams = Field(
        default_factory=GAParams,
        description="Parameter initilization class for the genetic algorithm."
    )
    calc_guide: Dict[str, Dict[str, str]] = Field(
        default={},
        description="Calculation guide for property evaluation. This is a "
                    "hard coded yaml file."
    )   

    # To use np.ndarray or other arbitrary types in your Pydantic models
    class Config:
        arbitrary_types_allowed = True

    @root_validator(pre=True)
    def set_default_values_and_costs(cls, values):
        # Extract or initialize ga_params
        ga_params = values.get('ga_params', GAParams())
        num_members = ga_params.num_members  # Assuming GAParams model has a num_members field directly accessible
        num_properties = values.get('num_properties', 0)
        num_materials = values.get('num_materials', 0)

        # Set default for values if not provided or is np.empty
        if values.get('values') is None or (isinstance(values.get('values'), np.ndarray) and values.get('values').size == 0):
            values['values'] = np.zeros((num_members, num_properties * num_materials))

        # Set default for costs in a similar manner
        if values.get('costs') is None or (isinstance(values.get('costs'), np.ndarray) and values.get('costs').size == 0):
            values['costs'] = np.zeros((num_members, num_properties * num_materials))

        return values
   
    #------ Getter Methods ------#
    def get_unique_designs(population, costs):

        # Costs are often equal to >10 decimal points
        # Truncate to obtain a richer set of suggestions
        new_costs = np.round(costs, decimals=3)
        
        # Obtain unique members and costs
        [unique_costs, unique_indices] = np.unique(new_costs, return_index=True)
        unique_members = population[unique_indices]

        return [unique_members, unique_costs] 
    
    #------ Setter Methods ------# 

    def set_random_values(self, lower_bounds = {}, upper_bounds = {}, num_members = 0):

        # Initialize bounds lists
        lower_bounds_list = []
        upper_bounds_list = []

        # Unpack bounds from dictionaries, include bounds for all materials
        for material in lower_bounds.keys():
            if material != "volume-fractions":
                for category, properties in lower_bounds[material].items():
                    if category in self.property_categories:
                        for property in properties:                         
                            lower_bounds_list.append(property)
                            
        for material in upper_bounds.keys():
            if material != "volume-fractions":
                for category, properties in upper_bounds[material].items():
                    if category in self.property_categories:
                        for property in properties:                            
                            upper_bounds_list.append(property)

        # Include volume fractions
        for vf in lower_bounds["volume-fractions"]:                           
            lower_bounds_list.append(vf)

        for vf in upper_bounds["volume-fractions"]:                           
            upper_bounds_list.append(vf)

        # Cast lists to ndarrays
        lower_bounds_array = np.array(lower_bounds_list)
        upper_bounds_array = np.array(upper_bounds_list)
        for i in range(num_members):
            self.values[i, :] = np.random.uniform(lower_bounds_array, upper_bounds_array)

        return self 
    
    def set_costs(self):
        population_values = self.values
        num_members = self.ga_params.num_members
        costs = np.zeros(num_members)
        for i in range(num_members):
            this_member = Member(num_materials=self.num_materials, 
                                 num_properties=self.num_properties,
                                 values=population_values[i, :], 
                                 property_categories=self.property_categories,
                                 property_docs=self.property_docs, 
                                 desired_props=self.desired_props, 
                                 ga_params=self.ga_params,
                                 calc_guide=self.calc_guide)
            costs[i] = this_member.get_cost()

        self.costs = costs
        return self
    
    def set_order_by_costs(self, sorted_indices):
        temporary = np.zeros((self.ga_params.num_members, self.num_properties * self.num_materials))
        for i in range(0, len(sorted_indices)):
            temporary[i,:] = self.values[int(sorted_indices[i]),:]
        self.values = temporary
        return self

    #------ Other Class Methods ------#   

    def sort_costs(self):
        sorted_costs = np.sort(self.costs, axis=0)
        sorted_indices = np.argsort(self.costs, axis=0)
        return [sorted_costs, sorted_indices]                        
    
    