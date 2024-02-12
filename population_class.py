import numpy as np
from member_class import Member
from ga_params_class import GAParams
from hs_logger import logger
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, root_validator

class Population(BaseModel):
    """
    Class to hold the population of members. The class also implements
    methods to generate the initial population, set the costs of the
    members, and sort the members based on their costs.
    """
    
    num_properties: int = Field(
        default=0,
        description="Number of properties that each member of the population has."
    )
    property_categories: List[str] = Field(
        default=[],
        description="List of property categories considered for optimization."
    )
    desired_props: Dict[str, List[float]] = Field(
        default={},
        description="Dictionary mapping individual properties to their desired "
                    "properties."
    )
    ga_params: GAParams = Field(
        default_factory=GAParams,
        description="Parameter initilization class for the genetic algorithm."
    )
    property_docs: Dict[str, Dict[str, Any]] = Field(
        default={},
        description="A hard coded yaml file containing property categories "
                    "and their individual properties."
    )
    calc_guide: Dict[str, Dict[str, str]] = Field(
        default={},
        description="Calculation guide for property evaluation. This is a "
                    "hard coded yaml file."
    )
    values: Optional[np.ndarray] = Field(
        default=None,
        description="Matrix of values representing the population's properties."
    )
    costs: Optional[np.ndarray] = Field(
        default=None,
        description="Array of costs associated with each member of the population."
    )

    # To use np.ndarray or other arbitrary types in your Pydantic models
    class Config:
        arbitrary_types_allowed = True

    @root_validator(pre=True)
    def set_default_values_and_costs(cls, values):
        # Extract or initialize ga_params
        ga_params = values.get('ga_params', GAParams())
        num_members = ga_params.get_num_members()  # Assuming GAParams model has a num_members field directly accessible
        num_properties = values.get('num_properties', 0)

        # Set default for values if not provided or is np.empty
        if values.get('values') is None or (isinstance(values.get('values'), np.ndarray) and values.get('values').size == 0):
            values['values'] = np.zeros((num_members, num_properties))

        # Set default for costs in a similar manner
        if values.get('costs') is None or (isinstance(values.get('costs'), np.ndarray) and values.get('costs').size == 0):
            values['costs'] = np.zeros((num_members, num_properties))

        return values

    #------ Getter Methods ------#
    def get_num_properties(self):
        return self.num_properties
    
    def get_property_docs(self):
        return self.property_docs
    
    def get_values(self):
        return self.values
    
    def get_costs(self):
        return self.costs
    
    def get_ga_params(self):
        return self.ga_params
    
    #------ Setter Methods ------#
    def set_num_properties(self, num_properties):
        self.num_properties = num_properties
        return self 

    def set_property_docs(self, property_categories):
        self.property_categories = property_categories
        return self

    def set_values(self, values):
        self.values = values
        return self

    def set_ga_params(self, ga_params):
        self.ga_params = ga_params
        return self  
    
    def set_initial_random(self, lower_bounds, upper_bounds):

        num_members = self.ga_params.get_num_members()

        lower_bounds_array, upper_bounds_array = self.append_lower_upper_bounds(lower_bounds, upper_bounds)

        for i in range(num_members):
            self.values[i, :] = np.random.uniform(lower_bounds_array, upper_bounds_array)

        return self 
    
    def set_new_random(self, members_minus_parents_minus_kids, lower_bounds, upper_bounds):

        num_members = self.ga_params.get_num_members()
        parents_and_kids = num_members - members_minus_parents_minus_kids # P + K = M - (M - P - K)

        lower_bounds_array, upper_bounds_array = self.append_lower_upper_bounds(lower_bounds, upper_bounds)
        
        for i in range (members_minus_parents_minus_kids):
            self.values[parents_and_kids+i, :] = np.random.uniform(lower_bounds_array, upper_bounds_array)

        return self

    def append_lower_upper_bounds(self, lower_bounds, upper_bounds):
        
        # Initialize bounds lists
        lower_bounds_list = []
        upper_bounds_list = []

        # Unpack bounds from dictionaries, include bounds for all materials
        # Could extend to more materials later
        for category in self.property_categories:
            lower_bounds_list.extend(lower_bounds["mat1"][category]) 
            lower_bounds_list.extend(lower_bounds["mat2"][category]) 
            upper_bounds_list.extend(upper_bounds["mat1"][category])
            upper_bounds_list.extend(upper_bounds["mat2"][category])

        # Include for mixing parameter and volume fraction
        lower_bounds_list.append(0) # mixing parameter gamma in [0,1]
        upper_bounds_list.append(1)
        lower_bounds_list.append(0) # volume fraction in [0,1]
        upper_bounds_list.append(1)

        # Cast lists to ndarrays
        lower_bounds_array = np.array(lower_bounds_list)
        upper_bounds_array = np.array(upper_bounds_list)

        return lower_bounds_array, upper_bounds_array
    
    def set_costs(self):
        population_values = self.values
        num_members = self.ga_params.get_num_members()
        costs = np.zeros(num_members)
        for i in range(num_members):
            this_member = Member(num_properties=self.num_properties, 
                                 values=population_values[i, :], 
                                 property_categories=self.property_categories, 
                                 desired_props=self.desired_props, 
                                 ga_params=self.ga_params,
                                 property_docs=self.property_docs,
                                 calc_guide=self.calc_guide)
            costs[i] = this_member.get_cost()

        self.costs = costs
        return self
    
    def set_order_by_costs(self, sorted_indices):
        num_members = self.ga_params.get_num_members()
        num_properties = self.num_properties
        temporary = np.zeros((num_members, num_properties))
        for i in range(0, len(sorted_indices)):
            temporary[i,:] = self.values[int(sorted_indices[i]),:]
        self.values = temporary
        return self

    #------ Other Class Methods ------#   

    def sort_costs(self):
        sorted_costs = np.sort(self.costs, axis=0)
        sorted_indices = np.argsort(self.costs, axis=0)
        return [sorted_costs, sorted_indices]                         

    def get_unique_designs(population, costs):

        # Costs are often equal to >10 decimal points
        # Truncate to obtain a richer set of suggestions
        new_costs = np.round(costs, decimals=3)
        
        # Obtain unique members and costs
        [unique_costs, unique_indices] = np.unique(new_costs, return_index=True)
        unique_members = population[unique_indices]

        return [unique_members, unique_costs] 
    
    