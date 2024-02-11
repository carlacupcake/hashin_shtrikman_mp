import numpy as np
from member_class import Member
from ga_params_class import GAParams
from hs_logger import logger

# DEFAULT_PROPERTY_DOCS = ["carrier-transport", "dielectric", "elastic", "magnetic", "piezoelectric"]
# DEFAULT_DESIRED_PROPS = {"carrier-transport": [],
#                          "dielectric": [],
#                          "elastic": [],
#                          "magnetic": [],
#                          "piezoelectric": []}

class Population:

    def __init__(
            self,
            num_properties: int = 0,
            property_categories:  list = [],
            desired_props:  dict = {},
            values:         np.ndarray = np.empty,
            costs:          np.ndarray = np.empty,
            ga_params:      GAParams = GAParams(),
            property_docs:  dict = {},
            calc_guide:     dict = {}
            ):
        
            self.num_properties = num_properties
            self.property_categories  = property_categories
            self.desired_props  = desired_props
            self.ga_params      = ga_params
            self.property_docs  = property_docs
            self.calc_guide     = calc_guide

            # Update from default based on self.property_docs
            self.values = np.zeros((self.ga_params.get_num_members(), self.num_properties)) if values is np.empty else values
            self.costs  = np.zeros((self.ga_params.get_num_members(), self.num_properties)) if costs  is np.empty else costs

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
    
    