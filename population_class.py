import numpy as np
from genetic_string_class import GeneticString
from ga_params_class import GAParams
from hs_logger import logger

DEFAULT_PROPERTY_DOCS = ["carrier-transport", "dielectric", "elastic", "magnetic", "piezoelectric"]
DEFAULT_DESIRED_PROPS = {"carrier-transport": [],
                         "dielectric": [],
                         "elastic": [],
                         "magnetic": [],
                         "piezoelectric": []}

class Population:

    def __init__(
            self,
            dv: int = 0,
            property_docs: list = DEFAULT_PROPERTY_DOCS,
            desired_props: dict = DEFAULT_DESIRED_PROPS,
            values: np.ndarray = np.empty,
            costs: np.ndarray = np.empty,
            ga_params: GAParams = GAParams(),
            ):
        
            self.dv = dv
            self.property_docs = property_docs
            self.desired_props = desired_props
            self.ga_params = ga_params

            # Update from default based on self.property_docs
            self.values = np.zeros((self.ga_params.get_S(), self.dv)) if values == np.empty else values
            self.costs  = np.zeros((self.ga_params.get_S(), self.dv)) if costs  == np.empty else costs

    #------ Getter Methods ------#
    def get_dv(self):
        return self.dv
    
    def get_property_docs(self):
        return self.property_docs
    
    def get_values(self):
        return self.values
    
    def get_costs(self):
        return self.costs
    
    def get_ga_params(self):
        return self.ga_params
    
    #------ Setter Methods ------#
    def set_dv(self, dv):
        self.dv = dv
        return self 

    def set_property_docs(self, property_docs):
        self.property_docs = property_docs
        return self

    def set_values(self, values):
        self.values = values
        return self

    def set_ga_params(self, ga_params):
        self.ga_params = ga_params
        return self  
    
    def set_initial_random(self, lower_bounds, upper_bounds):

        S = self.ga_params.get_S()

        # Initialize bounds lists
        lower_bounds_list = []
        upper_bounds_list = []

        # Unpack bounds from dictionaries, include bounds for all materials
        # Could extend to more materials later
        if "carrier-transport" in self.property_docs:
            lower_bounds_list.extend(lower_bounds["mat1"]["carrier-transport"]) 
            lower_bounds_list.extend(lower_bounds["mat2"]["carrier-transport"]) 
            upper_bounds_list.extend(upper_bounds["mat1"]["carrier-transport"])
            upper_bounds_list.extend(upper_bounds["mat2"]["carrier-transport"]) 
        if "dielectric" in self.property_docs:
            lower_bounds_list.extend(lower_bounds["mat1"]["dielectric"])
            lower_bounds_list.extend(lower_bounds["mat2"]["dielectric"]) 
            upper_bounds_list.extend(upper_bounds["mat1"]["dielectric"]) 
            upper_bounds_list.extend(upper_bounds["mat2"]["dielectric"]) 
        if "elastic" in self.property_docs:
            lower_bounds_list.extend(lower_bounds["mat1"]["elastic"]) 
            lower_bounds_list.extend(lower_bounds["mat2"]["elastic"]) 
            upper_bounds_list.extend(upper_bounds["mat1"]["elastic"]) 
            upper_bounds_list.extend(upper_bounds["mat2"]["elastic"]) 
        if "magnetic" in self.property_docs:
            lower_bounds_list.extend(lower_bounds["mat1"]["magnetic"])
            lower_bounds_list.extend(lower_bounds["mat2"]["magnetic"]) 
            upper_bounds_list.extend(upper_bounds["mat1"]["magnetic"]) 
            upper_bounds_list.extend(upper_bounds["mat2"]["magnetic"]) 
        if "piezoelectric" in self.property_docs:
            lower_bounds_list.extend(lower_bounds["mat1"]["piezoelectric"]) 
            lower_bounds_list.extend(lower_bounds["mat2"]["piezoelectric"]) 
            upper_bounds_list.extend(upper_bounds["mat1"]["piezoelectric"]) 
            upper_bounds_list.extend(upper_bounds["mat2"]["piezoelectric"]) 

        # Include for mixing parameter and volume fraction
        lower_bounds_list.append(0) # mixing parameter gamma in [0,1]
        upper_bounds_list.append(1)
        lower_bounds_list.append(0) # volume fraction in [0,1]
        upper_bounds_list.append(1)

        # Cast lists to ndarrays
        lower_bounds_array = np.array(lower_bounds_list)
        upper_bounds_array = np.array(upper_bounds_list)
        for i in range (S):
            self.values[i, :] = np.random.uniform(lower_bounds_array, upper_bounds_array)

        return self 
    
    def set_new_random(self, SPK, lower_bounds, upper_bounds):

        S = self.ga_params.get_S()
        PK = S - SPK # the number of parents plus the number of kids

        # Initialize bounds lists
        lower_bounds_list = []
        upper_bounds_list = []

        # Unpack bounds from dictionaries, include bounds for all materials
        # Could extend to more materials later
        if "carrier-transport" in self.property_docs:
            lower_bounds_list.extend(lower_bounds["mat1"]["carrier-transport"]) 
            lower_bounds_list.extend(lower_bounds["mat2"]["carrier-transport"]) 
            upper_bounds_list.extend(upper_bounds["mat1"]["carrier-transport"])
            upper_bounds_list.extend(upper_bounds["mat2"]["carrier-transport"]) 
        if "dielectric" in self.property_docs:
            lower_bounds_list.extend(lower_bounds["mat1"]["dielectric"])
            lower_bounds_list.extend(lower_bounds["mat2"]["dielectric"]) 
            upper_bounds_list.extend(upper_bounds["mat1"]["dielectric"]) 
            upper_bounds_list.extend(upper_bounds["mat2"]["dielectric"]) 
        if "elastic" in self.property_docs:
            lower_bounds_list.extend(lower_bounds["mat1"]["elastic"]) 
            lower_bounds_list.extend(lower_bounds["mat2"]["elastic"]) 
            upper_bounds_list.extend(upper_bounds["mat1"]["elastic"]) 
            upper_bounds_list.extend(upper_bounds["mat2"]["elastic"]) 
        if "magnetic" in self.property_docs:
            lower_bounds_list.extend(lower_bounds["mat1"]["magnetic"])
            lower_bounds_list.extend(lower_bounds["mat2"]["magnetic"]) 
            upper_bounds_list.extend(upper_bounds["mat1"]["magnetic"]) 
            upper_bounds_list.extend(upper_bounds["mat2"]["magnetic"]) 
        if "piezoelectric" in self.property_docs:
            lower_bounds_list.extend(lower_bounds["mat1"]["piezoelectric"]) 
            lower_bounds_list.extend(lower_bounds["mat2"]["piezoelectric"]) 
            upper_bounds_list.extend(upper_bounds["mat1"]["piezoelectric"]) 
            upper_bounds_list.extend(upper_bounds["mat2"]["piezoelectric"]) 

        # Include for mixing parameter and volume fraction
        lower_bounds_list.append(0) # mixing parameter gamma in [0,1]
        upper_bounds_list.append(1)
        lower_bounds_list.append(0) # volume fraction in [0,1]
        upper_bounds_list.append(1)

        # Cast lists to ndarrays
        lower_bounds_array = np.array(lower_bounds_list)
        upper_bounds_array = np.array(upper_bounds_list)
        for i in range (SPK):
            self.values[PK+i, :] = np.random.uniform(lower_bounds_array, upper_bounds_array)

        return self
    
    def set_costs(self):
        Lambda = self.values
        S = self.ga_params.get_S()
        costs = np.zeros(S)
        for i in range (S):
            this_genetic_string = GeneticString(dv=self.dv, 
                                                values=Lambda[i, :], 
                                                property_docs=self.property_docs, 
                                                desired_props=self.desired_props, 
                                                ga_params=self.ga_params)
            costs[i] = this_genetic_string.get_cost()

        self.costs = costs
        return self
    
    def set_order_by_costs(self, ind):
        S = self.ga_params.get_S()
        dv = self.dv
        temporary = np.zeros((S,dv))
        for i in range(0, len(ind)):
            temporary[i,:] = self.values[int(ind[i]),:]
        self.values = temporary
        return self

    #------ Other Class Methods ------#   

    def sort_costs(self):
        sorted_costs = np.sort(self.costs, axis=0)
        ind = np.argsort(self.costs, axis=0)
        return [sorted_costs, ind]                         
    
    def reorder(self):
        return
    
    def get_unique_designs(self):
        return

    def get_unique_designs(Lambda, costs):

        # Costs are often equal to >10 decimal points
        # Truncate to obtain a richer set of suggestions
        new_costs = np.round(costs, decimals=3)
        
        # Obtain Unique Strings and Costs
        [unique_costs, iuniq] = np.unique(new_costs, return_index=True)
        unique_strings = Lambda[iuniq]

        return [unique_strings, unique_costs] 
    