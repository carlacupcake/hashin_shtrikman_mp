import numpy as np
from genetic_string_class import GeneticString
from ga_params_class import GAParams

class Population:

    def __init__(
            self,
            dv: int = 0,
            material_properties: list = [],
            desired_properties: list=[],
            values: np.ndarray = np.empty,
            costs: np.ndarray = np.empty,
            ga_params: GAParams = None,
            ):
        
            self.dv = dv
            self.material_properties = material_properties
            self.desired_properties = desired_properties
            self.values = values or np.zeros(shape=(self.ga_params.get_S(), self.dv))
            self.costs = costs or np.zeros(shape=(self.ga_params.get_S(), self.dv))
            self.ga_params = ga_params

    #------ Getter Methods ------#
    def get_dv(self):
        return self.dv
    
    def get_material_properties(self):
        return self.material_properties
    
    def get_values(self):
        return self.values
    
    def get_costs(self):
        return self.costs
    
    def get_ga_params(self):
        return self.ga_params
    
    #------ Setter Methods ------#
    def set_dv(self, new_dv):
        self.dv = new_dv
        return self   

    def set_material_properties(self, new_material_properties):
        self.material_properties = new_material_properties
        return self

    def set_values(self, new_values):
        self.values = new_values
        return self

    def set_ga_params(self, new_ga_params):
        self.ga_params = new_ga_params
        return self  
    
    def set_initial_random(self, lower_bounds, upper_bounds):
        Lambda = self.values
        S = self.ga_params.get_S()
        for i in range (S):
            Lambda[i, :] = np.random.uniform(lower_bounds, upper_bounds)
        self.values = Lambda
        return self 
    
    def set_costs(self):
        Lambda = self.values
        costs = np.zeros(shape=(1, self.dv))
        for i in range (self.S):
            this_genetic_string = GeneticString(dv=self.dv, 
                                                values=Lambda[i, :], 
                                                material_properties=self.material_properties, 
                                                desired_properties=self.desired_properties, 
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
    