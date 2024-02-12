from pydantic import BaseModel

class GAParams(BaseModel):
    """
    Class to hold the parameters used for the genetic algorithm.
    """

    num_parents: int = 10
    num_kids: int = 10
    num_generations: int = 500
    num_members: int = 200
    tolerance: float = 0.5
    weight_eff_prop: float = 10.0
    weight_conc_factor: float = 0.5

    #------ Getter Methods ------#
    def get_num_parents(self):
        return self.num_parents
    
    def get_num_kids(self):
        return self.num_kids
    
    def get_num_generations(self):
        return self.num_generations
    
    def get_num_members(self):
        return self.num_members
    
    def get_tolerance(self):
        return self.tolerance
    
    def get_weight_eff_prop(self):
        return self.weight_eff_prop
    
    def get_weight_conc_factor(self):
        return self.weight_conc_factor
    
    #------ Setter Methods ------#
    def set_num_parents(self, num_parents):
        self.num_parents = num_parents
        return self
    
    def set_num_kids(self, num_kids):
        self.num_kids = num_kids
        return self
    
    def set_num_generations(self, num_generations):
        self.num_generations = num_generations
        return self
    
    def set_num_members(self, num_members):
        self.num_members = num_members
        return self
    
    def set_tolerance(self, tolerance):
        self.tolerance = tolerance
        return self
    
    def set_weight_eff_prop(self, weight_eff_prop):
        self.weight_eff_prop = weight_eff_prop
        return self
    
    def set_weight_conc_factor(self, weight_conc_factor):
        self.weight_conc_factor = weight_conc_factor
        return self

    

