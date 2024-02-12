import numpy as np
from core.genetic_algo import GAParams
from log.custom_logger import logger
from pydantic import BaseModel, root_validator, Field
from typing import List, Dict, Optional, Any

class Member(BaseModel):
    """
    Class to represent a member of the population in genetic algorithm optimization.
    Stores the properties and configuration for genetic algorithm operations.
    """
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
    property_docs: Dict[str, Dict[str, Any]] = Field(
        default={},
        description="A hard coded yaml file containing property categories "
                    "and their individual properties."
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
    
    #------ Getter Methods ------#
    def get_num_properties(self):
        return self.num_properties
    
    def get_values(self):
        return self.values
    
    def get_property_categories(self):
        return self.property_categories
    
    def get_desired_props(self):
        return self.desired_props
    
    def get_ga_params(self):
        return self.ga_params
            
    def get_cost(self):
        
        # MAIN COST FUNCTION
        # Concentration factors guide (for 2 material composites only): Carrier transport (6), Dielectric (8), Elastic (6), Magnetic (4), Piezoelectic (2)

        # Extract attributes from self
        tolerance = self.ga_params.get_tolerance()
        weight_eff_prop = self.ga_params.get_weight_eff_prop()
        weight_conc_factor = self.ga_params.get_weight_conc_factor()

        # Extract mixing_param and volume fraction from self.values        
        mixing_param = self.values[-2]
        phase1_vol_frac = self.values[-1]
        phase2_vol_frac = 1 - phase1_vol_frac

        # Initialize effective property, concentration factor, and weight arrays
        # Initialize to zero so as not to contribute to cost if unchanged
        effective_properties = []
        concentration_factors = [] 
        cost_func_weights = []   
    
        idx = 0
        for category in self.property_categories:
            props = self.property_docs[category]
            for prop in props:

                if prop == "bulk_modulus":
                    prop1, prop2, prop3, prop4 = self.values[idx:idx+4]
                    idx += 4
                    min_prop_bulk_mod = eval(self.calc_guide['property_comparison']['min'].format(prop1=prop1, prop2=prop2))
                    max_prop_bulk_mod = eval(self.calc_guide['property_comparison']['max'].format(prop1=prop1, prop2=prop2))
                    min_prop_shear_mod = eval(self.calc_guide['property_comparison']['min'].format(prop1=prop3, prop2=prop4))
                    max_prop_shear_mod = eval(self.calc_guide['property_comparison']['max'].format(prop1=prop3, prop2=prop4))
                    # Check if properties are equal
                    if min_prop_bulk_mod == max_prop_bulk_mod:
                        effective_min_bulk_mod = min_prop_bulk_mod  # or prop2, since they are equal
                        effective_max_bulk_mod = max_prop_bulk_mod
                    else:
                        effective_min_bulk_mod  = eval(self.calc_guide['effective_props']['min_bulk_mod'].format(min_prop_bulk_mod=min_prop_bulk_mod, max_prop_bulk_mod=max_prop_bulk_mod, min_prop_shear_mod=min_prop_shear_mod, max_prop_shear_mod=max_prop_shear_mod, phase1_vol_frac=phase1_vol_frac, phase2_vol_frac=phase2_vol_frac,))
                        effective_max_bulk_mod  = eval(self.calc_guide['effective_props']['max_bulk_mod'].format(min_prop_bulk_mod=min_prop_bulk_mod, max_prop_bulk_mod=max_prop_bulk_mod, min_prop_shear_mod=min_prop_shear_mod, max_prop_shear_mod=max_prop_shear_mod, phase1_vol_frac=phase1_vol_frac, phase2_vol_frac=phase2_vol_frac,))
                    
                    if min_prop_shear_mod == max_prop_shear_mod:
                        effective_min_shear_mod = min_prop_shear_mod
                        effective_max_shear_mod = max_prop_shear_mod
                    else:
                        effective_min_shear_mod  = eval(self.calc_guide['effective_props']['min_shear_mod'].format(min_prop_bulk_mod=min_prop_bulk_mod, max_prop_bulk_mod=max_prop_bulk_mod, min_prop_shear_mod=min_prop_shear_mod, max_prop_shear_mod=max_prop_shear_mod, phase1_vol_frac=phase1_vol_frac, phase2_vol_frac=phase2_vol_frac,))
                        effective_max_shear_mod  = eval(self.calc_guide['effective_props']['max_shear_mod'].format(min_prop_bulk_mod=min_prop_bulk_mod, max_prop_bulk_mod=max_prop_bulk_mod, min_prop_shear_mod=min_prop_shear_mod, max_prop_shear_mod=max_prop_shear_mod, phase1_vol_frac=phase1_vol_frac, phase2_vol_frac=phase2_vol_frac,))
                    
                    effective_property_bulk_mod  = mixing_param * effective_max_bulk_mod  + (1 - mixing_param) * effective_min_bulk_mod
                    effective_property_shear_mod = mixing_param * effective_max_shear_mod + (1 - mixing_param) * effective_min_shear_mod

                    effective_properties.append(effective_property_bulk_mod)
                    effective_properties.append(effective_property_shear_mod)

                    if min_prop_bulk_mod == max_prop_bulk_mod:
                        cf_bulk_mod2 = 1/phase2_vol_frac
                        cf_bulk_mod1 = 1/phase1_vol_frac
                    else:
                        cf_bulk_mod2 = eval(self.calc_guide['concentration_factor']['cf_2_other'].format(min_prop=min_prop_bulk_mod, max_prop=max_prop_bulk_mod, effective_property=effective_property_bulk_mod, phase2_vol_frac=phase2_vol_frac,))
                        cf_bulk_mod1 = eval(self.calc_guide['concentration_factor']['cf_1_other'].format(phase1_vol_frac=phase1_vol_frac, phase2_vol_frac=phase2_vol_frac, cf_2_other=cf_bulk_mod2))
                    
                    if min_prop_shear_mod == max_prop_shear_mod:
                        cf_shear_mod2 = 1/phase2_vol_frac
                        cf_shear_mod1 = 1/phase1_vol_frac
                    else:
                        cf_shear_mod2 = eval(self.calc_guide['concentration_factor']['cf_2_other'].format(min_prop=min_prop_shear_mod, max_prop=max_prop_shear_mod, effective_property=effective_property_shear_mod, phase2_vol_frac=phase2_vol_frac,))
                        cf_shear_mod1 = eval(self.calc_guide['concentration_factor']['cf_1_other'].format(phase1_vol_frac=phase1_vol_frac, phase2_vol_frac=phase2_vol_frac, cf_2_other=cf_shear_mod2))
                    
                    concentration_factors.append(cf_bulk_mod1)
                    concentration_factors.append(cf_bulk_mod2)
                    concentration_factors.append(cf_shear_mod1)
                    concentration_factors.append(cf_shear_mod2)

                elif prop == "shear_modulus":
                    pass

                else:
                    prop1, prop2 = self.values[idx:idx+2]
                    idx += 2
                
                    # If properties are not equal, proceed with Hashin-Shtrikman calculation
                    min_prop = eval(self.calc_guide['property_comparison']['min'].format(prop1=prop1, prop2=prop2))
                    max_prop = eval(self.calc_guide['property_comparison']['max'].format(prop1=prop1, prop2=prop2))

                    # Check if properties are equal
                    if min_prop == max_prop:
                        effective_min = min_prop  # or prop2, since they are equal
                        effective_max = max_prop
                    else:     
                        effective_min = eval(self.calc_guide['effective_props']['min'].format(min_prop=min_prop, max_prop=max_prop, phase1_vol_frac=phase1_vol_frac, phase2_vol_frac=phase2_vol_frac))
                        effective_max = eval(self.calc_guide['effective_props']['max'].format(min_prop=min_prop, max_prop=max_prop, phase1_vol_frac=phase1_vol_frac, phase2_vol_frac=phase2_vol_frac))

                    effective_property = mixing_param * effective_max + (1 - mixing_param) * effective_min

                    effective_properties.append(effective_property)

                    if min_prop == max_prop:
                        cf_current1 = (1/phase1_vol_frac)**2
                        cf_current2 = (1/phase2_vol_frac)**2
                    else:
                        cf_current1 = eval(self.calc_guide['concentration_factor']['cf_1'].format(min_prop=min_prop, max_prop=max_prop, phase1_vol_frac=phase1_vol_frac, effective_property=effective_property))
                        cf_current2 = eval(self.calc_guide['concentration_factor']['cf_2'].format(min_prop=min_prop, max_prop=max_prop, phase2_vol_frac=phase2_vol_frac, effective_property=effective_property))
                    
                    if prop == "therm_cond_300k_low_doping":
                        if min_prop == max_prop:
                            cf_current1 = (1/phase1_vol_frac)
                            cf_current2 = (1/phase2_vol_frac)
                        else:
                            cf_current2 = eval(self.calc_guide['concentration_factor']['cf_2_other'].format(min_prop=min_prop, max_prop=max_prop, phase2_vol_frac=phase2_vol_frac, effective_property=effective_property))
                            cf_current1 = eval(self.calc_guide['concentration_factor']['cf_1_other'].format(phase1_vol_frac=phase1_vol_frac, phase2_vol_frac=phase2_vol_frac, cf_2_other=cf_current2))
                        concentration_factors.append(cf_current1)
                        concentration_factors.append(cf_current2)
                    else:
                        cf_current2 = max_prop * cf_current2 * 1/effective_property
                        cf_current1 = 1/phase1_vol_frac * (1 - phase2_vol_frac * cf_current2)                                        
                        concentration_factors.append(cf_current1)
                        concentration_factors.append(cf_current2)

        # Determine weights based on concentration factor magnitudes
        for _, factor in enumerate(concentration_factors):
            if (factor - tolerance) / tolerance > 0:
                cost_func_weights.append(weight_conc_factor)
            else:
                cost_func_weights.append(0)

        # Cast concentration factors, effective properties and weights to numpy arrays
        concentration_factors = np.array(concentration_factors)
        effective_properties = np.array(effective_properties)
        cost_func_weights = np.array(cost_func_weights)

        # Extract desired properties from dictionary
        des_props = []
        for category in self.property_categories:
            des_props.extend(self.desired_props[category])
        
        des_props = np.array(des_props)

        # Assemble the cost function
        domains = len(self.property_categories)
        W = 1/domains
        cost = weight_eff_prop*W * np.sum(abs(np.divide(des_props - effective_properties, effective_properties))) + np.sum(np.multiply(cost_func_weights, abs(np.divide(concentration_factors - tolerance, tolerance))))

        return cost

    #------ Setter Methods ------#

    def set_num_properties(self, num_properties):
        self.num_properties = num_properties
        return self
    
    def set_values(self, values):
        self.values = values
        return self
    
    def set_property_categories(self, property_categories):
        self.property_categories = property_categories
        return self
    
    def set_desired_props(self, des_props):
        self.desired_props = des_props
        return self
    
    def set_ga_params(self, ga_params):
        self.ga_params = ga_params
        return self
    