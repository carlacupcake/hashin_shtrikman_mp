import numpy as np
from genetic_algo import GAParams, CGAParams
from hash_table import CHashEntry, CHashTable
import ctypes
from pydantic import BaseModel, root_validator, Field
from typing import List, Dict, Optional, Any
import warnings

# Load the C code
cmember = ctypes.CDLL('./cbuilds/member.so')

# Define the argument types and return type for the C function
cmember.get_cost.argtypes = [ctypes.c_int, 
                             ctypes.c_int, 
                             ctypes.POINTER(ctypes.c_double), 
                             ctypes.POINTER(ctypes.c_char_p), 
                             ctypes.c_int, 
                             CHashTable,
                             ctypes.POINTER(ctypes.c_double), 
                             CGAParams, 
                             CHashTable]  
cmember.get_cost.restype = ctypes.c_double

#------  Helper Functions ------#
def dict_to_hash(dict):
    # Create a CHashTable object
    size = len(dict)
    hash_table = CHashTable()
    hash_table.size = size
    hash_table.buckets = (ctypes.POINTER(CHashEntry) * size)()

    # Helper function to create a linked list of CHashEntry's for a given Python list of (key, value) tuples
    def create_chain(entries):
        head = None
        for key, value in entries:
            entry = CHashEntry()
            entry.key = ctypes.create_string_buffer(key.encode('utf-8'))
            entry.value = ctypes.create_string_buffer(value.encode('utf-8'))
            entry.next = head
            head = entry
        return ctypes.pointer(head) if head else None

    # Populate the CHashTable buckets with the corresponding CHashEntry linked lists
    for i, (key, value) in enumerate(dict.items()):
        hash_table.buckets[i] = create_chain([(key, value)])

    return hash_table

def numpy_to_double_array(np_array):
    # Ensure the input is a numpy array
    if not isinstance(np_array, np.ndarray):
        raise ValueError("Input must be a numpy array")
    
    # Ensure the numpy array is of type float64 (double in C)
    if np_array.dtype != np.float64:
        np_array = np_array.astype(np.float64)
    
    # Get a pointer to the numpy array's data as a double pointer
    double_array = np_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    
    return double_array

def list_to_char_array(py_list):
    # Ensure the input is a list of strings
    if not all(isinstance(item, str) for item in py_list):
        raise ValueError("Input must be a list of strings")
    
    # Create a ctypes array of c_char_p
    char_array = (ctypes.c_char_p * len(py_list))()
    
    # Populate the ctypes array with the strings converted to bytes
    for i, string in enumerate(py_list):
        char_array[i] = ctypes.create_string_buffer(string.encode('utf-8'))
    
    # Cast to a POINTER(c_char_p), which represents char**
    return ctypes.cast(char_array, ctypes.POINTER(ctypes.c_char_p))

def desired_props_to_double_array(desired_props):
    
    des_props_list = []
    for _, (_, value) in enumerate(desired_props.items()):
        des_props_list.append(value)
    des_props_numpy = np.array(des_props_list)
    des_props_double_array = numpy_to_double_array(des_props_numpy)
    
    return des_props_double_array

def ga_params_to_c(ga_params):

    cga_params = CGAParams()
    cga_params.num_parents = ctypes.cast(ga_params.num_parents, ctypes.c_int)
    cga_params.num_kids = ctypes.cast(ga_params.num_kids, ctypes.c_int)
    cga_params.num_generations = ctypes.cast(ga_params.num_generations, ctypes.c_int)
    cga_params.num_members = ctypes.cast(ga_params.num_members, ctypes.c_int)
    cga_params.mixing_param = ctypes.cast(ga_params.mixing_param, ctypes.c_double)
    cga_params.tolerance = ctypes.cast(ga_params.tolerance, ctypes.c_double)
    cga_params.weight_eff_props = ctypes.cast(ga_params.weight_eff_props, ctypes.c_double)
    cga_params.weight_conc_factor = ctypes.cast(ga_params.weight_conc_factor, ctypes.c_double)

    return cga_params
        
#------  Member Class ------#
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
    
    #------ Getter Methods ------#
    def cget_cost(self):

        num_materials = self.num_materials
        num_properties = self.num_properties
        values = numpy_to_double_array(self.values)
        property_categories = list_to_char_array(self.property_categories)
        num_property_categories = len(self.property_categories)
        property_docs = dict_to_hash(self.property_docs)
        desired_props = desired_props_to_double_array(self.desired_props)
        ga_params = ga_params_to_c(self.ga_params)
        calc_guide = dict_to_hash(self.calc_guide)

        return cmember.get_cost(num_materials,
                                num_properties,
                                values,
                                property_categories,
                                num_property_categories,
                                property_docs,
                                desired_props,
                                ga_params,
                                calc_guide)
   
    def get_cost(self):

        """ MAIN COST FUNCTION """
        
        # Extract attributes from self
        tolerance          = self.ga_params.tolerance
        weight_eff_prop    = self.ga_params.weight_eff_prop
        weight_conc_factor = self.ga_params.weight_conc_factor

        # Initialize effective property, concentration factor, and weight arrays
        # Initialize to zero so as not to contribute to cost if unchanged
        effective_properties  = []
        concentration_factors = [] 
        cost_func_weights     = []  

        # Get Hashin-Shtrikman effective properties for all properties           
        idx = 0

        for category in self.property_categories:
                
            if category == "elastic":
                moduli_eff_props, moduli_cfs = self.get_elastic_eff_props_and_cfs(idx=idx)
                effective_properties.extend(moduli_eff_props)
                concentration_factors.extend(moduli_cfs)

                eff_univ_aniso, cfs_univ_aniso = self.get_general_eff_prop_and_cfs(idx=idx+2)
                effective_properties.extend(eff_univ_aniso)
                concentration_factors.extend(cfs_univ_aniso)

            else:
                for p in range(idx, idx + len(self.property_docs[category])): # loop through all properties in the category
                    new_eff_props, new_cfs = self.get_general_eff_prop_and_cfs(idx=p)
                    effective_properties.extend(new_eff_props)
                    concentration_factors.extend(new_cfs)
            
            idx += len(self.property_docs[category])
                    
        # Determine weights based on concentration factor magnitudes
        for factor in concentration_factors:
            if (factor - tolerance) / tolerance > 0:
                cost_func_weights.append(weight_conc_factor)
            else:
                cost_func_weights.append(0)

        # Cast concentration factors, effective properties and weights to numpy arrays
        concentration_factors = np.array(concentration_factors)
        effective_properties  = np.array(effective_properties)
        cost_func_weights     = np.array(cost_func_weights)

        # Extract desired properties from dictionary
        des_props = []
        for category, properties in self.desired_props.items():
            des_props.extend(properties)
        des_props = np.array(des_props)

        # Assemble the cost function
        domains = len(self.property_categories)
        W = 1/domains
        cost = weight_eff_prop*W * np.sum(abs(np.divide(des_props - effective_properties, effective_properties))) + np.sum(np.multiply(cost_func_weights, abs(np.divide(concentration_factors - tolerance, tolerance))))

        return cost
    
    def get_general_eff_prop_and_cfs(self, idx = 0): # idx is the index in self.values where category properties begin

        # Initialize effective property, concentration factor, and weight arrays
        # Initialize to zero so as not to contribute to cost if unchanged
        effective_properties  = []
        concentration_factors = [] 

        # Prepare indices for looping over properties
        stop = -self.num_materials     # the last num_materials entries are volume fractions, not material properties
        step = self.num_properties - 1 # subtract 1 so as not to include volume fraction

        # Get Hashin-Shtrikman effective properties for all properties
        properties = self.values[idx:stop:step]
        phase1 = np.min(properties)
        phase2 = np.max(properties)
        phase1_idx = np.argmin(properties) 
        phase2_idx = np.argmax(properties)
        phase1_vol_frac = self.values[-self.num_materials + phase1_idx]
        phase2_vol_frac = self.values[-self.num_materials + phase2_idx]

        # Compute effective property bounds with Hashin-Shtrikman
        if phase1 == phase2:
            effective_prop_min = phase1
            effective_prop_max = phase2
        else:
            effective_prop_min = eval(self.calc_guide['effective_props']['eff_min'].format(phase1=phase1, phase2=phase2, phase1_vol_frac=phase1_vol_frac, phase2_vol_frac=phase2_vol_frac))
            effective_prop_max = eval(self.calc_guide['effective_props']['eff_max'].format(phase1=phase1, phase2=phase2, phase1_vol_frac=phase1_vol_frac, phase2_vol_frac=phase2_vol_frac))

        # Compute concentration factors for electrical load sharing
        mixing_param = self.ga_params.mixing_param
        effective_prop = mixing_param * effective_prop_max + (1 - mixing_param) * effective_prop_min
        effective_properties.append(effective_prop)

        if phase1_vol_frac == 0:
            cf_response1_cf_load1 = (1/phase2_vol_frac)**2
            cf_response2_cf_load2 = (1/phase2_vol_frac)**2
        elif phase2_vol_frac == 0:
            cf_response1_cf_load1 = (1/phase1_vol_frac)**2
            cf_response2_cf_load2 = (1/phase1_vol_frac)**2
        elif phase1 == phase2:
            cf_response1_cf_load1 = (1/phase1_vol_frac)**2 
            cf_response2_cf_load2 = (1/phase2_vol_frac)**2 
        else:
            cf_response1_cf_load1 = eval(self.calc_guide['concentration_factors']['cf_1'].format(phase1=phase1, phase2=phase2, phase1_vol_frac=phase1_vol_frac, effective_property=effective_prop))
            cf_response2_cf_load2 = eval(self.calc_guide['concentration_factors']['cf_2'].format(phase1=phase1, phase2=phase2, phase2_vol_frac=phase2_vol_frac, effective_property=effective_prop))

        concentration_factors.append(cf_response1_cf_load1)
        concentration_factors.append(cf_response2_cf_load2)

        return effective_properties, concentration_factors
    
    def get_elastic_eff_props_and_cfs(self, idx = 0): # idx is the index in self.values where elastic properties begin

        # Initialize effective property and concentration factor arrays
        # Initialize to zero so as not to contribute to cost if unchanged
        effective_properties  = []
        concentration_factors = []  

        # Prepare indices for looping over properties
        stop = -self.num_materials     # the last num_materials entries are volume fractions, not material properties
        step = self.num_properties - 1 # subtract 1 so as not to include volume fraction

        # Extract bulk moduli and shear moduli from member 
        bulk_mods = self.values[idx:stop:step]         
        phase1_bulk = np.min(bulk_mods)
        phase2_bulk = np.max(bulk_mods)
        phase1_bulk_idx = np.argmin(bulk_mods)
        phase2_bulk_idx = np.argmax(bulk_mods)

        shear_mods = self.values[idx+1:stop:step]
        phase1_shear = np.min(shear_mods)
        phase2_shear = np.max(shear_mods)
        phase1_shear_idx = np.argmax(shear_mods)
        phase2_shear_idx = np.argmax(shear_mods)

        if (phase1_bulk_idx != phase1_shear_idx) or (phase2_bulk_idx != phase2_shear_idx):
            warnings.warn("Cannot perform optimization when for bulk modulus phase 1 > phase 2 and for shear modulus phase 2 > phase 1 or vice versa.")

        phase1_vol_frac = self.values[-self.num_materials + phase1_bulk_idx] # shear should have the same index
        phase2_vol_frac = self.values[-self.num_materials + phase2_bulk_idx]          

        # Get Hashin-Shtrikman effective properties for bulk and shear moduli
        if phase1_bulk == phase2_bulk:
            effective_bulk_mod_min  = phase1_bulk
            effective_bulk_mod_max  = phase2_bulk
        else:
            effective_bulk_mod_min = eval(self.calc_guide['effective_props']['bulk_mod_min'].format(phase1_bulk=phase1_bulk, phase2_bulk=phase2_bulk, phase1_shear=phase1_shear, phase2_shear=phase2_shear, phase1_vol_frac=phase1_vol_frac, phase2_vol_frac=phase2_vol_frac))
            effective_bulk_mod_max = eval(self.calc_guide['effective_props']['bulk_mod_max'].format(phase1_bulk=phase1_bulk, phase2_bulk=phase2_bulk, phase1_shear=phase1_shear, phase2_shear=phase2_shear, phase1_vol_frac=phase1_vol_frac, phase2_vol_frac=phase2_vol_frac))

        if phase1_shear == phase2_shear:
            effective_shear_mod_min = phase1_shear
            effective_shear_mod_max = phase2_shear
        else:
            effective_shear_mod_min = eval(self.calc_guide['effective_props']['shear_mod_min'].format(phase1_bulk=phase1_bulk, phase2_bulk=phase2_bulk, phase1_shear=phase1_shear, phase2_shear=phase2_shear, phase1_vol_frac=phase1_vol_frac, phase2_vol_frac=phase2_vol_frac))
            effective_shear_mod_max = eval(self.calc_guide['effective_props']['shear_mod_max'].format(phase1_bulk=phase1_bulk, phase2_bulk=phase2_bulk, phase1_shear=phase1_shear, phase2_shear=phase2_shear, phase1_vol_frac=phase1_vol_frac, phase2_vol_frac=phase2_vol_frac))

        # Compute concentration factors for mechanical load sharing
        mixing_param = self.ga_params.mixing_param
        bulk_mod_eff  = eval(self.calc_guide['effective_props']['eff_prop'].format(mixing_param=mixing_param, eff_min=effective_bulk_mod_min, eff_max=effective_bulk_mod_max))         
        shear_mod_eff = eval(self.calc_guide['effective_props']['eff_prop'].format(mixing_param=mixing_param, eff_min=effective_shear_mod_min, eff_max=effective_shear_mod_max))

        effective_properties.append(bulk_mod_eff)
        effective_properties.append(shear_mod_eff)

        if phase1_vol_frac == 0:
            cf_phase2_bulk  = 1/phase2_vol_frac
            cf_phase1_bulk  = 1/phase2_vol_frac
            cf_phase2_shear = 1/phase2_vol_frac
            cf_phase1_shear = 1/phase2_vol_frac
        elif phase2_vol_frac == 0:
            cf_phase2_bulk  = 1/phase1_vol_frac
            cf_phase1_bulk  = 1/phase1_vol_frac
            cf_phase2_shear = 1/phase1_vol_frac
            cf_phase1_shear = 1/phase1_vol_frac
        elif phase1_bulk == phase2_bulk:
            cf_phase2_bulk = 1/phase2_vol_frac
            cf_phase1_bulk = 1/phase1_vol_frac
        else:
            cf_phase2_bulk = eval(self.calc_guide['concentration_factors']['cf_2_elastic'].format(phase2_vol_frac=phase2_vol_frac, phase1=phase1_bulk, phase2=phase2_bulk, effective_property=bulk_mod_eff))
            cf_phase1_bulk = eval(self.calc_guide['concentration_factors']['cf_1_elastic'].format(phase1_vol_frac=phase1_vol_frac, phase2_vol_frac=phase2_vol_frac, cf_2_elastic=cf_phase2_bulk))
  
        if phase1_vol_frac == 0:
            cf_phase2_shear = 1/phase2_vol_frac
            cf_phase1_shear = 1/phase2_vol_frac
        elif phase2_vol_frac == 0:
            cf_phase2_shear = 1/phase1_vol_frac
            cf_phase1_shear = 1/phase1_vol_frac
        elif phase1_shear == phase2_shear:
            cf_phase2_shear = 1/phase2_vol_frac
            cf_phase1_shear = 1/phase1_vol_frac
        else:
            cf_phase2_shear = eval(self.calc_guide['concentration_factors']['cf_2_elastic'].format(phase2_vol_frac=phase2_vol_frac, phase1=phase1_shear, phase2=phase2_shear, effective_property=shear_mod_eff))
            cf_phase1_shear = eval(self.calc_guide['concentration_factors']['cf_1_elastic'].format(phase1_vol_frac=phase1_vol_frac, phase2_vol_frac=phase2_vol_frac, cf_2_elastic=cf_phase2_shear))

        # Write over default calculation for concentration factor
        concentration_factors.append(cf_phase1_bulk)
        concentration_factors.append(cf_phase1_shear)
        concentration_factors.append(cf_phase2_bulk)
        concentration_factors.append(cf_phase2_shear)      

        return effective_properties, concentration_factors
            
    

        

    
