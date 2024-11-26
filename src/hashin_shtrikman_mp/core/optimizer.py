#import copy
#import itertools
import json
#import matplotlib.pyplot as plt
import numpy as np
#import plotly.graph_objects as go
import re
import sys
#import warnings
import yaml

from datetime import datetime
#from matplotlib import cm
#import matplotlib.gridspec as gridspec
from monty.serialization import loadfn
from mp_api.client import MPRester
#from mpcontribs.client import Client
#from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from pydantic import BaseModel, Field, model_validator
#from tabulate import tabulate
from typing import Any, Dict, List, Union, Optional

# Custom imports
from hashin_shtrikman_mp.core.genetic_algo import GAParams
from hashin_shtrikman_mp.core.member import Member
from hashin_shtrikman_mp.core.population import Population
from hashin_shtrikman_mp.core.user_input import UserInput
from ..log.custom_logger import logger

# YAML files
sys.path.insert(1, '../io/inputs')
CALC_GUIDE = "cost_calculation_formulas.yaml"
HS_HEADERS_YAML = "display_table_headers.yaml"
MP_PROPERTY_DOCS_YAML = "mp_property_docs.yaml"

# HashinShtrikman class defaults
DEFAULT_FIELDS: dict = {"material_id": [], 
                        "is_stable": [], 
                        "band_gap": [], 
                        "is_metal": [],
                        "formula_pretty": [],}
MODULE_DIR = Path(__file__).resolve().parent

# Load and compile cost calculation formulas
from ..io.inputs.compile_cost_calculation_formulas import compile_formulas
COMPILED_CALC_GUIDE = compile_formulas(loadfn(f"{MODULE_DIR}/../io/inputs/{CALC_GUIDE}"))

np.seterr(divide='raise')

class Optimizer(BaseModel):
    """
    Hashin-Shtrikman optimization class. 

    Class to integrate Hashin-Shtrikman (HS) bounds with a genetic algorithm
    and find optimal material properties for each composite constituent to achieve 
    desired properties.
    """

    api_key: Optional[str] = Field(
        default=None, 
        description="API key for accessing Materials Project database."
        )
    mp_contribs_project: Optional[str] = Field(
        default=None, 
        description="MPContribs project name for querying project-specific data."
        )
    user_input: UserInput = Field(
        default_factory=UserInput, 
        description="User input specifications for the optimization process."
        )
    fields: Dict[str, List[Any]] = Field(
        default_factory=lambda: DEFAULT_FIELDS.copy(), 
        description="Fields to query from the Materials Project database."
        )
    property_categories: List[str] = Field(
        default_factory=list, 
        description="List of property categories considered for optimization."
        )
    property_docs: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, 
        description="A hard coded yaml file containing property categories and their individual properties."
        )
    lower_bounds: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Lower bounds for properties of materials considered in the optimization."
        )
    upper_bounds: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Upper bounds for properties of materials considered in the optimization."
        )
    desired_props: Dict[str, List[float]] = Field(
        default_factory=dict, 
        description="Dictionary mapping individual properties to their desired properties."
        )
    num_materials: int = Field(
        default=0, 
        description="Number of materials to comprise the composite."
        )
    num_properties: int = Field(
        default=0, 
        description="Number of properties being optimized."
        )
    indices_elastic_moduli: List[Any] = Field(
        default=[None, None],
        description="For handling coupling between bulk & shear moduli"
                    "List of length 2, first element is index of bulk modulus"
                    "in list version of the bounds, second element is the"
                    "shear modulus"
        )
    ga_params: GAParams = Field(
        default_factory=GAParams, 
        description="Parameter initilization class for the genetic algorithm."
        )
    final_population: Population = Field(
        default_factory=Population, 
        description="Final population object after optimization."
        )
    lowest_costs: np.ndarray = Field(
        default_factory=lambda: np.empty(0), 
        description="Lowest cost values across generations."
        )
    avg_parent_costs: np.ndarray = Field(
        default_factory=lambda: np.empty(0), 
        description="Average cost of the top-performing parents across generations."
        )   
    calc_guide: Union[Dict[str, Any], Any] = Field(
        default_factory=lambda: COMPILED_CALC_GUIDE,
        description="Calculation guide for property evaluation with compiled expressions."
    )
    
    # To use np.ndarray or other arbitrary types in your Pydantic models
    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode='before')
    def load_and_set_properties(cls, values):
        # Load property categories and docs
        property_categories, property_docs = cls.load_property_categories(f"{MODULE_DIR}/../io/inputs/{MP_PROPERTY_DOCS_YAML}", user_input=values.get("user_input", {}))
        values["property_categories"] = property_categories
        values["property_docs"] = property_docs
        
        # Since user_input is required to set desired props and bounds, ensure it's processed last
        user_input = values.get("user_input", {})
        if user_input:
            num_materials  = cls.set_num_materials_from_user_input(user_input)
            desired_props  = cls.set_desired_props_from_user_input(user_input, property_categories=property_categories, property_docs=property_docs)
            lower_bounds   = cls.set_bounds_from_user_input(user_input, 'lower_bound', property_docs=property_docs, num_materials=num_materials)
            upper_bounds   = cls.set_bounds_from_user_input(user_input, 'upper_bound', property_docs=property_docs, num_materials=num_materials)
            num_properties = cls.set_num_properties_from_desired_props(desired_props=desired_props)
            indices_elastic_moduli = cls.set_elastic_idx_from_user_input(upper_bounds=upper_bounds, property_categories=property_categories)
            
            # Update values accordingly
            values.update({
                "desired_props":       desired_props,
                "lower_bounds":        lower_bounds,
                "upper_bounds":        upper_bounds,
                "num_properties":      num_properties,
                "num_materials":       num_materials,
                "indices_elastic_moduli": indices_elastic_moduli
            })
        
        return values

    @staticmethod
    def load_property_categories(filename=f"{MODULE_DIR}/../io/inputs/mp_property_docs.yaml", user_input: Dict = {}):
            logger.info(f"Loading property categories from {filename}.")

            """Load property categories from a JSON file."""
            property_categories = []
            try:
                property_docs = loadfn(filename)

                # Flatten the user input to get a list of all properties defined by the user
                user_defined_properties = []

                for entity in user_input.values():
                    for property in entity:
                        user_defined_properties.append(property)

                # Only keep the unique entries of the list
                user_defined_properties = list(set(user_defined_properties))

                # Iterate through property categories to check which are present in the user input
                for category, properties in property_docs.items():
                    if any(prop in user_defined_properties for prop in properties):
                        property_categories.append(category)

            except FileNotFoundError:
                logger.error(f"File {filename} not found.")
            except json.JSONDecodeError:
                logger.error(f"Error decoding JSON from file {filename}.")
            
            logger.info(f"property_categories = {property_categories}")
            return property_categories, property_docs

    @staticmethod
    def set_num_materials_from_user_input(user_input):
        num_materials = user_input.len() - 1
        return num_materials
    
    @staticmethod
    def set_num_properties_from_desired_props(desired_props):
        num_properties = 0

        # Iterate through property categories to count the total number of properties
        for _, properties in desired_props.items():
            num_properties += len(properties)  # Add the number of properties in each category

        # Account for volume fractions
        num_properties += 1

        return num_properties
    
    @staticmethod
    def set_bounds_from_user_input(user_input: Dict, bound_key: str, property_docs: Dict[str, List[str]], num_materials: int):
        if bound_key not in ['upper_bound', 'lower_bound']:
            raise ValueError("bound_key must be either 'upper_bound' or 'lower_bound'.")
        
        # Get bounds for material properties from user_input
        bounds: Dict[str, Dict[str, List[float]]] = {}
        for material, properties in user_input.items():
            if material == 'mixture':  # Skip 'mixture' as it's not a material
                continue

            bounds[material] = {}
            for category, prop_list in property_docs.items():
                category_bounds = []

                for prop in prop_list:
                    if prop in properties and bound_key in properties[prop]:
                        # Append the specified bound if the property is found
                        category_bounds.append(properties[prop][bound_key])

                if category_bounds:
                    bounds[material][category] = category_bounds

        # Add bounds for volume fractions, then set self
        if bound_key == 'upper_bound':
            bounds["volume-fractions"] = [0.99] * num_materials
        else:
            bounds["volume-fractions"] = [0.01] * num_materials

        return bounds

    @staticmethod
    def set_desired_props_from_user_input(user_input: Dict, property_categories: List[str], property_docs: Dict):

        # Initialize the dictionary to hold the desired properties
        desired_props: Dict[str, List[float]] = {category: [] for category in property_categories}

        # Extracting the desired properties from the 'mixture' part of final_dict
        mixture_props = user_input.get('mixture', {})
        logger.info(f"mixture_props = {mixture_props}")

        # Iterate through each property category and its associated properties
        for category, properties in property_docs.items():
            for prop in properties:
                # Check if the property is in the mixture; if so, append its desired value
                if prop in mixture_props:
                    desired_props[category].append(mixture_props[prop]['desired_prop'])

        return desired_props  
    
    @staticmethod
    def set_elastic_idx_from_user_input(upper_bounds: Dict, property_categories: List[str]):
        idx = 0
        indices = [None, None]
        for material in upper_bounds.keys():
            if material != "volume-fractions":
                for category, properties in upper_bounds[material].items():
                    if category in property_categories: 
                        for property in properties:      
                            print(f'material: {material}, category: {category}, property: {property}')               
                            if category == "elastic":  
                                indices = [idx, idx+1]                           
                                return indices 
                            idx += 1
                                
        return indices
 
    def set_HS_optim_params(self, gen_counter: bool = False):
        """ 
        MAIN OPTIMIZATION FUNCTION 
        """

        # Unpack necessary attributes from self
        num_parents     = self.ga_params.num_parents
        num_kids        = self.ga_params.num_kids
        num_generations = self.ga_params.num_generations
        num_members     = self.ga_params.num_members
        
        # Initialize arrays to store the cost and original indices of each generation
        all_costs = np.ones((num_generations, num_members))
        
        # Initialize arrays to store best performer and parent avg 
        lowest_costs = np.zeros(num_generations)     # best cost
        avg_parent_costs = np.zeros(num_generations) # avg cost of parents
        
        # Generation counter
        g = 0

        # Initialize array to store costs for current generation
        costs = np.zeros(num_members)

        # Randomly populate first generation  
        population = Population(num_materials=self.num_materials, 
                                num_properties=self.num_properties, 
                                property_categories=self.property_categories, 
                                property_docs=self.property_docs,
                                desired_props=self.desired_props, 
                                ga_params=self.ga_params,
                                calc_guide=self.calc_guide)
        population.set_random_values(lower_bounds=self.lower_bounds, 
                                     upper_bounds=self.upper_bounds, 
                                     start_member=0,
                                     indices_elastic_moduli=self.indices_elastic_moduli)

        # Calculate the costs of the first generation
        population.set_costs()   

        # Sort the costs of the first generation
        [sorted_costs, sorted_indices] = population.sort_costs()  
        all_costs[g, :] = sorted_costs.reshape(1, num_members) 

        # Store the cost of the best performer and average cost of the parents 
        lowest_costs[g] = np.min(sorted_costs)
        avg_parent_costs[g] = np.mean(sorted_costs[0:num_parents])
        
        # Update population based on sorted indices
        population.set_order_by_costs(sorted_indices)
        
        # Perform all later generations    
        while g < num_generations:

            if gen_counter:
                print(f"Generation {g} of {num_generations}")

            costs[0:num_parents] = sorted_costs[0:num_parents] # retain the parents from the previous generation
            
            # Select top parents from population to be breeders
            for p in range(0, num_parents, 2):
                phi1, phi2 = np.random.rand(2)
                kid1 = phi1 * population.values[p, :] + (1-phi1) * population.values[p+1, :]
                kid2 = phi2 * population.values[p, :] + (1-phi2) * population.values[p+1, :]
                
                # Append offspring to population, overwriting old population members 
                population.values[num_parents+p,   :] = kid1
                population.values[num_parents+p+1, :] = kid2
            
                # Cast offspring to members and evaluate costs
                kid1 = Member(num_materials=self.num_materials,
                              num_properties=self.num_properties, 
                              values=kid1, 
                              property_categories=self.property_categories,
                              property_docs=self.property_docs, 
                              desired_props=self.desired_props, 
                              ga_params=self.ga_params,
                              calc_guide=self.calc_guide)
                kid2 = Member(num_materials=self.num_materials,
                              num_properties=self.num_properties, 
                              values=kid2, 
                              property_categories=self.property_categories, 
                              property_docs=self.property_docs, 
                              desired_props=self.desired_props, 
                              ga_params=self.ga_params,
                              calc_guide=self.calc_guide)
                costs[num_parents+p]   = kid1.get_cost()
                costs[num_parents+p+1] = kid2.get_cost()
                        
            # Randomly generate new members to fill the rest of the population
            parents_plus_kids = num_parents + num_kids
            population.set_random_values(lower_bounds=self.lower_bounds, 
                                         upper_bounds=self.upper_bounds, 
                                         start_member=parents_plus_kids,
                                         indices_elastic_moduli=self.indices_elastic_moduli)

            # Calculate the costs of the gth generation
            population.set_costs()

            # Sort the costs for the gth generation
            [sorted_costs, sorted_indices] = population.sort_costs()  
            all_costs[g, :] = sorted_costs.reshape(1, num_members) 
        
            # Store the cost of the best performer and average cost of the parents 
            lowest_costs[g] = np.min(sorted_costs)
            avg_parent_costs[g] = np.mean(sorted_costs[0:num_parents])
        
            # Update population based on sorted indices
            population.set_order_by_costs(sorted_indices)

            # Update the generation counter
            g = g + 1 

        # Update self attributes following optimization
        self.final_population = population            
        self.lowest_costs = lowest_costs
        self.avg_parent_costs = avg_parent_costs     
        
        return                
    
    def generate_consolidated_dict(self, overall_bounds_dict: dict = {}):

        """
        MAIN FUNCTION USED TO GENERATE MATERIAL PROPERTY DICTIONARY DEPENDING ON USER REQUEST
        """
        
        # Base query initialization
        query = {}

        # Iterate over the properties in the overall_bounds_dict and dynamically build the query
        for prop, bounds in overall_bounds_dict.items():
            # Skip 'elec_cond_300k_low_doping' and 'therm_cond_300k_low_doping'
            if prop in ['elec_cond_300k_low_doping', 'therm_cond_300k_low_doping']:
                continue  # Skip the current iteration

            # Proceed if 'upper_bound' and 'lower_bound' exist for the property
            if 'upper_bound' in bounds and 'lower_bound' in bounds:
                query[prop] = (bounds['lower_bound'], bounds['upper_bound'])
        
        # Print the query before changes
        #print(f"Initial query: {query}")

        # Add additional fields you want to query, like 'material_id', 'formula_pretty', and all the properties in the initial query
        fields = ['material_id', 'formula_pretty']  # Fixed fields
        fields.extend(query.keys())  # Adding all the keys from the query to the fields list
        #print(f"Fields: {fields}")

        # Change 'bulk_modulus' to 'k_voigt'
        if 'bulk_modulus' in query:
            query['k_vrh'] = query.pop('bulk_modulus')

        # Change 'shear_modulus' to 'g_voigt'
        if 'shear_modulus' in query:
            query['g_vrh'] = query.pop('shear_modulus')

        # Change 'universal_anisotropy' to 'elastic_anisotropy'
        if 'universal_anisotropy' in query:
            query['elastic_anisotropy'] = query.pop('universal_anisotropy')
        
        # change 'e_ij_max' to 'piezoelectric_modulus'
        if 'e_ij_max' in query:
            query['piezoelectric_modulus'] = query.pop('e_ij_max')
        
        mpr = MPRester("QePM93qZsMKNPkI4fEYaJfB7dONoQjaM")

        # Perform the query on the Materials Project database using the built query
        materials = mpr.materials.summary.search(
            **query,  # Dynamically passing the property bounds as query filters
            fields=fields,
            # num_chunks=100
        )

        # Load the mp_property_docs.yaml file
        with open(f'{MODULE_DIR}/../io/inputs/mp_property_docs.yaml', 'r') as file:
            mp_property_docs = yaml.safe_load(file)
            
        # Initialize dictionary to hold the desired data format
        result_dict: Dict[str, List[Any]] = {
            "material_id": [],
            "formula_pretty": []
        }


        # Traverse the YAML structure to get all the keys
        for category, properties in mp_property_docs.items():
            # Check if the category is present in self.property_categories
            if category in self.property_categories:
                for prop, subprop in properties.items():
                    # If there's a subproperty (e.g., voigt for bulk_modulus), append the subproperty
                    if isinstance(subprop, str):
                        result_dict[f"{prop}_{subprop}"] = []
                    else:
                        # Otherwise, append the main property
                        result_dict[prop] = []

        # Print the initialized result_dict
        print(f"Initialized result_dict: {result_dict}")

        # remove all the rows that have None values
        materials = [material for material in materials if all([getattr(material, field, None) is not None for field in fields])]

        # Extract data and organize it into the result_dict
        for material in materials:
            result_dict["material_id"].append(material.material_id)
            result_dict["formula_pretty"].append(material.formula_pretty)

            # Define a mapping between query keys and result_dict keys and their corresponding material attributes
            property_map = {
                "k_voigt": ("bulk_modulus_voigt", "bulk_modulus", "voigt"),
                "g_voigt": ("shear_modulus_voigt", "shear_modulus", "voigt"),
                "elastic_anisotropy": ("universal_anisotropy", "universal_anisotropy"),
                "elec_cond_300k_low_doping": ("elec_cond_300k_low_doping", "elec_cond_300k_low_doping"),
                "therm_cond_300k_low_doping": ("therm_cond_300k_low_doping", "therm_cond_300k_low_doping"),
                "e_electronic": ("e_electronic", "e_electronic"),
                "e_ionic": ("e_ionic", "e_ionic"),
                "e_total": ("e_total", "e_total"),
                "n": ("n", "n"),
                "total_magnetization": ("total_magnetization", "total_magnetization"),
                "total_magnetization_normalized_vol": ("total_magnetization_normalized_vol", "total_magnetization_normalized_vol"),
                "e_ij_max": ("e_ij_max", "e_ij_max")
            }

            # Iterate over the properties in the query and append values to result_dict dynamically
            for prop, (result_key, material_attr, *sub_attr) in property_map.items():
                if prop in query:
                    # Check if there's a sub-attribute (e.g., "voigt" in "bulk_modulus")
                    if sub_attr:
                        value = getattr(material, material_attr, {})
                        result_dict[result_key].append(value.get(sub_attr[0], None))  # Access sub-attribute if it exists
                    else:
                        result_dict[result_key].append(getattr(material, material_attr, None))  # Direct access to attribute

        
        # Initialize variables
        formula_pretty_length = len(result_dict['formula_pretty'])

        # Print lengths of all properties
        #for key in result_dict:
        #    print(f"Length of {key}: {len(result_dict[key])}")

        # Filter out incomplete or empty lists that don't need sorting
        non_empty_keys = [key for key in result_dict if len(result_dict[key]) == formula_pretty_length]

        # Sort the result_dict by ascending order of material_id for non-empty lists
        sorted_indices = sorted(range(formula_pretty_length), key=lambda i: result_dict['formula_pretty'][i])

        # Re-arrange all the properties in result_dict based on the sorted indices, but only for non-empty lists
        for key in non_empty_keys:
            result_dict[key] = [result_dict[key][i] for i in sorted_indices]

        # for all the empty lists, append None to the corresponding material_id
        for key in result_dict:
            if key not in non_empty_keys:
                result_dict[key] = [None] * formula_pretty_length

        # Print the length of material_id after sorting
        #print(f"Length of material_id after sorting: {len(result_dict['formula_pretty'])}")

        # Print the sorted result_dict keys and lengths of the lists
        #for key in result_dict:
        #    print(f"Final Length of {key}: {len(result_dict[key])}")

        # Print keys of the result_dict
        #print(f"Keys of result_dict: {result_dict.keys()}")
        #print(f"result_dict['material_id'] = {result_dict['material_id']}")


        if "carrier-transport" in self.property_categories:
            print("Carrier transport is in the property categories")
            from mpcontribs.client import Client
            client = Client(apikey="QePM93qZsMKNPkI4fEYaJfB7dONoQjaM")
            client.get_project("carrier_transport")

            # Iterate over the properties in the overall_bounds_dict and dynamically build the query
            query_carrier_transport = {}
            for prop, bounds in overall_bounds_dict.items():
                # Skip 'elec_cond_300k_low_doping' and 'therm_cond_300k_low_doping'
                if prop in ['elec_cond_300k_low_doping', 'therm_cond_300k_low_doping']:
                    # Proceed if 'upper_bound' and 'lower_bound' exist for the property
                    if 'upper_bound' in bounds and 'lower_bound' in bounds:
                        query_carrier_transport[prop] = (bounds['lower_bound'], bounds['upper_bound'])

            #print(f"Query for carrier transport: {query_carrier_transport}")

            tables = client.query_contributions({"project":"carrier_transport",
                                        "data__sigma__p__value__gt": query_carrier_transport['elec_cond_300k_low_doping'][0] / 1e15 / 1e-14, # the 1003100.0,
                                        "data__sigma__p__value__lt": query_carrier_transport['elec_cond_300k_low_doping'][1] / 1e15 / 1e-14, #2093100.0,
                                        "data__kappa__p__value__gt": query_carrier_transport['therm_cond_300k_low_doping'][0] / 1e9 / 1e-14, #7091050.0,
                                        "data__kappa__p__value__lt": query_carrier_transport['therm_cond_300k_low_doping'][1] / 1e9 / 1e-14, #8591050.0,
                                        "identifier__in": result_dict['material_id'],
                                    },
                                    fields=['identifier', 'formula', 'data.sigma.p', 'data.kappa.p'],
                                    sort='+formula') #  'identifier','data.V', 'tables', 'kappa' , 'kappa.p.value', 'sigma.p.value', '_all' (2769600.0, 1093100.0)
            
            #print(f"Tables: {tables}")

            # Only append the values to the corresponding material_id from the result_dict. At the end, make all the remaning values 
            # corresponding to the material_id as None
            # Iterate over the tables returned and map the data to the result_dict
            for table in tables['data']:
                #print(f"Table: {table}")
                material_id = table['identifier']  # Material ID from the table
                if material_id in result_dict['material_id']:  # Only map for materials already in result_dict
                    index = result_dict['material_id'].index(material_id)
                    #print(f"Index: {index}")
                    
                    # Access the electrical conductivity and thermal conductivity values
                    sigma_value = table['data']['sigma']['p']['value']  # Electrical conductivity
                    kappa_value = table['data']['kappa']['p']['value']  # Thermal conductivity
                    
                    # Convert and append the values to the correct positions in the result_dict
                    result_dict['elec_cond_300k_low_doping'][index] = sigma_value * 1e15 * 1e-14
                    result_dict['therm_cond_300k_low_doping'][index] = kappa_value * 1e9 * 1e-14
            
            # Drop rows with None values
            keys_to_check = result_dict.keys()
            indices_to_drop = [i for i in range(formula_pretty_length) if any(result_dict[key][i] is None for key in keys_to_check)]

            for i in sorted(indices_to_drop, reverse=True):
                for key in result_dict:
                    result_dict[key].pop(i)
            
            # # change the key name of bulk_modugus_vrh to bulk_modulus & shear_modulus_vrh to shear_modulus
            # if 'bulk_modulus_vrh' in result_dict:
            #     result_dict['bulk_modulus'] = result_dict.pop('bulk_modulus_vrh')
            # if 'shear_modulus_vrh' in result_dict:
            #     result_dict['shear_modulus'] = result_dict.pop('shear_modulus_vrh')

            # change the key name of bulk_modugus_voigt to bulk_modulus & shear_modulus_voigt to shear_modulus
            if 'bulk_modulus_voigt' in result_dict:
                result_dict['bulk_modulus'] = result_dict.pop('bulk_modulus_voigt')
            if 'shear_modulus_voigt' in result_dict:
                result_dict['shear_modulus'] = result_dict.pop('shear_modulus_voigt')


        # Save the consolidated results to a JSON file
        now = datetime.now()
        my_file_name = f"{MODULE_DIR}/../io/outputs/consolidated_dict_" + now.strftime("%m_%d_%Y_%H_%M_%S")
        with open(my_file_name, "w") as my_file:
            json.dump(result_dict, my_file)
        
        return result_dict
    
    def superscript_to_int(self, superscript_str):
        superscript_to_normal = {
            "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
            "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9"
        }
        normal_str = "".join(superscript_to_normal.get(char, char) for char in superscript_str)
        return int(normal_str)
    
    def mp_contribs_prop(self, prop, my_dict):
        if prop == "therm_cond_300k_low_doping":
            table_column = 7
        elif prop == "elec_cond_300k_low_doping":
            table_column = 5
        #print(f"my_dict[tables] = {my_dict['tables']}")
        # prop_str = my_dict["tables"][table_column].iloc[2, 0]
        if table_column < len(my_dict["tables"]):
            prop_str = my_dict["tables"][table_column].iloc[2, 0]
        else:
            print(f"No table available at index {table_column}.")
            prop_str = 0

        if not isinstance(prop_str, str):
            prop_str = str(prop_str)
        prop_str = prop_str.replace(",", "")

        if "×10" in prop_str:
            # Extract the numeric part before the "±" symbol and the exponent
            prop_str, prop_exponent_str = re.search(r"\((.*?) ±.*?\)×10(.*)", prop_str).groups()
            # Convert the exponent part to a format that Python can understand
            prop_exponent = self.superscript_to_int(prop_exponent_str.strip())
            # Combine the numeric part and the exponent part, and convert the result to a float
            prop_value = float(f"{prop_str}e{prop_exponent}") * 1e-14  # multply by relaxation time, 10 fs
            logger.info(f"{prop}_if_statement = {prop_value}")
        else:
            prop_value = float(prop_str) * 1e-14  # multply by relaxation time, 10 fs
            logger.info(f"{prop}_else_statement = {prop_value}")
        
        self.fields[prop].append(prop_value)