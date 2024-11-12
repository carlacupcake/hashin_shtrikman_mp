import copy
import itertools
import json
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import re
import sys
import warnings
import yaml

from datetime import datetime
from matplotlib import cm
import matplotlib.gridspec as gridspec
from monty.serialization import loadfn
from mp_api.client import MPRester
from mpcontribs.client import Client
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from pydantic import BaseModel, Field, model_validator
from tabulate import tabulate
from typing import Any, Dict, List, Union, Optional


from ..log.custom_logger import logger

# Custom imports
from .genetic_algo import GAParams
from .member import Member
from .population import Population
from .user_input import UserInput

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

class HashinShtrikman(BaseModel):
    """
    Hashin-Shtrikman optimization class. 

    Class to integrate Hashin-Shtrikman (HS) bounds with a genetic algorithm, 
    leveraging the Materials Project (MP) database.
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

    #------ Load property docs from MP ------# 
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
        
    #------ Getter Methods ------#
    def get_headers(self, include_mpids=False, filename = f"{MODULE_DIR}/../io/inputs/{HS_HEADERS_YAML}"):

        with open(filename, 'r') as stream:
            try:
                data = yaml.safe_load(stream)
                headers = []

                # Add headers for mp-ids
                if include_mpids:
                    for m in range(1, self.num_materials + 1):
                        headers.append(f"Material {m} MP-ID")

                # Add headers for material properties
                for category, properties in data["Per Material"].items():
                    if category in self.property_categories:
                        for property in properties.values():                            
                            for m in range(1, self.num_materials + 1):
                                headers.append(f"Phase {m} " + property)

                # Add headers for volume fractions
                for m in range(1, self.num_materials + 1):
                    headers.append(f"Phase {m} Volume Fraction")

                # Add headers for "Common" properties if present
                if "Common" in data:
                    for common_key in data["Common"].keys():
                        headers.append(common_key)

            except yaml.YAMLError as exc:
                print(exc)
        
        return headers
        
    def get_unique_designs(self):

        # Costs are often equal to >10 decimal points, truncate to obtain a richer set of suggestions
        self.final_population.set_costs()
        final_costs = self.final_population.costs
        rounded_costs = np.round(final_costs, decimals=3)
    
        # Obtain unique members and costs
        [unique_costs, unique_indices] = np.unique(rounded_costs, return_index=True)
        unique_members = self.final_population.values[unique_indices]

        return [unique_members, unique_costs] 

    def get_table_of_best_designs(self, rows: int = 10):

        [unique_members, unique_costs] = self.get_unique_designs()
        table_data = np.hstack((unique_members[0:rows, :], unique_costs[0:rows].reshape(-1, 1))) 

        return table_data

    def get_dict_of_best_designs(self):

        # Initialize dictionaries for each material based on selected property categories
        best_designs_dict = {}
        for m in range(1, self.num_materials + 1):
            best_designs_dict[f"mat{m}"] = {}
        
        # Initialize the structure for each category
        for category in self.property_categories:
            for mat in best_designs_dict.keys():
                best_designs_dict[mat][category] = {}
                for property in self.property_docs[category]:
                    best_designs_dict[mat][category][property] = []

        [unique_members, unique_costs] = self.get_unique_designs()

        # Populate the dictionary with unique design values
        stop = -self.num_materials     # the last num_materials entries are volume fractions, not material properties
        step = self.num_properties - 1 # subtract 1 so as not to include volume fraction

        for i, _ in enumerate(unique_costs):
            idx = 0
            for category in self.property_categories:                
                for property in self.property_docs[category]:
                    all_phase_props = unique_members[i][idx:stop:step] 
                    for m, mat in enumerate(best_designs_dict.keys()):
                        best_designs_dict[mat][category][property].append(all_phase_props[m])
                    idx += 1

        return best_designs_dict
    
    def get_material_matches(self, overall_bounds_dict: dict = {}, user_input: Dict = {}):

        best_designs_dict = self.get_dict_of_best_designs()  
        print(f"best_designs_dict: {best_designs_dict}")
        print(f"keys of best_designs_dict: {best_designs_dict.keys()}")     
        print(f"keys of best_designs_dict['mat1']: {best_designs_dict['mat1'].keys()}")
        print(f"overall_bounds_dict: {overall_bounds_dict}")
        
        # Generate the consolidated dict based on overall bounds
        consolidated_dict = self.generate_consolidated_dict(overall_bounds_dict) # TODO uncomment this line and remove the above lines
        
        # save consolidated_dict to file
        with open(f"consolidated_dict_{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}", 'w') as f:
            json.dump(consolidated_dict, f)


        # Initialize an empty dictionary to store the output in the required format
        final_matching_materials = {}

        # Iterate through each material in best_designs_dict
        for mat_key, mat_data in best_designs_dict.items():

            # Extract the property data from best_designs_dict
            best_design_props = {}
            if "carrier-transport" in mat_data:
                best_design_props.update({
                    'elec_cond': mat_data['carrier-transport'].get('elec_cond_300k_low_doping'),
                    'therm_cond': mat_data['carrier-transport'].get('therm_cond_300k_low_doping')
                })

            if "elastic" in mat_data:
                best_design_props.update({
                    'bulk_modulus': mat_data['elastic'].get('bulk_modulus'),
                    'shear_modulus': mat_data['elastic'].get('shear_modulus'),
                    'universal_anisotropy': mat_data['elastic'].get('universal_anisotropy')
                })

            if "dielectric" in mat_data:
                best_design_props.update({
                    'e_electronic': mat_data['dielectric'].get('e_electronic'),
                    'e_ionic': mat_data['dielectric'].get('e_ionic'),
                    'e_total': mat_data['dielectric'].get('e_total'),
                    'n': mat_data['dielectric'].get('n')
                })
            
            if "magnetic" in mat_data:
                best_design_props.update({
                    'total_magnetization': mat_data['magnetic'].get('total_magnetization'),
                    'total_magnetization_normalized_vol': mat_data['magnetic'].get('total_magnetization_normalized_vol')
                })
            
            if "piezoelectric" in mat_data:
                best_design_props.update({
                    'e_ij_max': mat_data['piezoelectric'].get('e_ij_max')
                })

            
            # Initialize an empty list to store matching materials for the current mat_key
            matching_materials_for_current_mat = []

            # print len of material_id, elec_cond, therm_cond, bulk_modulus, shear_modulus, universal_anisotropy
            if "carrier-transport" in mat_data:
                print(f"len(elec_cond): {len(consolidated_dict['elec_cond_300k_low_doping'])}")
                print(f"len(therm_cond): {len(consolidated_dict['therm_cond_300k_low_doping'])}")
            
            if "elastic" in mat_data:
                print(f"len(bulk_modulus): {len(consolidated_dict['bulk_modulus_vrh'])}")
                print(f"len(shear_modulus): {len(consolidated_dict['shear_modulus_vrh'])}")
                print(f"len(universal_anisotropy): {len(consolidated_dict['universal_anisotropy'])}")
            
            print(f"len(material_id): {len(consolidated_dict['material_id'])}")
            

            # Iterate through each material in consolidated_dict
            for i, material_id in enumerate(consolidated_dict["material_id"]):

                # Convert material_id to a string before storing it
                material_id_str = str(material_id)
                
                # Retrieve the properties for this material from consolidated_dict
                material_props = {}
                if "carrier-transport" in mat_data:
                    material_props.update({
                        'elec_cond': consolidated_dict['elec_cond_300k_low_doping'][i],
                        'therm_cond': consolidated_dict['therm_cond_300k_low_doping'][i]
                    })
                
                if "elastic" in mat_data:
                    material_props.update({
                        'bulk_modulus': consolidated_dict['bulk_modulus_vrh'][i],
                        'shear_modulus': consolidated_dict['shear_modulus_vrh'][i],
                        'universal_anisotropy': consolidated_dict['universal_anisotropy'][i]
                    })

                if "dielectric" in mat_data:
                    material_props.update({
                        'e_electronic': consolidated_dict['e_electronic'][i],
                        'e_ionic': consolidated_dict['e_ionic'][i],
                        'e_total': consolidated_dict['e_total'][i],
                        'n': consolidated_dict['n'][i]
                    })
                
                if "magnetic" in mat_data:
                    material_props.update({
                        'total_magnetization': consolidated_dict['total_magnetization'][i],
                        'total_magnetization_normalized_vol': consolidated_dict['total_magnetization_normalized_vol'][i]
                    })
                
                if "piezoelectric" in mat_data:
                    material_props.update({
                        'e_ij_max': consolidated_dict['e_ij_max'][i]
                    })

                # Compare properties with best_designs_dict (within 1% threshold)
                matching = {}
                for prop_key, values in best_design_props.items():
                    if prop_key in material_props:
                        # Iterate through each value in best_design_props for comparison
                        for value in values:
                            if abs(value - material_props[prop_key]) / value < 1:
                                matching[prop_key] = material_props[prop_key]

                # If all the props are within the threshold, add the material_id to the matching_materials_for_current_mat
                if len(matching) == len(best_design_props):
                    matching_materials_for_current_mat.append({material_id_str: matching})

            # If any matches were found for this mat_key, add them to the final output
            if matching_materials_for_current_mat:
                final_matching_materials[mat_key] = matching_materials_for_current_mat

        return final_matching_materials     
    
    def get_all_possible_vol_frac_combos(self, num_fractions: int = 30):
        all_vol_frac_ranges = []
        for _ in range(self.num_materials - 1):
            all_vol_frac_ranges.append(list(np.linspace(0.01, 0.99, num_fractions))) 

        all_vol_frac_combos = []
        all_vol_frac_combo_tups = list(itertools.product(*all_vol_frac_ranges))
        for vol_frac_tup in all_vol_frac_combo_tups:
            new_combo = []
            new_element = 1.0
            for element in vol_frac_tup:
                new_combo.append(element)
                new_element = new_element - element
            new_combo.append(new_element)
            all_vol_frac_combos.append(new_combo)

        return all_vol_frac_combos
    
    def get_material_match_costs(self, matches_dict, consolidated_dict: dict = {}):
    
        if consolidated_dict == {}:
            with open("consolidated_dict_02_11_2024_23_45_58") as f: # TODO change to get most recent consolidated dict
                consolidated_dict = json.load(f)

        all_vol_frac_combos = self.get_all_possible_vol_frac_combos()
        materials = list(matches_dict.values())
        material_combinations = list(itertools.product(*materials))
        
        # List to keep track of the lowest 5 costs and their corresponding data
        top_rows = []

        for combo in material_combinations:

            material_values = []
            mat_ids = np.zeros((len(material_combinations), self.num_materials))   

            for category in self.property_categories:
                for property in self.property_docs[category]:
                    for material_dict in combo:

                        # Extract the material ID string from the dictionary
                        material_id = list(material_dict.keys())[0]  # Extract the material ID from the dictionary key

                        # Ensure material_id is a string (it should already be, but this is to be safe)
                        material_str = str(material_id)

                        if property in consolidated_dict.keys(): # TODO carrier-transport not registering ??
                            m = consolidated_dict["material_id"].index(material_str)               
                            material_values.append(consolidated_dict[property][m])
                        else:
                            material_values.append(1.0) # TODO remove later, this is for debugging

            # Create population of same properties for all members based on material match combination
            population_values = np.tile(material_values, (len(all_vol_frac_combos),1))

            # Vary the volume fractions across the population
            volume_fractions = np.array(all_vol_frac_combos).reshape(len(all_vol_frac_combos), self.num_materials)

            # Include the random mixing parameters and volume fractions in the population
            values = np.c_[population_values, volume_fractions] 

            # Instantiate the population and find the best performers
            population = Population(num_materials=self.num_materials,
                                    num_properties=self.num_properties, 
                                    values=values, 
                                    property_categories=self.property_categories,
                                    property_docs=self.property_docs, 
                                    desired_props=self.desired_props, 
                                    ga_params=self.ga_params,
                                    calc_guide=self.calc_guide)
            population.set_costs()
            [sorted_costs, sorted_indices] = population.sort_costs()
            population.set_order_by_costs(sorted_indices)
            sorted_costs = np.reshape(sorted_costs, (len(sorted_costs), 1))

            # Assemble a table for printing
            mat_ids = []
            for material_dict in combo:                    
                material_id = list(material_dict.keys())[0] 
                material_str = str(material_id)
                mat_ids.append(np.reshape([material_id]*self.ga_params.num_members, (self.ga_params.num_members,1)))
            mat_ids = np.column_stack(mat_ids)

            # Combine material IDs, values, and costs into a single table for this combination
            table_data = np.c_[mat_ids, population.values, sorted_costs] 

            # Only keep top 5 rows of this combo
            table_data = table_data[0:5, :]  # hardcoded to be 5 rows, could change
            
            # Add these rows to the global top_rows list and sort by cost
            top_rows.extend(table_data.tolist())  # Convert to list to extend
            top_rows.sort(key=lambda x: x[-1])  # Sort by the last column (costs)

            # Keep only the lowest 5 rows across all combinations
            top_rows = top_rows[:5]

        # At this point, top_rows contains the lowest 5 costs across all combinations
        # Now prepare the table for final output

        headers = self.get_headers(include_mpids=True)

        header_color = 'lavender'
        odd_row_color = 'white'
        even_row_color = 'lightgrey'
        cells_color = [[odd_row_color, even_row_color, odd_row_color, even_row_color, odd_row_color]]  # Hardcoded to 5 rows

        # Create the final table figure
        fig = go.Figure(data=[go.Table(
            columnwidth=1000,
            header=dict(
                values=headers,
                fill_color=header_color,
                align='left',
                font=dict(size=12),
                height=30
            ),
            cells=dict(
                values=[list(col) for col in zip(*top_rows)],  # Transpose top_rows to get columns
                fill_color=cells_color,
                align='left',
                font=dict(size=12),
                height=30,
            )
        )])

        # Update layout for horizontal scrolling
        fig.update_layout(
            title="Optimal Material Combinations to Comprise Desired Composite",
            title_font_size=20,
            title_x=0.15,
            margin=dict(l=0, r=0, t=40, b=0),
            height=400,
            autosize=True
        )
        fig.show()

        return

    #------ Setter Methods ------#
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
        
        """ MAIN OPTIMIZATION FUNCTION """

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

    #------ Other Methods ------#
    def print_table_of_best_designs(self, rows: int = 10):

        table_data = self.get_table_of_best_designs(rows)
        headers = self.get_headers()

        header_color = 'lavender'
        odd_row_color = 'white'
        even_row_color = 'lightgrey'
        if rows % 2 == 0:
            multiplier = int(rows/2)
            cells_color = [[odd_row_color, even_row_color]*multiplier]
        else:
            multiplier = int(np.floor(rows/2))
            cells_color = [[odd_row_color, even_row_color]*multiplier]
            cells_color.append(odd_row_color)

        fig = go.Figure(data=[go.Table(
            columnwidth = 1000,
            header = dict(
                values=headers,
                fill_color=header_color,
                align='left',
                font=dict(size=12),
                height=30
            ),
            cells = dict(
                values=[table_data[:, i] for i in range(table_data.shape[1])],
                fill_color=cells_color,
                align='left',
                font=dict(size=12),
                height=30,
            )
        )])

        # Update layout for horizontal scrolling
        fig.update_layout(
            title="Optimal Properties Recommended by Genetic Algorithm",
            title_font_size=20,
            title_x=0.2,
            margin=dict(l=0, r=0, t=40, b=0),
            height=400,
            autosize=True
        )
        fig.show()

        return
    
    def plot_optimization_results(self):
 
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=list(range(self.ga_params.num_generations)),
            y=self.avg_parent_costs,
            mode='lines',
            name='Avg. of top 10 performers'
        ))

        fig.add_trace(go.Scatter(
            x=list(range(self.ga_params.num_generations)),
            y=self.lowest_costs,
            mode='lines',
            name='Best costs'
        ))

        fig.update_layout(
            title='Convergence of Genetic Algorithm',
            title_x=0.25,
            xaxis_title='Generation',
            yaxis_title='Cost',
            legend=dict(
                font=dict(size=14),
                x=1,
                y=1,
                xanchor='right',
                yanchor='top',
                bgcolor='rgba(255, 255, 255, 0.5)'
            ),
            title_font_size=24,
            xaxis_title_font_size=20,
            yaxis_title_font_size=20,
            width=600,  
            height=400, 
            margin=dict(l=50, r=50, t=50, b=50) 
        )
        fig.show()

        return

    def visualize_composite_eff_props(self, match, consolidated_dict: dict = {}, num_fractions: int = 99):

        if consolidated_dict == {}:
            with open("consolidated_dict_02_11_2024_23_45_58") as f: # TODO get from latest final_dict file: change this to a method that reads from the latest MP database
                consolidated_dict = json.load(f)

        # Too much computation to use the default for 4 phase, so reduce num_fractions
        if len(match) == 4:
            num_fractions = 20
        if len(match) == 1 or len(match) > 4:
            warnings.warn("No visualizations available for composites with 5 or more phases.")
            return
        
        all_vol_frac_combos = self.get_all_possible_vol_frac_combos(num_fractions=num_fractions)
        
        material_values = []
        for category in self.property_categories:
            for property in self.property_docs[category]:
                for material in match:
                    if property in consolidated_dict.keys(): 
                        m = consolidated_dict["material_id"].index(material)               
                        material_values.append(consolidated_dict[property][m])

        # Create population of same properties for all members based on material match combination
        population_values = np.tile(material_values, (len(all_vol_frac_combos),1))

        # Only the vary the volume fractions across the population
        # Create uniform volume fractions from 0 to 1 with a spacing of 0.02 but with a shape of self.ga_params.get_num_members() & 1
        volume_fractions = np.array(all_vol_frac_combos).reshape(len(all_vol_frac_combos), self.num_materials)

        # Include the random mixing parameters and volume fractions in the population
        values = np.c_[population_values, volume_fractions] 

        # Instantiate the population and find the best performers
        # For 2 phases and x volume fractions, there are x   possible volume fraction combinations
        # For 3 phases and x volume fractions, there are x^2 possible volume fraction combinations
        # for 4 phases and x volume fractions, there are x^3 possible volume fraction combinations
        this_pop_ga_params = self.ga_params
        this_pop_ga_params.num_members = num_fractions**(len(match) - 1)
        population = Population(num_materials=self.num_materials,
                                num_properties=self.num_properties, 
                                values=values, 
                                property_categories=self.property_categories,
                                property_docs=self.property_docs, 
                                desired_props=self.desired_props, 
                                ga_params=this_pop_ga_params,
                                calc_guide=self.calc_guide)
        all_effective_properties = population.get_effective_properties()
        print(f'all_effective_properties.shape: {all_effective_properties.shape}')
        
        # Get property strings for labeling the plot(s)
        file_name = f"{MODULE_DIR}/../io/inputs/{HS_HEADERS_YAML}"
        property_strings = []
        with open(file_name, 'r') as stream:
            data = yaml.safe_load(stream)
            for category, properties in data["Per Material"].items():
                if category in self.property_categories:
                    for property in properties.values():                            
                        property_strings.append(property)

        def extract_property(text):
            match = re.match(r'([^,]+), \[.*\]', text)
            if match:
                return match.group(1).strip()
            return None
        
        def extract_units(text):
            match = re.search(r'\[.*?\]', text)
            if match:
                return match.group(0)
            return None
        
        for i, property_string in enumerate(property_strings):

            property = extract_property(property_string)
            units = extract_units(property_string)
            effective_properties = all_effective_properties[:, i]
        
            if len(match) == 2:
                self.visualize_composite_eff_props_2_phase(match, property, units, volume_fractions, effective_properties)
            elif len(match) == 3:
                self.visualize_composite_eff_props_3_phase(match, property, units, volume_fractions, effective_properties)
            elif len(match) == 4:
                self.visualize_composite_eff_props_4_phase(match, property, units, volume_fractions, effective_properties)
            else:
                warnings.warn("No visualizations available for composites with 5 or more phases.")
                return

        return 

    def visualize_composite_eff_props_2_phase(self, match, property, units, volume_fractions, effective_properties):
    
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=volume_fractions[:, 0], y=effective_properties, mode='lines'))
        
        fig.update_layout(
            xaxis_title=f"Volume fraction, {match[0]}",
            yaxis_title=f"{units}",
            title=f"{property}\n{match}",
            title_font_size=24,
            xaxis_title_font_size=20,
            yaxis_title_font_size=20,
            width=600,  
            height=600, 
            margin=dict(l=50, r=50, t=50, b=50) 
        )
        fig.show()
        
        return 

    def visualize_composite_eff_props_3_phase(self, match, property, units, volume_fractions, effective_properties):

        phase1_vol_fracs = np.unique(volume_fractions[:, 0])
        phase2_vol_fracs = np.unique(volume_fractions[:, 1])
        X, Y = np.meshgrid(phase1_vol_fracs, phase2_vol_fracs)
        Z = effective_properties.reshape(len(phase1_vol_fracs), len(phase2_vol_fracs))

        fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])
        
        fig.update_layout(
            scene=dict(
                xaxis_title=f"Volume fraction, {match[0]}",
                yaxis_title=f"Volume fraction, {match[1]}",
                zaxis_title=f"{units}",
            ),
            title=f"{property}\n{match}",
            width=600,  
            height=600, 
            margin=dict(l=50, r=50, t=50, b=50) 
        )
        fig.show()

        return 

    def visualize_composite_eff_props_4_phase(self, match, property, units, volume_fractions, effective_properties):

        phase1_vol_fracs = np.unique(volume_fractions[:, 0])
        phase2_vol_fracs = np.unique(volume_fractions[:, 1])
        phase3_vol_fracs = np.unique(volume_fractions[:, 2])
        X, Y, Z = np.meshgrid(phase1_vol_fracs, phase2_vol_fracs, phase3_vol_fracs)
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        Z_flat = Z.flatten()

        fig = go.Figure()

        fig.add_trace(go.Scatter3d(
            x=X_flat,
            y=Y_flat,
            z=Z_flat,
            mode='markers',
            marker=dict(
                size=5,
                color=effective_properties.flatten(),
                colorscale='Viridis',
                colorbar=dict(title=f"{units}"),
                opacity=0.8
            )
        ))

        fig.update_layout(
            scene=dict(
                xaxis_title=f"Volume fraction, {match[0]}",
                yaxis_title=f"Volume fraction, {match[1]}",
                zaxis_title=f"Volume fraction, {match[2]}",
            ),
            title=f"{property}\n{match}",
            title_font_size=14,
            width=600,  
            height=600, 
            margin=dict(l=50, r=50, t=50, b=50) 
        )
        fig.show()
    
    def generate_consolidated_dict(self, overall_bounds_dict: dict = {}):

        """
        MAIN FUNCTION USED TO GENERATE MATRIAL PROPERTY DICTIONARY DEPENDING ON USER REQUEST
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
        print(f"Initial query: {query}")

        # Add additional fields you want to query, like 'material_id', 'formula_pretty', and all the properties in the initial query
        fields = ['material_id', 'formula_pretty']  # Fixed fields
        fields.extend(query.keys())  # Adding all the keys from the query to the fields list
        print(f"Fields: {fields}")

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
        
        print(f"Final query: {query}")
        
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
                "k_vrh": ("bulk_modulus_vrh", "bulk_modulus", "vrh"),
                "g_vrh": ("shear_modulus_vrh", "shear_modulus", "vrh"),
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
        for key in result_dict:
            print(f"Length of {key}: {len(result_dict[key])}")

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
        print(f"Length of material_id after sorting: {len(result_dict['formula_pretty'])}")

        # Print the sorted result_dict keys and lengths of the lists
        for key in result_dict:
            print(f"Final Length of {key}: {len(result_dict[key])}")

        # print keys of the result_dict
        print(f"Keys of result_dict: {result_dict.keys()}")
        print(f"result_dict['material_id'] = {result_dict['material_id']}")


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

            print(f"Query for carrier transport: {query_carrier_transport}")

            tables = client.query_contributions({"project":"carrier_transport",
                                        "data__sigma__p__value__gt": query_carrier_transport['elec_cond_300k_low_doping'][0] / 1e15 / 1e-14, # the 1003100.0,
                                        "data__sigma__p__value__lt": query_carrier_transport['elec_cond_300k_low_doping'][1] / 1e15 / 1e-14, #2093100.0,
                                        "data__kappa__p__value__gt": query_carrier_transport['therm_cond_300k_low_doping'][0] / 1e9 / 1e-14, #7091050.0,
                                        "data__kappa__p__value__lt": query_carrier_transport['therm_cond_300k_low_doping'][1] / 1e9 / 1e-14, #8591050.0,
                                        "identifier__in": result_dict['material_id'],
                                    },
                                    fields=['identifier', 'formula', 'data.sigma.p', 'data.kappa.p'],
                                    sort='+formula') #  'identifier','data.V', 'tables', 'kappa' , 'kappa.p.value', 'sigma.p.value', '_all' (2769600.0, 1093100.0)
            
            print(f"Tables: {tables}")

            # only append the values to the corresponding material_id from the result_dict. At the end, make all the remaning values 
            # corresponding to the material_id as None
            # Iterate over the tables returned and map the data to the result_dict
            for table in tables['data']:
                print(f"Table: {table}")
                material_id = table['identifier']  # Material ID from the table
                if material_id in result_dict['material_id']:  # Only map for materials already in result_dict
                    index = result_dict['material_id'].index(material_id)
                    print(f"Index: {index}")
                    
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
        print(f"my_dict[tables] = {my_dict['tables']}")
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