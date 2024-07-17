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

sys.path.insert(1, '../log')
from custom_logger import logger

# Custom imports
from genetic_algo import GAParams
from member import Member
from population import Population

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
from compile_cost_calculation_formulas import compile_formulas
COMPILED_CALC_GUIDE = compile_formulas(loadfn(f"{MODULE_DIR}/../io/inputs/{CALC_GUIDE}"))

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
    user_input: Dict = Field(
        default_factory=dict, 
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
            
            # Update values accordingly
            values.update({
                "desired_props":  desired_props,
                "lower_bounds":   lower_bounds,
                "upper_bounds":   upper_bounds,
                "num_properties": num_properties,
                "num_materials":  num_materials
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
    def get_headers(self, include_mpids=False, file_name = f"{MODULE_DIR}/../io/inputs/{HS_HEADERS_YAML}"):

        with open(file_name, 'r') as stream:
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
    
    def get_material_matches(self, consolidated_dict: dict = {}): 

        best_designs_dict = self.get_dict_of_best_designs()       
        if consolidated_dict == {}:  
            with open("consolidated_dict_02_11_2024_23_45_58") as f: # TODO get from latest final_dict file: change this to a method that reads from the latest MP database
                consolidated_dict = json.load(f)

        # Initialize list of sets for matching indices
        matches = []
        for _ in range(self.num_materials):
            matches.append(set(range(len(consolidated_dict["material_id"]))))

        # Helper function to get matching indices based on property extrema
        def get_matching_indices(property, bounds_dict, mat_key):
            lower_bound = min(bounds_dict[mat_key][property])
            upper_bound = max(bounds_dict[mat_key][property])
            return {i for i, value in enumerate(consolidated_dict[property]) if lower_bound <= value <= upper_bound}

        # Iterate over categories and properties
        for category in self.property_categories:
            if category in self.property_docs:
                for property in self.property_docs[category]:
                    if property in consolidated_dict:  # Ensure property exists in consolidated_dict
                        for m in range(self.num_materials):
                            matches[m] &= get_matching_indices(property, best_designs_dict[f"mat{m+1}"], category)                        

        # Extract mp-ids based on matching indices
        matches_dict = {}
        for m in range(self.num_materials):
            matches_dict[f"mat{m+1}"] = [consolidated_dict["material_id"][i] for i in matches[m]]

        return matches_dict     
    
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
        
        for combo in material_combinations:

            material_values = []
            mat_ids = np.zeros((len(material_combinations), self.num_materials))

            for category in self.property_categories:
                for property in self.property_docs[category]:
                    for material in combo:
                        if property in consolidated_dict.keys(): # TODO carrier-transport not registering ??
                            m = consolidated_dict["material_id"].index(material)               
                            material_values.append(consolidated_dict[property][m])
                        else:
                            material_values.append(1.0) # TODO remove later, this is for debugging

            # Create population of same properties for all members based on material match combination
            population_values = np.tile(material_values, (len(all_vol_frac_combos),1))

            # Only the vary the volume fractions across the population
            # Create uniform volume fractions from 0 to 1 with a spacing of 0.02 but with a shape of self.ga_params.get_num_members() & 1
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
            for material in combo:
                mat_ids.append(np.reshape([material]*self.ga_params.num_members, (self.ga_params.num_members,1)))
            mat_ids = np.column_stack(mat_ids)
            table_data = np.c_[mat_ids, population.values, sorted_costs] 
            print("\nMATERIALS PROJECT PAIRS AND HASHIN-SHTRIKMAN RECOMMENDED VOLUME FRACTION")
            print(tabulate(table_data[0:5, :], headers=self.get_headers(include_mpids=True))) # hardcoded to be 5 rows, could change

    #------ Setter Methods ------#
    @staticmethod
    def set_num_materials_from_user_input(user_input):
        num_materials = len(user_input) - 1
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
        bounds = {}
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

    def set_fields(self, fields):
        self.fields = fields
        return self 
 
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
                                     num_members=self.ga_params.num_members)

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
            members_minus_parents_minus_kids = num_members - num_parents - num_kids
            population.set_random_values(lower_bounds=self.lower_bounds, 
                                         upper_bounds=self.upper_bounds, 
                                         num_members=members_minus_parents_minus_kids)

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
        print("\nHASHIN-SHTRIKMAN + GENETIC ALGORITHM RECOMMENDED MATERIAL PROPERTIES")
        print(tabulate(table_data, headers=self.get_headers()))
    
    def plot_optimization_results(self):
        fig, ax = plt.subplots(figsize=(10,6))
        ax.plot(range(self.ga_params.num_generations), self.avg_parent_costs, label="Avg. of top 10 performers")
        ax.plot(range(self.ga_params.num_generations), self.lowest_costs, label="Best costs")
        plt.xlabel("Generation", fontsize= 20)
        plt.ylabel("Cost", fontsize=20)
        plt.title("Genetic Algorithm Results", fontsize = 24)
        plt.legend(fontsize = 14)
        plt.show()  

    # IN PROGRESS
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
            yaxis_title_font_size=20
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
            title_font_size=14
        )
        fig.show()
    
    def generate_consolidated_dict(self, total_docs = None):

        """
        MAIN FUNCTION USED TO GENERATE MATRIAL PROPERTY DICTIONARY DEPENDING ON USER REQUEST
        """

        if "carrier-transport" in self.property_categories:
            client = Client(apikey=self.api_key, project=self.mp_contribs_project)
        else:
            client = Client(apikey=self.api_key)
        
        new_fields = copy.deepcopy(self.fields)

        # Iterate over the user input to dynamically update self.fields based on requested property categories
        # Iterate over property categories and update new_fields based on mp_property_docs
        for category in self.property_categories:
            if category in self.property_docs:
                if category == "carrier-transport":
                    new_fields["mp-ids-contrib"] = []

                for prop in self.property_docs[category]:
                    # Initialize empty list for each property under the category
                    new_fields[prop] = []

        self.set_fields(new_fields)

        logger.info(f"self.fields: {self.fields}")

        with MPRester(self.api_key) as mpr:
            
            docs = mpr.materials.summary.search(fields=self.fields)
            from mpi4py import MPI

            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()

            # Calculate the size of each chunk
            if total_docs is None:
                total_docs = len(docs)
                chunk_size = len(docs) // size
            elif isinstance(total_docs, int):
                chunk_size = total_docs // size

            # Calculate the start and end indices for this process"s chunk
            start = rank * chunk_size
            end = start + chunk_size if rank != size - 1 else total_docs  # The last process gets the remainder

            # Each process gets a different chunk
            chunk = docs[start:end]

            # for i, doc in enumerate(docs):
            for i, doc in enumerate(chunk[0:total_docs]):

                logger.info(f"Process {rank}: {i} of {len(chunk[0:total_docs])}")

                required_fields = [doc.material_id, doc.is_stable, doc.is_metal]

                for category in self.property_categories:
                    if category in self.property_docs:
                        for prop, value in self.property_docs[category].items():
                            if category == "carrier-transport":
                                try:
                                    mp_id = doc.material_id                           
                                    query = {"identifier": mp_id}
                                    my_dict = client.download_contributions(query=query, include=["tables"])[0]
                                    required_fields.append(my_dict["identifier"])
                                except IndexError:
                                    continue
                            elif category == "elastic":
                                prop_value = getattr(doc, prop, None)
                                # For "elastic", you expect 'value' to be directly usable
                                if value is not None:  # 'voigt' or similar
                                    if prop_value is not None:
                                        prop_value = prop_value.get(value, None)
                                required_fields.append(prop_value)
                            else:
                                # For other categories, just append the property name for now
                                prop_value = getattr(doc, prop, None)
                                required_fields.append(prop_value)

                if all(field is not None for field in required_fields):

                    for c in DEFAULT_FIELDS.keys():
                        self.fields[c].append(getattr(doc, c, None))
                    
                    for category in self.property_categories:
                        if category in self.property_docs:
                            if category == "carrier-transport":
                                self.fields["mp-ids-contrib"].append(my_dict["identifier"])
                                
                            for prop, value in self.property_docs[category].items():
                                
                                # Carrier transport
                                if category == "carrier-transport":
                                    self.mp_contribs_prop(prop, my_dict)
                                
                                # Elastic
                                elif category == "elastic":
                                    prop_value = getattr(doc, prop, None)
                                    if value is not None:  # 'voigt' or similar
                                        if prop_value is not None:
                                            prop_value = prop_value.get(value, None)
                                    self.fields[prop].append(prop_value)
                                
                                # All other property categories
                                else:
                                    for prop in self.property_docs[category]:
                                        # Dynamically get the property value from the doc
                                        prop_value = getattr(doc, prop, None)  # Returns None if prop doesn't exist
                                        # Initialize empty list for each property under the category
                                        self.fields[prop].append(prop_value)
        
            # comm.gather the self.fields data after the for loop
            gathered_fields = comm.gather(self.fields, root=0)

            # On process 0, consolidate self.fields data into a single dictionary -- consolidated_dict
            if rank == 0:
                consolidated_dict: Dict[str, List[Any]] = {}
                if gathered_fields is not None:
                    for fields in gathered_fields:
                        for key, value in fields.items():
                            if key in consolidated_dict:
                                consolidated_dict[key].extend(value)
                            else:
                                consolidated_dict[key] = value

                # Save the consolidated results to a JSON file
                now = datetime.now()
                my_file_name = f"{MODULE_DIR}/../io/outputs/consolidated_dict_" + now.strftime("%m_%d_%Y_%H_%M_%S")
                with open(my_file_name, "w") as my_file:
                    json.dump(consolidated_dict, my_file)

        return consolidated_dict
    
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

        prop_str = my_dict["tables"][table_column].iloc[2, 0]
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