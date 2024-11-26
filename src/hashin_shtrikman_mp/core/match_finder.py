#import copy
import itertools
import json
#import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import re
import sys
import warnings
import yaml

from datetime import datetime
from matplotlib import cm
#import matplotlib.gridspec as gridspec
from monty.serialization import loadfn
from mp_api.client import MPRester
#from mpcontribs.client import Client
#from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
#from pydantic import BaseModel, Field, model_validator
#rom tabulate import tabulate
#rom typing import Any, Dict, List, Union, Optional
from typing import Any, Dict, List

# Custom imports
from hashin_shtrikman_mp.core.genetic_algo import GAParams
from hashin_shtrikman_mp.core.member import Member
from hashin_shtrikman_mp.core.population import Population
from hashin_shtrikman_mp.core.user_input import UserInput
from hashin_shtrikman_mp.core.optimizer import Optimizer
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
#from compile_cost_calculation_formulas import compile_formulas
#COMPILED_CALC_GUIDE = compile_formulas(loadfn(f"{MODULE_DIR}/../io/inputs/{CALC_GUIDE}"))

np.seterr(divide='raise')

class MatchFinder(Optimizer):
    """
    MatchFinder class for Hashin-Shtrikman optimization.

    This class extends the HashinShtrikman class to include methods 
    for finding real materials in the MP database which match
    (fictitious) materials recommended by the optimization.
    """

    def __init__(self, optimizer: Optimizer):
        # Dump the optimizer's data into a dictionary
        optimizer_attributes = optimizer.model_dump()

        # Ensure user_input is passed as an instance of UserInput
        if isinstance(optimizer_attributes.get("user_input"), dict):
            optimizer_attributes["user_input"] = UserInput(**optimizer_attributes["user_input"])

        # Ensure ga_params is passed as an instance of GAParams
        if isinstance(optimizer_attributes.get("ga_params"), dict):
            optimizer_attributes["ga_params"] = GAParams(**optimizer_attributes["ga_params"])

        # Pass all the optimizer attributes (including user_input, ga_params) to the parent class
        super().__init__(**optimizer_attributes)

    # To use np.ndarray or other arbitrary types in your Pydantic models
    class Config:
        arbitrary_types_allowed = True

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
        #print(f"best_designs_dict: {best_designs_dict}")
        #print(f"keys of best_designs_dict: {best_designs_dict.keys()}")     
        #print(f"keys of best_designs_dict['mat1']: {best_designs_dict['mat1'].keys()}")
        #print(f"overall_bounds_dict: {overall_bounds_dict}")
        
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
            best_design_props = {
                'elec_cond': mat_data['carrier-transport']['elec_cond_300k_low_doping'],
                'therm_cond': mat_data['carrier-transport']['therm_cond_300k_low_doping'],
                'bulk_modulus': mat_data['elastic']['bulk_modulus'],
                'shear_modulus': mat_data['elastic']['shear_modulus'],
                'universal_anisotropy': mat_data['elastic']['universal_anisotropy']
            }
            
            # Initialize an empty list to store matching materials for the current mat_key
            matching_materials_for_current_mat = []

            # Print len of material_id, elec_cond, therm_cond, bulk_modulus, shear_modulus, universal_anisotropy
            #print(f"len(material_id): {len(consolidated_dict['material_id'])}")
            #print(f"len(elec_cond): {len(consolidated_dict['elec_cond_300k_low_doping'])}")
            #print(f"len(therm_cond): {len(consolidated_dict['therm_cond_300k_low_doping'])}")
            #print(f"len(bulk_modulus): {len(consolidated_dict['bulk_modulus_vrh'])}")
            #print(f"len(shear_modulus): {len(consolidated_dict['shear_modulus_vrh'])}")
            #print(f"len(universal_anisotropy): {len(consolidated_dict['universal_anisotropy'])}")
            

            # Iterate through each material in consolidated_dict
            for i, material_id in enumerate(consolidated_dict["material_id"]):

                # Convert material_id to a string before storing it
                material_id_str = str(material_id)
                
                # Retrieve the properties for this material from consolidated_dict
                material_props = {
                    'elec_cond': consolidated_dict['elec_cond_300k_low_doping'][i],
                    'therm_cond': consolidated_dict['therm_cond_300k_low_doping'][i],
                    'bulk_modulus': consolidated_dict['bulk_modulus'][i],
                    'shear_modulus': consolidated_dict['shear_modulus'][i],
                    'universal_anisotropy': consolidated_dict['universal_anisotropy'][i]
                }

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

        if matches_dict == {}:
            print("No materials match the recommended composite formulation.")
            return 

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