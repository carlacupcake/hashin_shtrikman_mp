#import copy
import itertools
import json
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import re
import sys
import warnings
import yaml

#from datetime import datetime
#from matplotlib import cm
#import matplotlib.gridspec as gridspec
from monty.serialization import loadfn
#from mp_api.client import MPRester
#from mpcontribs.client import Client
#from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
#from pydantic import BaseModel, Field, model_validator
#from tabulate import tabulate
from typing import Any, Dict, List, Union, Optional

#sys.path.insert(1, '../log')
#from custom_logger import logger

# Custom imports
from hashin_shtrikman_mp.core.genetic_algo import GAParams
from hashin_shtrikman_mp.core.member import Member
from hashin_shtrikman_mp.core.population import Population
from hashin_shtrikman_mp.core.user_input import UserInput
from hashin_shtrikman_mp.core.optimizer import Optimizer

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

# Custom imports
from .optimizer import Optimizer

class Visualizer(Optimizer):
    """
    Visualizer class for Hashin-Shtrikman optimization.

    This class extends the HashinShtrikman class to include methods 
    for visualizing optimization results.
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

    def plot_cost_func_contribs(self):

        # Get the best design
        best_design = Member(num_materials=self.num_materials, 
                     num_properties=self.num_properties,
                     values=self.final_population.values[0], 
                     property_categories=self.property_categories,
                     property_docs=self.property_docs, 
                     desired_props=self.desired_props, 
                     ga_params=self.ga_params,
                     calc_guide=self.calc_guide)
        cost, costs_eff_props, costs_cfs = best_design.get_cost(include_cost_breakdown=True)
        print(f'Cost of best design: {cost}')

        # Scale the costs of the effective properties and concentration factors
        # Scale according to weights from GAParams
        scaled_costs_eff_props = 1/2 * self.ga_params.weight_eff_prop * costs_eff_props
        scaled_costs_cfs = 1/2 * self.ga_params.weight_conc_factor * costs_cfs

        # Labels for the pie chart 
        eff_prop_labels = []
        cf_labels = []
        for category in self.property_categories:
            for property in self.property_docs[category]:
                eff_prop_labels.append(f'eff. {property}')
                if property == 'bulk_modulus':
                    cf_labels.append(f'cf hydrostatic stress')
                elif property == 'shear_modulus':
                    cf_labels.append(f'cf deviatoric stress')
                else:
                    cf_labels.append(f'cf load on {property}')
                    cf_labels.append(f'cf response from {property}')
        labels = eff_prop_labels + cf_labels

        # Combine the data and labels for the eff props and concentration factors
        scaled_costs_eff_props = np.array(scaled_costs_eff_props)
        scaled_costs_cfs = np.array(scaled_costs_cfs)
        cost_func_contribs = np.concatenate((scaled_costs_eff_props, scaled_costs_cfs)) 

        # Create the pie chart figure 
        fig = go.Figure(data=[go.Pie(labels=labels, values=cost_func_contribs, 
                                     textinfo='percent', 
                                     insidetextorientation='radial', 
                                     hole=.25)])

        fig.update_layout(
            title_text='Cost Function Contributions',
            showlegend=True
        )

        # Display the chart
        fig.show()