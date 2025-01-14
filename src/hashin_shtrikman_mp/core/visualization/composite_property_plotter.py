"""visualizer.py."""
import itertools
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import yaml

# Custom imports
from ..genetic_algorithm import GeneticAlgorithmResult, OptimizationParams, GeneticAlgorithmParams, Population
from ..user_input import UserInput

# YAML files
sys.path.insert(1, "../io/inputs")
HS_HEADERS_YAML = "display_table_headers.yaml"

# Optimizer class defaults
MODULE_DIR = Path(__file__).resolve().parent

np.seterr(divide="raise")


class CompositePropertyPlotter():
    """
    Visualizer class for Hashin-Shtrikman optimization.

    This class extends the HashinShtrikman class to include methods
    for visualizing optimization results.
    """

    def __init__(self,
                 opt_params: OptimizationParams,
                 ga_params: GeneticAlgorithmParams) -> None:
        self.opt_params = opt_params
        self.ga_params = ga_params
        
    @classmethod
    def from_optimization_result(cls, ga_result: GeneticAlgorithmResult):
        return cls(ga_result.optimization_params, ga_result.algo_parameters)

    @classmethod
    def from_user_input(cls, user_input: UserInput):
        opt_params = OptimizationParams.from_user_input(user_input)
        return cls(opt_params, GeneticAlgorithmParams())

    def get_all_possible_vol_frac_combos(self, num_fractions: int = 30):
        all_vol_frac_ranges = []
        for _ in range(self.opt_params.num_materials - 1):
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

    def visualize_composite_eff_props(self, match, consolidated_dict: dict, num_fractions: int = 99):

        # Too much computation to use the default for 4 phase, so reduce num_fractions
        if len(match) == 4:
            num_fractions = 20
        if len(match) == 1 or len(match) > 4:
            warnings.warn("No visualizations available for composites with 5 or more phases.")
            return

        all_vol_frac_combos = self.get_all_possible_vol_frac_combos(num_fractions=num_fractions)

        material_values = []
        for category in self.opt_params.property_categories:
            for property in self.opt_params.property_docs[category]:
                for material in match:
                    if property in consolidated_dict:
                        m = consolidated_dict["material_id"].index(material)
                        material_values.append(consolidated_dict[property][m])

        # Create population of same properties for all members based on material match combination
        population_values = np.tile(material_values, (len(all_vol_frac_combos),1))

        # Only the vary the volume fractions across the population
        # Create uniform volume fractions from 0 to 1 with a spacing of 0.02 but with a shape of self.ga_params.get_num_members() & 1
        volume_fractions = np.array(all_vol_frac_combos).reshape(len(all_vol_frac_combos), self.opt_params.num_materials)

        # Include the random mixing parameters and volume fractions in the population
        values = np.c_[population_values, volume_fractions]

        # Instantiate the population and find the best performers
        # For 2 phases and x volume fractions, there are x   possible volume fraction combinations
        # For 3 phases and x volume fractions, there are x^2 possible volume fraction combinations
        # for 4 phases and x volume fractions, there are x^3 possible volume fraction combinations
        this_pop_ga_params = self.ga_params
        this_pop_ga_params.num_members = num_fractions**(len(match) - 1)
        population = Population(optimization_params=self.opt_params,
                                values=values,
                                ga_params=this_pop_ga_params)
        all_effective_properties = population.get_effective_properties()
        print(f"all_effective_properties.shape: {all_effective_properties.shape}")

        # Get property strings for labeling the plot(s)
        file_name = f"{MODULE_DIR}/../io/inputs/{HS_HEADERS_YAML}"
        property_strings = []
        with open(file_name) as stream:
            data = yaml.safe_load(stream)
            for category, properties in data["Per Material"].items():
                if category in self.opt_params.property_categories:
                    for property in properties.values():
                        property_strings.append(property)

        def extract_property(text):
            match = re.match(r"([^,]+), \[.*\]", text)
            if match:
                return match.group(1).strip()
            return None

        def extract_units(text):
            match = re.search(r"\[.*?\]", text)
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
        fig.add_trace(go.Scatter(x=volume_fractions[:, 0], y=effective_properties, mode="lines"))

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


    def visualize_composite_eff_props_3_phase(self, match, property, units, volume_fractions, effective_properties):

        phase1_vol_fracs = np.unique(volume_fractions[:, 0])
        phase2_vol_fracs = np.unique(volume_fractions[:, 1])
        X, Y = np.meshgrid(phase1_vol_fracs, phase2_vol_fracs)
        Z = effective_properties.reshape(len(phase1_vol_fracs), len(phase2_vol_fracs))

        fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale="Viridis")])

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
            mode="markers",
            marker=dict(
                size=5,
                color=effective_properties.flatten(),
                colorscale="Viridis",
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
