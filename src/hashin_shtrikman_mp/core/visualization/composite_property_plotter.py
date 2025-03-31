"""composite_property_plotter.py."""
import itertools
import re
import sys
import warnings
import numpy as np
import plotly.graph_objects as go
import yaml

from pathlib import Path
from scipy.interpolate import griddata

from ..genetic_algorithm import GeneticAlgorithmResult, Population


# YAML files
sys.path.insert(1, "../io/inputs/")
HS_HEADERS_YAML = "display_table_headers.yaml"

# Optimizer class defaults
MODULE_DIR = Path(__file__).resolve().parent

np.seterr(divide="raise")


class CompositePropertyPlotter():
    """
    Visualization class for phase diagrams.
    """

    def __init__(self, ga_result: GeneticAlgorithmResult) -> None:
        self.ga_params = ga_result.algo_parameters
        self.opt_params = ga_result.optimization_params
        self.ga_result = ga_result


    @classmethod
    def from_optimization_result(cls, ga_result: GeneticAlgorithmResult):
        return cls(ga_result)


    def get_all_possible_vol_frac_combos(self, num_fractions: int = 30):
        """
        Generates all possible unique volume fraction combinations for a given number of materials, 
        ensuring that the sum of fractions equals 1. 

        Args:
            num_fractions (int, optional)

        Returns
        -------
            all_vol_frac_combos (list of list of float)
        """

        spacing = np.linspace(0.0, 1.0, num_fractions + 1)
        combinations = itertools.product(spacing, repeat=self.opt_params.num_materials)

        # Convert to set to ensure uniqueness, tuples are hashable
        unique_combos = {tuple(combo) for combo in combinations if np.isclose(sum(combo), 1)}

        # Convert tuples back to lists
        all_vol_frac_combos = [list(combo) for combo in unique_combos]

        return all_vol_frac_combos


    def visualize_composite_eff_props(self,
                                      match,
                                      consolidated_dict: dict,
                                      num_fractions: int = 49):
        """
        Generates visualizations of effective properties for composite materials.

        Args:
            match (list)
            - A list of material identifiers used to construct the composite.
            consolidated_dict (dict)
            - A dictionary containing material property values with keys as 
              property names and values as lists of corresponding data.
            num_fractions (int, optional)
            - The number of discrete volume fraction values to consider.

        Returns
        -------
            None

        Notes:
            - The function does not support visualization for 
              single-phase or five-or-more-phase composites.
        """

        # Too much computation to use the default for 4 phase, so reduce num_fractions
        if len(match) == 1:
            warnings.warn(UserWarning(
                "No visualizations available for single materials (not a composite)."
                ))
            return
        if len(match) == 4:
            num_fractions = 20
        if len(match) > 4:
            warnings.warn(UserWarning(
                "No visualizations available for composites with 5 or more phases."
                ))
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
        population_values = np.tile(material_values, (len(all_vol_frac_combos), 1))

        # Only the vary the volume fractions across the population
        # Create uniform volume fractions from 0 to 1 with a spacing of 1/num_fractions
        volume_fractions = np.array(all_vol_frac_combos).reshape(len(all_vol_frac_combos),
                                                                 self.opt_params.num_materials)

        # Include the random mixing parameters and volume fractions in the population
        values = np.c_[population_values, volume_fractions]

        # Instantiate the population and find the best performers
        this_pop_ga_params = self.ga_params
        this_pop_ga_params.num_members = len(all_vol_frac_combos)#num_fractions**(len(match) - 1)
        population = Population(optimization_params=self.opt_params,
                                values=values,
                                ga_params=this_pop_ga_params)
        all_effective_properties = population.get_effective_properties()

        # Get property strings for labeling the plot(s)
        file_name = MODULE_DIR.joinpath("../../io/inputs/data", HS_HEADERS_YAML).resolve()
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

            prop = extract_property(property_string)
            units = extract_units(property_string)
            effective_properties = all_effective_properties[:, i]

            if len(match) == 2:
                self.visualize_composite_eff_props_2_phase(match,
                                                           prop,
                                                           units,
                                                           volume_fractions,
                                                           effective_properties)
            elif len(match) == 3:
                self.visualize_composite_eff_props_3_phase(match,
                                                           prop,
                                                           units,
                                                           volume_fractions,
                                                           effective_properties)
            elif len(match) == 4:
                self.visualize_composite_eff_props_4_phase(match,
                                                           prop,
                                                           units,
                                                           volume_fractions,
                                                           effective_properties)
            else:
                warnings.warn(
                    f"No visualizations available for composites with {len(match)} phase(s)."
                )
                return

        return


    def visualize_composite_eff_props_2_phase(self,
                                              match,
                                              prop,
                                              units,
                                              volume_fractions,
                                              effective_properties):
        """
        Generates a 2D line plot to visualize the effective properties of a two-phase composite. 

        Args:
            match (list)
            property (str)
            units (str)
            volume_fractions (ndarray)
            effective_properties (ndarray)

        Returns
        -------
            fig (plotly.graph_objects.Figure)
        """

        sorted_indices = np.argsort(volume_fractions[:, 0])
        sorted_volume_fractions = volume_fractions[sorted_indices]
        sorted_effective_properties = effective_properties[sorted_indices]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sorted_volume_fractions[:, 0],
                                 y=sorted_effective_properties,
                                 mode="lines"))

        fig.update_layout(
            xaxis_title=f"Volume fraction, {match[0]}",
            yaxis_title=f"{units}",
            title=f"{prop}\n{match}",
            title_font_size=24,
            xaxis_title_font_size=20,
            yaxis_title_font_size=20,
            width=600,
            height=600,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        fig.show()
        return fig


    def visualize_composite_eff_props_3_phase(self,
                                              match,
                                              prop,
                                              units,
                                              volume_fractions,
                                              effective_properties):
        """
        Generates a 3D surface plot to visualize the effective properties
        of a three-phase composite. 

        Args:
            match (list)
            property (str)
            units (str)
            volume_fractions (ndarray)
            effective_properties (ndarray)

        Returns
        -------
            fig (plotly.graph_objects.Figure)
        """

        # Interpolate scattered effective property data to grid
        x_data, y_data = np.mgrid[0:1:100j, 0:1:100j]
        z_data = griddata(volume_fractions[:, 0:-1], effective_properties, (x_data, y_data), method='linear')

        fig = go.Figure(data=[go.Surface(z=z_data, x=x_data, y=y_data, colorscale="Viridis")])

        fig.update_layout(
            scene=dict(
                xaxis_title=f"Volume fraction, {match[0]}",
                yaxis_title=f"Volume fraction, {match[1]}",
                zaxis_title=f"{units}",
            ),
            title=f"{prop}\n{match}",
            width=600,
            height=600,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        fig.show()
        return fig


    def visualize_composite_eff_props_4_phase(self,
                                              match,
                                              prop,
                                              units,
                                              volume_fractions,
                                              effective_properties):
        """
        Generates a 3D scatter plot to visualize the effective properties of a four-phase composite. 

        Args:
            match (list)
            property (str)
            units (str)
            volume_fractions (ndarray)
            effective_properties (ndarray)

        Returns
        -------
            fig (plotly.graph_objects.Figure)
        """

        # Create a 3D surface plot
        fig = go.Figure()
        fig = go.Figure(data=[go.Scatter3d(
            x=volume_fractions[:, 0],
            y=volume_fractions[:, 1],
            z=volume_fractions[:, 2],
            mode="markers",
            marker=dict(
                size=5,
                color=effective_properties,
                colorscale="Viridis",
                colorbar=dict(title=f"{units}"),
                opacity=0.8
            )
        )])

        fig.update_layout(
            scene=dict(
                xaxis_title=f"Volume fraction, {match[0]}",
                yaxis_title=f"Volume fraction, {match[1]}",
                zaxis_title=f"Volume fraction, {match[2]}",
            ),
            title=f"{prop}\n{match}",
            title_font_size=14,
            width=600,
            height=600,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        fig.show()
        return fig
