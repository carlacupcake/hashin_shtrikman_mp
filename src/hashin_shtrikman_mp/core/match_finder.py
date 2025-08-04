"""match_finder.py."""
import itertools
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
from mp_api.client import MPRester

from .genetic_algorithm import GeneticAlgorithmResult, Population
from .utilities import get_headers, load_property_docs

# YAML files
sys.path.insert(1, "../io/inputs")
HS_HEADERS_YAML = "display_table_headers.yaml"

# Optimizer class defaults
MODULE_DIR = Path(__file__).resolve().parent

np.seterr(divide="raise")


class MatchFinder:
    """
    A class which uses optimization results to find real material matches
    in the Materials Project databases, using the Materials Project API.
    """


    def __init__(self,
                 ga_result: GeneticAlgorithmResult) -> None:
        self.optimization_params = ga_result.optimization_params
        self.optimized_population = ga_result.final_population
        self.ga_params = ga_result.algo_parameters


    def get_dict_of_best_designs(self) -> dict:
        """
        Constructs a dictionary containing the best designs found.

        Returns
        -------
            best_designs_dict (dict)
        """

        # Initialize dictionaries for each material based on selected property categories
        best_designs_dict = {}
        for m in range(1, self.optimization_params.num_materials + 1):
            best_designs_dict[f"mat{m}"] = {}

        # Initialize the structure for each category
        for category in self.optimization_params.property_categories:
            for mat in best_designs_dict:
                best_designs_dict[mat][category] = {}
                for prop in self.optimization_params.property_docs[category]:
                    best_designs_dict[mat][category][prop] = []

        [unique_values, unique_eff_props, unique_costs] = self.optimized_population.get_unique_designs()

        # Populate the dictionary with unique design values
        # The last num_materials entries are volume fractions, not material properties
        stop = -self.optimization_params.num_materials

        # subtract 1 so as not to include volume fraction
        step = self.optimization_params.num_properties - 1

        for i, _ in enumerate(unique_costs):
            idx = 0
            for category in self.optimization_params.property_categories:
                for prop in self.optimization_params.property_docs[category]:
                    all_phase_props = unique_values[i][idx:stop:step]
                    for m, mat in enumerate(best_designs_dict.keys()):
                        best_designs_dict[mat][category][prop].append(all_phase_props[m])
                    idx += 1

        return best_designs_dict


    def get_material_matches(self,
                             overall_bounds_dict: dict = None,
                             consolidated_dict: dict = None,
                             threshold: float = 1) -> dict:
        """
        Identifies materials in the MP database which match those recommended by the optimization.

        Args:
            overall_bounds_dict (dict, optional)
            consolidated_dict (dict, optional)
            threshold (float, optional)
            - Should be between 0 and 1, by default 1

        Returns
        -------
            final_matching_materials (dict)
            - Keys are fake materials recommended by the genetic algorithm
            - Values are mp-ids of real materials
        """
        # Make sure overall_bounds_dict is defined
        if overall_bounds_dict is None:
            overall_bounds_dict = {}

        # Generate the consolidated dict based on overall bounds
        if consolidated_dict is None:
            consolidated_dict = self.generate_consolidated_dict(overall_bounds_dict)

        # Generate a dictionary of the best designs - same format as consolidated_dict
        best_designs_dict = self.get_dict_of_best_designs()

        # Initialize an empty dictionary to store the output in the required format
        final_matching_materials = {}

        # Iterate through each material in best_designs_dict
        for mat_key, mat_data in best_designs_dict.items():

            # Extract the property data from best_designs_dict
            best_design_props = {
                "elec_cond": mat_data["carrier-transport"]["elec_cond_300k_low_doping"],
                "therm_cond": mat_data["carrier-transport"]["therm_cond_300k_low_doping"],
                "bulk_modulus": mat_data["elastic"]["bulk_modulus"],
                "shear_modulus": mat_data["elastic"]["shear_modulus"],
                "universal_anisotropy": mat_data["elastic"]["universal_anisotropy"]
            }

            # Initialize an empty list to store matching materials for the current mat_key
            matching_materials_for_current_mat = []

            # Iterate through each material in consolidated_dict
            for i, material_id in enumerate(consolidated_dict["material_id"]):

                # Convert material_id to a string before storing it
                material_id_str = str(material_id)

                # Retrieve the properties for this material from consolidated_dict
                material_props = {
                    "elec_cond": consolidated_dict["elec_cond_300k_low_doping"][i],
                    "therm_cond": consolidated_dict["therm_cond_300k_low_doping"][i],
                    "bulk_modulus": consolidated_dict["bulk_modulus"][i],
                    "shear_modulus": consolidated_dict["shear_modulus"][i],
                    "universal_anisotropy": consolidated_dict["universal_anisotropy"][i]
                }

                # Compare properties with best_designs_dict (within % threshold)
                matching = {}
                for prop_key, values in best_design_props.items():
                    if prop_key in material_props:
                        # Iterate through each value in best_design_props for comparison
                        for value in values:
                            if abs(value - material_props[prop_key]) / value < threshold:
                                matching[prop_key] = material_props[prop_key]

                # If all the props are within the threshold,
                # add the material_id to the matching_materials_for_current_mat
                if len(matching) == len(best_design_props):
                    matching_materials_for_current_mat.append({material_id_str: matching})

            # If any matches were found for this mat_key, add them to the final output
            if matching_materials_for_current_mat:
                final_matching_materials[mat_key] = matching_materials_for_current_mat

        return final_matching_materials

    def get_all_possible_vol_frac_combos(self, num_fractions: int = 30) -> list:
        """
        Computes the optimal volume fractions of known materials.

        Once real materials have been identified, we must calculate
        which volume fraction combinations are 'best' for the composite.

        Args:
            num_fractions (int)

        Returns
        -------
            all_vol_frac_combos (list)
        """

        spacing = np.linspace(0.0, 1.0, num_fractions + 1)
        combinations = itertools.product(spacing, repeat=self.optimization_params.num_materials)

        # Convert to set to ensure uniqueness, tuples are hashable
        unique_combos = {tuple(combo) for combo in combinations if np.isclose(sum(combo), 1)}

        # Convert tuples back to lists
        all_vol_frac_combos = [list(combo) for combo in unique_combos]

        return all_vol_frac_combos

    def generate_consolidated_dict(self, overall_bounds_dict: dict = None) -> dict:
        """
        MAIN FUNCTION USED TO GENERATE MATERIAL PROPERTY DICTIONARY DEPENDING ON USER REQUEST.
        
        Args:
            overall_bounds_dict (dict)
            - Dictionary of upper and lower bounds for material search
            - User-defined

        Returns
        -------
            consolidated_dict (dict)
            - mp-ids with properties which match the bounds criteria
        """

        # Base query initialization
        if overall_bounds_dict is None:
            overall_bounds_dict = {}
        query = {}

        # Iterate over the properties in the overall_bounds_dict and dynamically build the query
        for prop, bounds in overall_bounds_dict.items():
            # Skip 'elec_cond_300k_low_doping' and 'therm_cond_300k_low_doping'
            if prop in ["elec_cond_300k_low_doping", "therm_cond_300k_low_doping"]:
                continue  # Skip the current iteration

            # Proceed if 'upper_bound' and 'lower_bound' exist for the property
            if "upper_bound" in bounds and "lower_bound" in bounds:
                query[prop] = (bounds["lower_bound"], bounds["upper_bound"])

        # Add additional fields you want to query, like 'material_id', 'formula_pretty',
        # and all the properties in the initial query
        fields = ["material_id", "formula_pretty"]  # Fixed fields
        fields.extend(query.keys())  # Adding all the keys from the query to the fields list

        # Change 'bulk_modulus' to 'k_voigt'
        if "bulk_modulus" in query:
            query["k_vrh"] = query.pop("bulk_modulus")

        # Change 'shear_modulus' to 'g_voigt'
        if "shear_modulus" in query:
            query["g_vrh"] = query.pop("shear_modulus")

        # Change 'universal_anisotropy' to 'elastic_anisotropy'
        if "universal_anisotropy" in query:
            query["elastic_anisotropy"] = query.pop("universal_anisotropy")

        # change 'e_ij_max' to 'piezoelectric_modulus'
        if "e_ij_max" in query:
            query["piezoelectric_modulus"] = query.pop("e_ij_max")

        mpr = MPRester("QePM93qZsMKNPkI4fEYaJfB7dONoQjaM")

        # Perform the query on the Materials Project database using the built query
        materials = mpr.materials.summary.search(
            **query,  # Dynamically passing the property bounds as query filters
            fields=fields,
            # num_chunks=100
        )

        mp_property_docs = load_property_docs()

        # Initialize dictionary to hold the desired data format
        result_dict: dict[str, list[Any]] = {
            "material_id": [],
            "formula_pretty": []
        }

        # Traverse the YAML structure to get all the keys
        for category, properties in mp_property_docs.items():
            # Check if the category is present in self.optimization_params.property_categories
            if category in self.optimization_params.property_categories:
                for prop, subprop in properties.items():
                    # Append if there's a subproperty (e.g., voigt for bulk_modulus)
                    if isinstance(subprop, str):
                        result_dict[f"{prop}_{subprop}"] = []
                    else:
                        # Otherwise, append the main property
                        result_dict[prop] = []

        # remove all the rows that have None values
        materials = [material for material in materials
                     if all(getattr(material, field, None) is not None for field in fields)]

        # Extract data and organize it into the result_dict
        for material in materials:
            result_dict["material_id"].append(material.material_id)
            result_dict["formula_pretty"].append(material.formula_pretty)

            # Define a mapping between query keys and result_dict keys
            # and their corresponding material attributes
            property_map = {
                "k_vrh": ("bulk_modulus_vrh",
                          "bulk_modulus",
                          "vrh"),
                "g_vrh": ("shear_modulus_vrh",
                          "shear_modulus",
                          "vrh"),
                "elastic_anisotropy": ("universal_anisotropy",
                                       "universal_anisotropy"),
                "elec_cond_300k_low_doping": ("elec_cond_300k_low_doping",
                                              "elec_cond_300k_low_doping"),
                "therm_cond_300k_low_doping": ("therm_cond_300k_low_doping",
                                               "therm_cond_300k_low_doping"),
                "e_electronic": ("e_electronic",
                                 "e_electronic"),
                "e_ionic": ("e_ionic",
                            "e_ionic"),
                "e_total": ("e_total",
                            "e_total"),
                "n": ("n",
                      "n"),
                "total_magnetization": ("total_magnetization",
                                        "total_magnetization"),
                "total_magnetization_normalized_vol": ("total_magnetization_normalized_vol",
                                                       "total_magnetization_normalized_vol"),
                "e_ij_max": ("e_ij_max",
                             "e_ij_max")
            }

            # Iterate over the properties in the query and append values to result_dict dynamically
            for prop, (result_key, material_attr, *sub_attr) in property_map.items():
                if prop in query:

                    # Check if there's a sub-attribute (e.g., "voigt" in "bulk_modulus")
                    if sub_attr:
                        # Access sub-attribute if it exists
                        value = getattr(material, material_attr, {})
                        result_dict[result_key].append(value.get(sub_attr[0], None))
                    else:
                        # Direct access to attribute
                        result_dict[result_key].append(getattr(material, material_attr, None))

        # Initialize variables
        formula_pretty_length = len(result_dict["formula_pretty"])

        # Filter out incomplete or empty lists that don't need sorting
        non_empty_keys = [key for key in result_dict
                          if len(result_dict[key]) == formula_pretty_length]

        # Sort the result_dict by ascending order of material_id for non-empty lists
        sorted_indices = sorted(range(formula_pretty_length),
                                key=lambda i: result_dict["formula_pretty"][i])

        # Re-arrange all the properties in result_dict based on the sorted indices,
        # but only for non-empty lists
        for key in non_empty_keys:
            result_dict[key] = [result_dict[key][i] for i in sorted_indices]

        # for all the empty lists, append None to the corresponding material_id
        for key in result_dict:
            if key not in non_empty_keys:
                result_dict[key] = [None] * formula_pretty_length

        if "carrier-transport" in self.optimization_params.property_categories:
            from mpcontribs.client import Client
            client = Client(apikey="QePM93qZsMKNPkI4fEYaJfB7dONoQjaM")
            client.get_project("carrier_transport")

            # Iterate over the properties in the overall_bounds_dict and dynamically build the query
            query_carrier_transport = {}
            for prop, bounds in overall_bounds_dict.items():
                # Skip 'elec_cond_300k_low_doping' and 'therm_cond_300k_low_doping'
                if prop in ["elec_cond_300k_low_doping", "therm_cond_300k_low_doping"]:
                    # Proceed if 'upper_bound' and 'lower_bound' exist for the property
                    if "upper_bound" in bounds and "lower_bound" in bounds:
                        query_carrier_transport[prop] = (bounds["lower_bound"],
                                                         bounds["upper_bound"])

            tables = client.query_contributions(
                {"project":"carrier_transport",
                 "data__sigma__p__value__gt": query_carrier_transport["elec_cond_300k_low_doping"][0]/ 1e15 / 1e-14,
                 "data__sigma__p__value__lt": query_carrier_transport["elec_cond_300k_low_doping"][1]/ 1e15 / 1e-14,
                 "data__kappa__p__value__gt": query_carrier_transport["therm_cond_300k_low_doping"][0]/ 1e9 / 1e-14,
                 "data__kappa__p__value__lt": query_carrier_transport["therm_cond_300k_low_doping"][1]/ 1e9 / 1e-14,
                 "identifier__in": result_dict["material_id"],
                 },
                fields=["identifier", "formula", "data.sigma.p", "data.kappa.p"],
                sort="+formula"
            )

            # Only append the values to the corresponding material_id from
            # the result_dict. At the end, make all the remaining values
            # corresponding to the material_id as None
            # Iterate over the tables returned and map the data to the result_dict
            for table in tables["data"]:
                material_id = table["identifier"]  # Material ID from the table

                # Only map for materials already in result_dict
                if material_id in result_dict["material_id"]:
                    index = result_dict["material_id"].index(material_id)

                    # Access the electrical conductivity and thermal conductivity values
                    sigma_value = table["data"]["sigma"]["p"]["value"]  # Electrical conductivity
                    kappa_value = table["data"]["kappa"]["p"]["value"]  # Thermal conductivity

                    # Convert and append the values to the correct positions in the result_dict
                    result_dict["elec_cond_300k_low_doping"][index] = sigma_value * 1e15 * 1e-14
                    result_dict["therm_cond_300k_low_doping"][index] = kappa_value * 1e9 * 1e-14

            # Drop rows with None values
            keys_to_check = result_dict.keys()
            indices_to_drop = [i for i in range(formula_pretty_length)
                               if any(result_dict[key][i] is None for key in keys_to_check)]

            for i in sorted(indices_to_drop, reverse=True):
                for key in result_dict:
                    result_dict[key].pop(i)

            # Change the key name of bulk_modulus_vrh to bulk_modulus
            # & shear_modulus_vrh to shear_modulus
            if "bulk_modulus_vrh" in result_dict:
                result_dict["bulk_modulus"] = result_dict.pop("bulk_modulus_vrh")
            if "shear_modulus_vrh" in result_dict:
                result_dict["shear_modulus"] = result_dict.pop("shear_modulus_vrh")

        # Save the consolidated results to a JSON file
        now = datetime.now()
        my_file_name = f"{MODULE_DIR}/../io/outputs/consolidated_dict_" \
                       + now.strftime("%m_%d_%Y_%H_%M_%S")
        with open(my_file_name, "w") as my_file:
            json.dump(result_dict, my_file)

        return result_dict

    def get_material_match_costs(self,
                                 matches_dict: dict,
                                 consolidated_dict: dict = None) -> go.Figure:
        """
        Evaluates the 'real' candidate composites with
        the same cost function used for optimization.

        Args:
            matches_dict (dict)
            consolidated_dict (dict)

        Returns
        -------
            plotly.graph_objects.Figure
            - A table of the matches and their costs as evaluated by
              the genetic algorithm
        """
        if consolidated_dict is None:
            consolidated_dict = {}
        if matches_dict == {}:
            print("No materials match the recommended composite formulation.")
            return None

        if consolidated_dict == {}:
            with open("test_consolidated_dict") as f:
                consolidated_dict = json.load(f)

        all_vol_frac_combos = self.get_all_possible_vol_frac_combos()
        materials = list(matches_dict.values())
        material_combinations = list(itertools.product(*materials))

        # List to keep track of the lowest 5 costs and their corresponding data
        top_rows = []

        for combo in material_combinations:

            material_values = []
            mat_ids = np.zeros((len(material_combinations), self.optimization_params.num_materials))

            for category in self.optimization_params.property_categories:
                for prop in self.optimization_params.property_docs[category]:
                    for material_dict in combo:

                        # Extract the material ID string from the dictionary
                        material_id = next(iter(material_dict.keys()))

                        # Ensure material_id is a string
                        # (it should already be, but this is to be safe)
                        material_str = str(material_id)

                        if prop in consolidated_dict:
                            m = consolidated_dict["material_id"].index(material_str)
                            material_values.append(consolidated_dict[prop][m])

            # Create population of same properties for all members
            # based on material match combination
            population_values = np.tile(material_values, (len(all_vol_frac_combos),1))

            # Vary the volume fractions across the population
            num_vol_frac_combos = len(all_vol_frac_combos)
            num_materials = self.optimization_params.num_materials
            volume_fractions = np.array(all_vol_frac_combos).reshape(num_vol_frac_combos,
                                                                     num_materials)

            # Include the random mixing parameters and volume fractions in the population
            values = np.c_[population_values, volume_fractions]

            # Instantiate the population and find the best performers
            population = Population(optimization_params=self.optimization_params,
                                    ga_params=self.ga_params,
                                    num_properties=self.optimization_params.num_properties,
                                    values=values)
            population.set_costs()
            [sorted_costs, sorted_indices] = population.sort_costs()
            population.set_order_by_costs(sorted_indices)
            sorted_costs = np.reshape(sorted_costs, (len(sorted_costs), 1))

            # Assemble a table for printing
            mat_ids = []
            for material_dict in combo:
                material_id = next(iter(material_dict.keys()))
                material_str = str(material_id)
                mat_ids.append(np.reshape([material_id]*self.ga_params.num_members,
                                          (self.ga_params.num_members,1)))
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

        headers = get_headers(self.optimization_params.num_materials,
                              self.optimization_params.property_categories,
                              include_mpids=True)

        header_color = "lavender"
        odd_row_color = "white"
        even_row_color = "lightgrey"
        cells_color = [[odd_row_color,
                        even_row_color,
                        odd_row_color,
                        even_row_color,
                        odd_row_color]] # Hardcoded to 5 rows

        # Create the final table figure
        fig = go.Figure(data=[go.Table(
            columnwidth=1000,
            header=dict(
                values=headers,
                fill_color=header_color,
                align="left",
                font=dict(size=12),
                height=30
            ),
            cells=dict( # Transpose top_rows to get columns
                values=[list(col) for col in zip(*top_rows, strict=False)],
                fill_color=cells_color,
                align="left",
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

        return fig
