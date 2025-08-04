"""utilities.py."""
from importlib import resources as impresources
from typing import Any

import yaml
from monty.serialization import loadfn

import hashin_shtrikman_mp.io.inputs.data as io_data

from ..log import logger


HEADERS_PATH = str(impresources.files(io_data) / "display_table_headers.yaml")
PROPERTY_CATEGORIES = str(impresources.files(io_data) / "mp_property_docs.yaml")
COST_FORMULAS_PATH = str(impresources.files(io_data) / "cost_calculation_formulas.yaml")


def get_headers(num_materials: int,
                property_categories: list[str],
                include_mpids: bool = False) -> list:
    """
    Generates headers for a data table based on the number of materials
    and selected property categories.

    Args:
        num_materials (int)
        - The number of materials to include in the headers.
        property_categories (list[str])
        - List of property categories to include.
        include_mpids (bool, optional)
        - Whether to include MP-ID headers. Defaults to False.

    Returns
    -------
        headers (list)
    """
    with open(HEADERS_PATH) as stream:
        try:
            data = yaml.safe_load(stream)
            headers = []

            # Add headers for MP-IDs
            if include_mpids:
                headers += [f"Material {m} MP-ID" for m in range(1, num_materials + 1)]

            # Add headers for material properties
            for category, properties in data["Per Material"].items():
                if category in property_categories:
                    for prop in sorted(properties.values()):
                        for m in range(1, num_materials + 1):
                            headers.append(f"Phase {m} " + prop)

            # Add headers for volume fractions
            for m in range(1, num_materials + 1):
                headers.append(f"Phase {m} Volume Fraction")

            # Add headers for effective properties
            for category, properties in data["Per Material"].items():
                if category in property_categories:
                    for prop in sorted(properties.values()):
                        headers.append(f"Effective" + prop)

            # Add headers for "Common" properties if present
            if "Common" in data:
                for common_key in data["Common"]:
                    headers.append(common_key)

        except yaml.YAMLError as exc:
            print(exc)

    return headers


def load_property_docs() -> dict:
    """
    Loads property documentation from a predefined YAML file.

    Returns
    -------
        dict
        - A dictionary containing property documentation.
    """
    return loadfn(PROPERTY_CATEGORIES)


def load_property_categories(user_input: dict[Any, Any] | None = None) -> tuple[list[str], dict]:
    """
    Identifies property categories present in the user input by comparing them
    with predefined property documentation.

    Args:
        user_input (dict[Any, Any] | None, optional)
        - A dictionary of user-defined properties. Defaults to None.

    Returns
    -------
        tuple: A tuple containing:
            - property_categories (list): A list of property categories found in the user input.
            - property_docs (dict): A dictionary of all property documentation.
    """
    if user_input is None:
        user_input = {}

    property_categories = []
    property_docs = load_property_docs()

    # Flatten the user input to get a list of all properties defined by the user
    user_defined_properties = []

    for entity in user_input.values():
        for prop in entity:
            user_defined_properties.append(prop)

    # Only keep the unique entries of the list
    user_defined_properties = list(set(user_defined_properties))

    # Iterate through property categories to check which are present in the user input
    for category, properties in property_docs.items():
        if any(prop in user_defined_properties for prop in properties):
            property_categories.append(category)

    return property_categories, property_docs


def compile_formulas(formulas_dict: dict[Any, Any] | None = None) -> dict:
    """
    Compiles mathematical formulas defined as strings in a dictionary into executable Python code.

    Args:
        formulas_dict (dict)
        - A dictionary where the keys are formula names and the values are
          formulas as strings or nested dictionaries.

    Returns
    -------
        compiled_formulas (dict)
        - A dictionary with the same structure as `formulas_dict`, but with formulas compiled.
    """
    if formulas_dict is None:
        formulas_dict = {}

    compiled_formulas = {}
    for key, formula in formulas_dict.items():
        if isinstance(formula, str):
            # List all variables used in cost_calculation_formulas.yaml as Python variables
            compiled_formula = formula.format(
                a_1="a_1",
                a_n="a_n",
                a_1_term_i="a_1_term_i",
                a_n_term_i="a_n_term_i",
                alpha_1="alpha_1",
                alpha_n="alpha_n",
                bulk_alpha_1="bulk_alpha_1",
                bulk_alpha_n="bulk_alpha_n",
                cf_load_i="cf_load_i",
                cf_response_i="cf_response_i",
                cf_elastic_i="cf_eleastic_i",
                eff_max="eff_max",
                eff_min="eff_min",
                eff_prop="eff_prop",
                eff_elastic="eff_elastic",
                mixing_param="mixing_param",
                n="n",
                phase_1="phase_1",
                phase_1_vol_frac="phase_1_vol_frac",
                phase_1_bulk="phase_1_bulk",
                phase_1_shear="phase_1_shear",
                phase_i="phase_i",
                phase_i_elastic="phase_i_elastic",
                phase_i_vol_frac="phase_i_vol_frac",
                phase_n="phase_n",
                phase_n_bulk="phase_n_bulk",
                phase_n_shear="phase_n_shear",
                phase_n_vol_frac="phase_n_vol_frac",
                shear_alpha_1="shear_alpha_1",
                shear_alpha_n="shear_alpha_n",
                vf_weighted_sum_cfs="vf_weighted_sum_cfs"
            )
            # Compile the formula
            compiled_formulas[key] = compile(compiled_formula, "<string>", "eval")
        elif isinstance(formula, dict):
            # Recursively compile nested dictionaries
            compiled_formulas[key] = compile_formulas(formula)

    return compiled_formulas

COMPILED_CALC_GUIDE = compile_formulas(loadfn(COST_FORMULAS_PATH))
