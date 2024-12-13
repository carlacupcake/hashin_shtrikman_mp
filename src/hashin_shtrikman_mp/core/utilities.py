from importlib import resources as impresources
import yaml
import json
from typing import Any, List

from monty.serialization import loadfn

from ..log import logger

import hashin_shtrikman_mp.io.inputs.data as io_data

HEADERS_PATH = str(impresources.files(io_data) / "display_table_headers.yaml")
PROPERTY_CATEGORIES = str(impresources.files(io_data) / "mp_property_docs.yaml")
COST_FORMULAS_PATH = str(impresources.files(io_data) / "cost_calculation_formulas.yaml")


def get_headers(num_materials: int,
                property_categories: List[str],
                include_mpids: bool = False) -> list:

    with open(HEADERS_PATH) as stream:
        try:
            data = yaml.safe_load(stream)
            headers = []

            # Add headers for mp-ids
            if include_mpids:
                headers += [f"Material {m} MP-ID" for m in range(1, num_materials + 1)]

            # Add headers for material properties
            for category, properties in data["Per Material"].items():
                if category in property_categories:
                    for prop in properties.values():
                        for m in range(1, num_materials + 1):
                            headers.append(f"Phase {m} " + prop)

            # Add headers for volume fractions
            for m in range(1, num_materials + 1):
                headers.append(f"Phase {m} Volume Fraction")

            # Add headers for "Common" properties if present
            if "Common" in data:
                for common_key in data["Common"]:
                    headers.append(common_key)

        except yaml.YAMLError as exc:
            print(exc)

    return headers

def load_property_docs():
    return loadfn(PROPERTY_CATEGORIES)

def load_property_categories(user_input: dict[Any, Any] | None = None):
    if user_input is None:
        user_input = {}
    logger.info(f"Loading property categories from {PROPERTY_CATEGORIES}.")

    property_categories = []
    property_docs = load_property_docs()

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

    logger.info(f"property_categories = {property_categories}")
    return property_categories, property_docs

def compile_formulas(formulas_dict):
    compiled_formulas = {}
    for key, formula in formulas_dict.items():
        if isinstance(formula, str):
            # List all variables used in cost_calculation_formulas.yaml as Python variables
            compiled_formula = formula.format(
                A1="A1",
                An="An",
                A1_term_i="A1_term_i",
                An_term_i="An_term_i",
                alpha_1="alpha_1",
                alpha_n="alpha_n",
                bulk_alpha_1="bulk_alpha_1",
                bulk_alpha_n="bulk_alpha_n",
                shear_alpha_1="shear_alpha_1",
                shear_alpha_n="shear_alpha_n",
                phase_1="phase_1",
                phase_i="phase_i",
                phase_n="phase_n",
                phase_1_vol_frac="phase_1_vol_frac",
                phase_i_vol_frac="phase_i_vol_frac",
                phase_n_vol_frac="phase_n_vol_frac",
                phase_1_bulk="phase_1_bulk",
                phase_1_shear="phase_1_shear",
                phase_i_elastic="phase_i_elastic",
                phase_n_bulk="phase_n_bulk",
                phase_n_shear="phase_n_shear",
                eff_max="eff_max",
                eff_min="eff_min",
                eff_prop="eff_prop",
                eff_elastic="eff_elastic",
                mixing_param="mixing_param",
                cf_load_i="cf_load_i",
                cf_response_i="cf_response_i",
                cf_elastic_i="cf_eleastic_i",
                vf_weighted_sum_cfs="vf_weighted_sum_cfs"
            )
            # Compile the formula
            compiled_formulas[key] = compile(compiled_formula, "<string>", "eval")
        elif isinstance(formula, dict):
            # Recursively compile nested dictionaries
            compiled_formulas[key] = compile_formulas(formula)

    return compiled_formulas

COMPILED_CALC_GUIDE = compile_formulas(loadfn(COST_FORMULAS_PATH))