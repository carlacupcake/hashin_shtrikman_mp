import json
from typing import Any
from importlib import resources as impresources

from monty.serialization import loadfn

from ...log import logger
import hashin_shtrikman_mp.io.inputs.data as io_data

DEFAULT_PROPERTY_CATEGORIES = str(impresources.files(io_data) / "mp_property_docs.yaml")

def load_property_categories(filename=DEFAULT_PROPERTY_CATEGORIES, user_input: dict[Any, Any] | None = None):
    if user_input is None:
        user_input = {}
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