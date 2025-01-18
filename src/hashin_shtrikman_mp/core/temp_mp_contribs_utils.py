"""optimizer.py."""
import re
from typing import Any

from pydantic import BaseModel, Field

from hashin_shtrikman_mp.log.custom_logger import logger

# Optimizer class defaults
DEFAULT_FIELDS: dict = {"material_id": [],
                        "is_stable": [],
                        "band_gap": [],
                        "is_metal": [],
                        "formula_pretty": []}

class Optimizer(BaseModel):
    """
    Hashin-Shtrikman optimization class.

    Class to integrate Hashin-Shtrikman (HS) bounds with a genetic algorithm
    and find optimal material properties for each composite constituent to achieve
    desired properties.
    """

    api_key: str | None = Field(
        default=None,
        description="API key for accessing Materials Project database."
        )
    mp_contribs_project: str | None = Field(
        default=None,
        description="MPContribs project name for querying project-specific data."
        )
    fields: dict[str, list[Any]] = Field(
        default_factory=lambda: DEFAULT_FIELDS.copy(),
        description="Fields to query from the Materials Project database."
        )

    # To use np.ndarray or other arbitrary types in your Pydantic models
    class Config:
        arbitrary_types_allowed = True

    def mp_contribs_prop(self, prop, my_dict):
        if prop == "therm_cond_300k_low_doping":
            table_column = 7
        elif prop == "elec_cond_300k_low_doping":
            table_column = 5
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
            prop_value = float(f"{prop_str}e{prop_exponent}") * 1e-14  # multiply by relaxation time, 10 fs
            logger.info(f"{prop}_if_statement = {prop_value}")
        else:
            prop_value = float(prop_str) * 1e-14  # multiply by relaxation time, 10 fs
            logger.info(f"{prop}_else_statement = {prop_value}")

        self.fields[prop].append(prop_value)

    def superscript_to_int(self, superscript_str):
        superscript_to_normal = {
            "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
            "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9"
        }
        normal_str = "".join(superscript_to_normal.get(char, char) for char in superscript_str)
        return int(normal_str)


