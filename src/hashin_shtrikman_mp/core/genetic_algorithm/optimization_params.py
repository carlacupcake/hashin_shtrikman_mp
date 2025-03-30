"""optimization_params.py."""
import numpy as np

from typing import Any
from pydantic import BaseModel, Field

from ...log.custom_logger import logger
from ..user_input import UserInput
from ..utilities import load_property_categories

np.seterr(divide="raise")


class OptimizationParams(BaseModel):
    """
    Hashin-Shtrikman optimization class.

    Class to integrate Hashin-Shtrikman (HS) bounds with a genetic algorithm
    and find optimal material properties for each composite constituent to achieve
    desired properties.
    """

    property_categories: list[str] = Field(
        default_factory=list,
        description="List of property categories considered for optimization."
    )
    property_docs: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="A hard coded yaml file containing property categories and their individual properties."
    )
    lower_bounds: dict[str, Any] = Field(
        default_factory=dict,
        description="Lower bounds for properties of materials considered in the optimization."
    )
    upper_bounds: dict[str, Any] = Field(
        default_factory=dict,
        description="Upper bounds for properties of materials considered in the optimization."
    )
    desired_props: dict[str, list[float]] = Field(
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
    indices_elastic_moduli: list[Any] = Field(
        default=[None, None],
        description="For handling coupling between bulk & shear moduli"
                    "List of length 2, first element is index of bulk modulus"
                    "in list version of the bounds, second element is the"
                    "shear modulus"
    )

    class Config:
        arbitrary_types_allowed = True


    @classmethod
    def from_user_input(cls, user_input: UserInput | None = None) -> "OptimizationParams":
        """
        Initializes the `OptimizationParams` class from the user input dictionary.
        """

        params = {}
        # Load property categories and docs
        property_categories, property_docs = load_property_categories(user_input=user_input)
        params["property_categories"] = property_categories
        params["property_docs"] = property_docs

        # Since user_input is required to set desired props and bounds, ensure it's processed last
        # Is user_input ever None??
        if user_input is not None:
            num_materials  = len(user_input) - 1

            desired_props = cls.get_desired_props_from_user_input(
                user_input,
                property_categories=property_categories,
                property_docs=property_docs
            )

            lower_bounds = cls.get_bounds_from_user_input(
                user_input,
                "lower_bound",
                property_docs=property_docs,
                num_materials=num_materials
            )

            upper_bounds = cls.get_bounds_from_user_input(
                user_input,
                "upper_bound",
                property_docs=property_docs,
                num_materials=num_materials
            )

            num_properties = cls.get_num_properties_from_desired_props(desired_props=desired_props)
            indices_elastic_moduli = cls.get_elastic_idx_from_user_input(
                upper_bounds=upper_bounds,
                property_categories=property_categories
            )

            # Update values accordingly
            params.update({
                "desired_props":       desired_props,
                "lower_bounds":        lower_bounds,
                "upper_bounds":        upper_bounds,
                "num_properties":      num_properties,
                "num_materials":       num_materials,
                "indices_elastic_moduli": indices_elastic_moduli
            })

        return cls(**params)


    @staticmethod
    def get_num_properties_from_desired_props(desired_props: dict[str, list[float]]) -> int:
        """
        Calculates the total number of properties from the desired properties dictionary.

        Args:
            desired_props (dict)

        Returns:
            num_properties (int)
        """

        num_properties = 0

        # Iterate through property categories to count the total number of properties
        for properties in desired_props.values():
            num_properties += len(properties)  # Add the number of properties in each category

        # Account for volume fractions
        num_properties += 1

        return num_properties


    @staticmethod
    def get_bounds_from_user_input(user_input:    dict,
                                   bound_key:     str,
                                   property_docs: dict[str, list[str]],
                                   num_materials: int) -> dict:
        """
        Extracts bounds (upper or lower) for material properties from the user input.

        Args:
            user_input (dict)
            bound_key (str)
            property_docs (dict[str, list[str]])
            num_materials (int)

        Returns:
            bounds (dict)
        """

        if bound_key not in ["upper_bound", "lower_bound"]:
            raise ValueError("bound_key must be either 'upper_bound' or 'lower_bound'.")

        # Get bounds for material properties from user_input
        bounds: dict[str, dict[str, list[float]]] = {}
        for material, properties in user_input.items():
            if material == "mixture":  # Skip 'mixture' as it's not a material
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
        if bound_key == "upper_bound":
            bounds["volume-fractions"] = [0.99] * num_materials
        else:
            bounds["volume-fractions"] = [0.01] * num_materials

        return bounds


    @staticmethod
    def get_desired_props_from_user_input(user_input:          dict,
                                          property_categories: list[str],
                                          property_docs:       dict) -> dict:
        """
        Extracts the desired values for each property category.

        Args:
            user_input (dict)
            property_categories (list[str])
            property_docs (dict)

        Returns:
            desired_props (dict)
        """

        # Initialize the dictionary to hold the desired properties
        desired_props: dict[str, list[float]] = {category: [] for category in property_categories}

        # Extracting the desired properties from the 'mixture' part of final_dict
        mixture_props = user_input.get("mixture", {})

        # Iterate through each property category and its associated properties
        for category, properties in property_docs.items():
            for prop in properties:
                # Check if the property is in the mixture; if so, append its desired value
                if prop in mixture_props:
                    desired_props[category].append(mixture_props[prop]["desired_prop"])

        return desired_props


    @staticmethod
    def get_elastic_idx_from_user_input(upper_bounds:        dict,
                                        property_categories: list[str]) -> list[int]:
        """
        Gets the indices of elastic properties from the upper bounds.

        Args:
            upper_bounds (dict)
            property_categories (list[str])

        Returns:
            indices (list[int]): ensures proper ordering of elastic moduli
        """

        idx = 0
        indices = [None, None]
        for material in upper_bounds:
            if material != "volume-fractions":
                for category, properties in upper_bounds[material].items():
                    if category in property_categories:
                        for _ in properties:
                            if category == "elastic":
                                return [idx, idx+1]
                            idx += 1

        return indices
