"""member.py."""
import warnings
from typing import Any, Union

import numpy as np
from pydantic import BaseModel, Field, PositiveInt, model_validator

# Custom imports
from .genetic_algo import GAParams

class Member(BaseModel):
    """
    Class to represent a member of the population in genetic algorithm optimization.
    Stores the properties and configuration for genetic algorithm operations.
    """

    num_materials: PositiveInt = Field(
        default=0,
        description="Number of materials in the ultimate composite."
    )
    num_properties: PositiveInt = Field(
        default=0,
        description="Number of properties that each member of the population has."
    )
    values: Union[np.ndarray, None] = Field(
        default=None,
        description="Values array representing the member's properties."
    )
    property_categories: list[str] = Field(
        default=[],
        description="List of property categories considered for optimization."
    )
    property_docs: dict[str, dict[str, Any]] = Field(
        default={},
        description="A hard coded yaml file containing property categories and their individual properties."
    )
    desired_props: dict[str, Any] = Field(
        default={},
        description="Dictionary mapping individual properties to their desired properties."
    )
    ga_params: Union['GAParams', None] = Field(
        default=None,
        description="Parameter initialization class for the genetic algorithm."
    )
    calc_guide: Union[dict[str, Any], Any] = Field(
        default_factory=lambda: None,
        description="Calculation guide for property evaluation with compiled expressions."
    )

    # To use np.ndarray or other arbitrary types in your Pydantic models
    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode="before")
    def check_and_initialize_arrays(cls, values):
        # Initialize 'values' with zeros if not provided or if it is np.empty
        if values.get("values") is None or (isinstance(values.get("values"), np.ndarray) and values.get("values").size == 0):
            num_properties = values.get("num_properties", 0)
            # Assuming you want a 2D array shape based on your original code
            values["values"] = np.zeros(shape=(num_properties, 1))
        return values

    #------ Getter Methods ------#
    def get_cost(self, include_cost_breakdown=False):
        """MAIN COST FUNCTION."""
        # Extract attributes from self
        tolerance          = self.ga_params.tolerance/self.num_materials
        weight_eff_prop    = 1/(self.num_materials) # one effective property per property
        weight_conc_factor = 1/(2 * (self.num_properties - 1) * self.num_materials) # two concentration factors per property per material
        weight_domains     = 1/(2 * len(self.property_categories))

        # Initialize effective property, concentration factor, and weight arrays
        # Initialize to zero so as not to contribute to cost if unchanged
        effective_properties  = []
        concentration_factors = []
        cost_func_weights     = []

        # Get Hashin-Shtrikman effective properties for all properties
        idx = 0

        for category in self.property_categories:

            if category == "elastic":
                moduli_eff_props, moduli_cfs = self.get_elastic_eff_props_and_cfs(idx=idx)
                effective_properties.extend(moduli_eff_props)
                concentration_factors.extend(moduli_cfs)

                eff_univ_aniso, cfs_univ_aniso = self.get_general_eff_prop_and_cfs(idx=idx+2)
                effective_properties.extend(eff_univ_aniso)
                concentration_factors.extend(cfs_univ_aniso)

            else:
                for p in range(idx, idx + len(self.property_docs[category])): # loop through all properties in the category
                    new_eff_props, new_cfs = self.get_general_eff_prop_and_cfs(idx=p)
                    effective_properties.extend(new_eff_props)
                    concentration_factors.extend(new_cfs)

            idx += len(self.property_docs[category])

        # Determine weights based on concentration factor magnitudes
        for factor in concentration_factors:
            if (factor - tolerance) / tolerance > 0:
                cost_func_weights.append(weight_conc_factor)
            else:
                cost_func_weights.append(0)

        # Cast concentration factors, effective properties and weights to numpy arrays
        concentration_factors = np.array(concentration_factors)
        effective_properties  = np.array(effective_properties)
        cost_func_weights     = np.array(cost_func_weights)

        # Extract desired properties from dictionary
        des_props = []
        for category, properties in self.desired_props.items():
            des_props.extend(properties)
        des_props = np.array(des_props)

        # Assemble the cost function
        costs_eff_props = abs(np.divide(des_props - effective_properties, effective_properties))
        costs_cfs = np.multiply(cost_func_weights, abs(np.divide(concentration_factors - tolerance, tolerance)))

        cost = weight_domains * (weight_eff_prop * np.sum(costs_eff_props) + weight_conc_factor * np.sum(costs_cfs))

        if include_cost_breakdown:
            return cost, costs_eff_props, costs_cfs

        else:
            return cost

    def get_general_eff_prop_and_cfs(self, idx = 0): # idx is the index in self.values where category properties begin

        # Initialize effective property, concentration factor, and weight arrays
        # Initialize to zero so as not to contribute to cost if unchanged
        effective_properties  = []
        concentration_factors = []

        # Prepare indices for looping over properties
        stop = -self.num_materials     # the last num_materials entries are volume fractions, not material properties
        step = self.num_properties - 1 # subtract 1 so as not to include volume fraction

        # Get Hashin-Shtrikman effective properties for all properties
        volume_fractions = self.values[-self.num_materials:]
        properties = self.values[idx:stop:step]

        sorted_properties = np.sort(properties, axis=0)
        sorted_indices = np.argsort(properties, axis=0)
        sorted_vol_fracs = volume_fractions[sorted_indices]

        phase_1 = sorted_properties[0]
        phase_n = sorted_properties[-1]
        if phase_1 == 0:
            alpha_1 = 0
        else:
            alpha_1 = eval(self.calc_guide["effective_props"]["alpha_1"], {}, {"phase_1": phase_1})
        if phase_n == 0:
            alpha_n = 0
        else:
            alpha_n = eval(self.calc_guide["effective_props"]["alpha_n"], {}, {"phase_n": phase_n})

        A1 = 0
        for i in range(1, self.num_materials):
            phase_i_vol_frac = sorted_vol_fracs[i]
            phase_i = sorted_properties[i]
            if phase_1 == phase_i:
                A1_term_i = 0
            else:
                A1_term_i = eval(self.calc_guide["effective_props"]["A1_term_i"], {}, {"phase_1": phase_1, "phase_i": phase_i, "phase_i_vol_frac": phase_i_vol_frac, "alpha_1": alpha_1})
            A1 += A1_term_i

        An = 0
        for i in range(self.num_materials - 1):
            phase_i_vol_frac = sorted_vol_fracs[i]
            phase_i = sorted_properties[i]
            if phase_i == phase_n:
                An_term_i = 0
            else:
                An_term_i = eval(self.calc_guide["effective_props"]["An_term_i"], {}, {"phase_n": phase_n, "phase_i": phase_i, "phase_i_vol_frac": phase_i_vol_frac, "alpha_n": alpha_n})
            An += An_term_i

        # Compute effective property bounds with Hashin-Shtrikman
        if phase_1 == phase_n:
            effective_prop_min = phase_1
            effective_prop_max = phase_n
        else:
            effective_prop_min = eval(self.calc_guide["effective_props"]["eff_min"], {}, {"phase_1": phase_1, "phase_n": phase_n, "alpha_1": alpha_1, "A1": A1})
            effective_prop_max = eval(self.calc_guide["effective_props"]["eff_max"], {}, {"phase_1": phase_1, "phase_n": phase_n, "alpha_n": alpha_n, "An": An})

        mixing_param = self.ga_params.mixing_param
        effective_prop = mixing_param * effective_prop_max + (1 - mixing_param) * effective_prop_min
        effective_properties.append(effective_prop)

        # Compute concentration factors for load sharing
        vf_weighted_sum_cf_load_i = 0
        vf_weighted_sum_cf_response_i = 0
        for i in range(1, self.num_materials):
            phase_i_vol_frac = sorted_vol_fracs[i]
            phase_i = sorted_properties[i]
            if phase_i_vol_frac == 0:
                cf_load_i = 0
                cf_response_i = 0
            else:
                # if cf_load_i has div by zero error, then set cf_load_i to 0
                try:
                    cf_load_i = eval(self.calc_guide["concentration_factors"]["cf_load_i"], {}, {"phase_1": phase_1, "phase_i": phase_i, "phase_i_vol_frac": phase_i_vol_frac, "eff_prop": effective_prop})
                    cf_response_i = eval(self.calc_guide["concentration_factors"]["cf_response_i"], {}, {"phase_i": phase_i, "cf_load_i": cf_load_i, "eff_prop": effective_prop})
                except FloatingPointError:
                    cf_load_i = 0
                    cf_response_i = 0

            concentration_factors.append(cf_load_i)
            concentration_factors.append(cf_response_i)
            vf_weighted_sum_cf_load_i += phase_i_vol_frac * cf_load_i
            vf_weighted_sum_cf_response_i += phase_i_vol_frac * cf_response_i

        phase_1_vol_frac = sorted_vol_fracs[0]
        if phase_1_vol_frac == 0:
            cf_load_1 = 0
            cf_response_1 = 0
        else:
            cf_load_1     = eval(self.calc_guide["concentration_factors"]["cf_1"], {}, {"phase_1_vol_frac": phase_1_vol_frac, "vf_weighted_sum_cfs": vf_weighted_sum_cf_load_i})
            cf_response_1 = eval(self.calc_guide["concentration_factors"]["cf_1"], {}, {"phase_1_vol_frac": phase_1_vol_frac, "vf_weighted_sum_cfs": vf_weighted_sum_cf_response_i})
        concentration_factors.insert(0, cf_response_1)
        concentration_factors.insert(0, cf_load_1)

        return effective_properties, concentration_factors

    def get_elastic_eff_props_and_cfs(self, idx = 0): # idx is the index in self.values where elastic properties begin

        # Initialize effective property and concentration factor arrays
        # Initialize to zero so as not to contribute to cost if unchanged
        effective_properties  = []
        concentration_factors = []

        # Prepare indices for looping over properties
        stop = -self.num_materials     # the last num_materials entries are volume fractions, not material properties
        step = self.num_properties - 1 # subtract 1 so as not to include volume fraction

        # Extract bulk moduli and shear moduli from member
        volume_fractions = self.values[-self.num_materials:]

        bulk_mods = self.values[idx:stop:step]
        sorted_bulk = np.sort(bulk_mods, axis=0)
        sorted_bulk_indices = np.argsort(bulk_mods, axis=0)
        sorted_vol_fracs = volume_fractions[sorted_bulk_indices]

        shear_mods = self.values[idx+1:stop:step]
        sorted_shear = np.sort(shear_mods, axis=0)
        sorted_shear_indices = np.argsort(shear_mods, axis=0)
        if not np.array_equal(sorted_bulk_indices, sorted_shear_indices):
            warnings.warn("Warning: Cannot properly apply Hashin-Shtrikman bounds on effective properties when bulk_i < bulk_j and shear_i > shear_j, or vice versa.")

        phase_1_bulk  = sorted_bulk[0]
        phase_n_bulk  = sorted_bulk[-1]
        phase_1_shear = sorted_shear[0]
        phase_n_shear = sorted_shear[-1]

        if (phase_1_bulk == 0) and (phase_1_shear == 0):
            bulk_alpha_1 = 0
        else:
            bulk_alpha_1  = eval(self.calc_guide["effective_props"]["bulk_alpha_1"],  {}, {"phase_1_bulk": phase_1_bulk, "phase_1_shear": phase_1_shear})
        if (phase_n_bulk == 0) and (phase_n_shear == 0):
            bulk_alpha_1 = 0
        else:
            bulk_alpha_n  = eval(self.calc_guide["effective_props"]["bulk_alpha_n"],  {}, {"phase_n_bulk": phase_n_bulk, "phase_n_shear": phase_n_shear})
        if phase_1_shear == 0:
            shear_alpha_1 = 0
        else:
            shear_alpha_1 = eval(self.calc_guide["effective_props"]["shear_alpha_1"], {}, {"phase_1_bulk": phase_1_bulk, "phase_1_shear": phase_1_shear})
        if phase_n_shear == 0:
            shear_alpha_n = 0
        else:
            shear_alpha_n = eval(self.calc_guide["effective_props"]["shear_alpha_n"], {}, {"phase_n_bulk": phase_n_bulk, "phase_n_shear": phase_n_shear})

        A1_bulk = 0
        A1_shear = 0
        for i in range(1, self.num_materials):
            phase_i_vol_frac = sorted_vol_fracs[i]
            phase_i_bulk = sorted_bulk[i]
            if phase_1_bulk == phase_i_bulk:
                A1_term_i = 0
            else:
                A1_term_i = eval(self.calc_guide["effective_props"]["A1_term_i"], {}, {"phase_1": phase_1_bulk, "phase_i": phase_i_bulk, "phase_i_vol_frac": phase_i_vol_frac, "alpha_1": bulk_alpha_1})
            A1_bulk += A1_term_i

            phase_i_shear = sorted_shear[i]
            if phase_1_shear == phase_i_shear:
                A1_term_i = 0
            else:
                A1_term_i = eval(self.calc_guide["effective_props"]["A1_term_i"], {}, {"phase_1": phase_1_shear, "phase_i": phase_i_shear, "phase_i_vol_frac": phase_i_vol_frac, "alpha_1": shear_alpha_1})
            A1_shear += A1_term_i

        An_bulk = 0
        An_shear = 0
        for i in range(self.num_materials - 1):
            phase_i_vol_frac = sorted_vol_fracs[i]
            phase_i_bulk = sorted_bulk[i]
            if phase_i_bulk == phase_n_bulk:
                An_term_i = 0
            else:
                An_term_i = eval(self.calc_guide["effective_props"]["An_term_i"], {}, {"phase_n": phase_n_bulk, "phase_i": phase_i_bulk, "phase_i_vol_frac": phase_i_vol_frac, "alpha_n": bulk_alpha_n})
            An_bulk += An_term_i

            phase_i_shear = sorted_shear[i]
            if phase_i_shear == phase_n_shear:
                An_term_i = 0
            else:
                An_term_i = eval(self.calc_guide["effective_props"]["An_term_i"], {}, {"phase_n": phase_n_shear, "phase_i": phase_i_shear, "phase_i_vol_frac": phase_i_vol_frac, "alpha_n": shear_alpha_n})
            An_shear += An_term_i

        # Compute effective bulk modulus bounds with Hashin-Shtrikman
        if phase_1_bulk == phase_n_bulk:
            effective_bulk_min = phase_1_bulk
            effective_bulk_max = phase_n_bulk
        else:
            effective_bulk_min = eval(self.calc_guide["effective_props"]["eff_min"], {}, {"phase_1": phase_1_bulk, "phase_n": phase_n_bulk, "alpha_1": bulk_alpha_1, "A1": A1_bulk})
            effective_bulk_max = eval(self.calc_guide["effective_props"]["eff_max"], {}, {"phase_1": phase_1_bulk, "phase_n": phase_n_bulk, "alpha_n": bulk_alpha_n, "An": An_bulk})

        # Compute effective shear modulus bounds with Hashin-Shtrikman
        if phase_1_bulk == phase_n_bulk:
            effective_shear_min = phase_1_shear
            effective_shear_max = phase_n_shear
        else:
            effective_shear_min = eval(self.calc_guide["effective_props"]["eff_min"], {}, {"phase_1": phase_1_shear, "phase_n": phase_n_shear, "alpha_1": shear_alpha_1, "A1": A1_shear})
            effective_shear_max = eval(self.calc_guide["effective_props"]["eff_max"], {}, {"phase_1": phase_1_shear, "phase_n": phase_n_shear, "alpha_n": shear_alpha_n, "An": An_shear})

        # Compute concentration factors for load sharing
        mixing_param = self.ga_params.mixing_param
        effective_bulk  = mixing_param * effective_bulk_max +  (1 - mixing_param) * effective_bulk_min
        effective_properties.append(effective_bulk)
        effective_shear = mixing_param * effective_shear_max + (1 - mixing_param) * effective_shear_min
        effective_properties.append(effective_shear)

        vf_weighted_sum_cf_bulk_i = 0
        vf_weighted_sum_cf_shear_i = 0
        for i in range(1, self.num_materials):
            phase_i_vol_frac = sorted_vol_fracs[i]
            phase_i_bulk = sorted_bulk[i]
            phase_i_shear = sorted_shear[i]
            if phase_i_vol_frac == 0:
                cf_bulk_i = 0
                cf_shear_i = 0
            else:
                if phase_1_bulk == phase_i_bulk:
                    cf_bulk_i = 0
                else:
                    cf_bulk_i = eval(self.calc_guide["concentration_factors"]["cf_elastic_i"],  {}, {"phase_i_vol_frac": phase_i_vol_frac, "phase_i_elastic": phase_1_bulk,  "eff_elastic": effective_bulk,  "phase_1": phase_1_bulk,  "phase_i": phase_i_bulk})
                if phase_1_shear == phase_i_shear:
                    cf_shear_i = 0
                else:
                    cf_shear_i = eval(self.calc_guide["concentration_factors"]["cf_elastic_i"], {}, {"phase_i_vol_frac": phase_i_vol_frac, "phase_i_elastic": phase_1_shear, "eff_elastic": effective_shear, "phase_1": phase_1_shear, "phase_i": phase_i_shear})

            concentration_factors.append(cf_bulk_i)
            concentration_factors.append(cf_shear_i)
            vf_weighted_sum_cf_bulk_i  += phase_i_vol_frac * cf_bulk_i
            vf_weighted_sum_cf_shear_i += phase_i_vol_frac * cf_shear_i

        phase_1_vol_frac = sorted_vol_fracs[0]
        if phase_1_vol_frac == 0:
            cf_bulk_1 = 0
            cf_shear_1 = 0
        else:
            cf_bulk_1  = eval(self.calc_guide["concentration_factors"]["cf_1"], {}, {"phase_1_vol_frac": phase_1_vol_frac, "vf_weighted_sum_cfs": vf_weighted_sum_cf_bulk_i})
            cf_shear_1 = eval(self.calc_guide["concentration_factors"]["cf_1"], {}, {"phase_1_vol_frac": phase_1_vol_frac, "vf_weighted_sum_cfs": vf_weighted_sum_cf_shear_i})
        concentration_factors.insert(0, cf_shear_1)
        concentration_factors.insert(0, cf_bulk_1)

        return effective_properties, concentration_factors

    def get_effective_properties(self):

        # Initialize effective property array
        effective_properties  = []

        # Get Hashin-Shtrikman effective properties for all properties
        idx = 0

        for category in self.property_categories:

            if category == "elastic":
                moduli_eff_props, _ = self.get_elastic_eff_props_and_cfs(idx=idx)
                effective_properties.extend(moduli_eff_props)

                eff_univ_aniso, _ = self.get_general_eff_prop_and_cfs(idx=idx+2)
                effective_properties.extend(eff_univ_aniso)

            else:
                for p in range(idx, idx + len(self.property_docs[category])): # loop through all properties in the category
                    new_eff_props, _ = self.get_general_eff_prop_and_cfs(idx=p)
                    effective_properties.extend(new_eff_props)

            idx += len(self.property_docs[category])

        # Cast effective properties to numpy arrays
        return np.array(effective_properties)
