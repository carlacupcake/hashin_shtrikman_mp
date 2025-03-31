"""member.py."""
import numpy as np
import warnings

from ..utilities import COMPILED_CALC_GUIDE
from .genetic_algorithm_parameters import GeneticAlgorithmParams
from .optimization_params import OptimizationParams


class Member:
    """
    Class to represent a member of the population in genetic algorithm optimization.
    Stores the functions for cost function calculation.
    """

    def __init__(self,
                 ga_params:           GeneticAlgorithmParams,
                 optimization_params: OptimizationParams,
                 values:              np.ndarray = None,
    ):
        self.values = values
        if self.values is None or (isinstance(values, np.ndarray) and values.size == 0):
            self.values = np.zeros(shape=(optimization_params.num_properties, 1))

        self.ga_params = ga_params
        self.opt_params = optimization_params


    def get_cost(self, include_cost_breakdown: bool = False):
        """
        Calculates the total cost for the current member based on effective properties 
        and concentration factors, incorporating domain-specific weights and tolerances.

        The cost function evaluates the deviation of effective properties and 
        concentration factors from desired values, penalizing larger deviations 
        while accounting for the relative importance of different property domains.

        Args:
            include_cost_breakdown (bool, optional)

        Returns:
            cost (float)

        Notes:
            - There is one effective property per property
            - There are two concentration factors per property per material
            - Except for bulk and shear which collectively have two
              concentration factors instead of four
        """

        # Extract attributes from self
        tolerance          = self.ga_params.tolerance / self.opt_params.num_materials
        weight_eff_prop    = 1/(self.opt_params.num_properties - 1)
        weight_conc_factor = 1/(2 * (self.opt_params.num_properties - 1) * \
                                self.opt_params.num_materials)
        weight_domains     = 1/(2 * len(self.opt_params.property_categories))

        # Initialize effective property, concentration factor, and weight arrays
        # Initialize to zero so as not to contribute to cost if unchanged
        effective_properties  = []
        concentration_factors = []
        cost_func_weights     = []

        # Get Hashin-Shtrikman effective properties for all properties
        idx = 0
        for category in self.opt_params.property_categories:

            if category == "elastic":
                moduli_eff_props, moduli_cfs = self.get_elastic_eff_props_and_cfs(idx=idx)
                effective_properties.extend(moduli_eff_props)
                concentration_factors.extend(moduli_cfs)

                eff_univ_aniso, cfs_univ_aniso = self.get_general_eff_prop_and_cfs(idx=idx+2)
                effective_properties.extend(eff_univ_aniso)
                concentration_factors.extend(cfs_univ_aniso)

            else:
                # Loop through all properties in the category
                for p in range(idx, idx + len(self.opt_params.property_docs[category])):
                    new_eff_props, new_cfs = self.get_general_eff_prop_and_cfs(idx=p)
                    effective_properties.extend(new_eff_props)
                    concentration_factors.extend(new_cfs)

            idx += len(self.opt_params.property_docs[category])

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
        for category, properties in self.opt_params.desired_props.items():
            des_props.extend(properties)
        des_props = np.array(des_props)

        # Assemble the cost function
        costs_eff_props = abs(np.divide(des_props - effective_properties, effective_properties))
        costs_cfs = np.multiply(cost_func_weights,
                                abs(np.divide(concentration_factors - tolerance, tolerance)))

        cost = weight_domains * (weight_eff_prop * np.sum(costs_eff_props) + \
                                 weight_conc_factor * np.sum(costs_cfs))

        if include_cost_breakdown:
            return cost, costs_eff_props, costs_cfs

        return cost


    def get_general_eff_prop_and_cfs(self, idx = 0):
        """
        Compute the effective non-modulus properties and concentration factors of a
        composite material using the Hashin-Shtrikman bounds.

        Args:
            idx (int, optional)

        Returns:
            A tuple containing:
            - effective_properties (list)
            - concentration_factors (list)

        Notes:
            idx is the index in self.values where category properties begin
        """

        # Initialize effective property, concentration factor, and weight arrays
        # Initialize to zero so as not to contribute to cost if unchanged
        effective_properties  = []
        concentration_factors = []

        # Prepare indices for looping over properties
        # The last num_materials entries are volume fractions, not material properties
        stop = -self.opt_params.num_materials
        # Subtract 1 so as not to include volume fraction
        step = self.opt_params.num_properties - 1

        # Get Hashin-Shtrikman effective properties for all properties
        volume_fractions = self.values[-self.opt_params.num_materials:]
        properties = self.values[idx:stop:step]

        sorted_properties = np.sort(properties, axis=0)
        sorted_indices = np.argsort(properties, axis=0)
        sorted_vol_fracs = volume_fractions[sorted_indices]

        phase_1 = sorted_properties[0]
        phase_n = sorted_properties[-1]
        if phase_1 == 0:
            alpha_1 = 0
        else:
            alpha_1 = eval(COMPILED_CALC_GUIDE["effective_props"]["alpha_1"],
                           {},
                           {"phase_1": phase_1})
        if phase_n == 0:
            alpha_n = 0
        else:
            alpha_n = eval(COMPILED_CALC_GUIDE["effective_props"]["alpha_n"],
                           {},
                           {"phase_n": phase_n})

        a_1 = 0
        for i in range(1, self.opt_params.num_materials):
            phase_i_vol_frac = sorted_vol_fracs[i]
            phase_i = sorted_properties[i]
            if phase_1 == phase_i:
                a_1_term_i = 0
            else:
                a_1_term_i = eval(COMPILED_CALC_GUIDE["effective_props"]["a_1_term_i"],
                                  {},
                                  {"phase_1":          phase_1,
                                   "phase_i":          phase_i,
                                   "phase_i_vol_frac": phase_i_vol_frac,
                                   "alpha_1":          alpha_1})
            a_1 += a_1_term_i

        a_n = 0
        for i in range(self.opt_params.num_materials - 1):
            phase_i_vol_frac = sorted_vol_fracs[i]
            phase_i = sorted_properties[i]
            if phase_i == phase_n:
                a_n_term_i = 0
            else:
                a_n_term_i = eval(COMPILED_CALC_GUIDE["effective_props"]["a_n_term_i"],
                                  {},
                                 {"phase_n":           phase_n,
                                   "phase_i":          phase_i,
                                   "phase_i_vol_frac": phase_i_vol_frac,
                                   "alpha_n":          alpha_n})
            a_n += a_n_term_i

        # Compute effective property bounds with Hashin-Shtrikman
        if phase_1 == phase_n:
            effective_prop_min = phase_1
            effective_prop_max = phase_n
        else:
            effective_prop_min = eval(COMPILED_CALC_GUIDE["effective_props"]["eff_min"],
                                      {},
                                      {"phase_1": phase_1,
                                       "phase_n": phase_n,
                                       "alpha_1": alpha_1,
                                       "a_1":     a_1})
            effective_prop_max = eval(COMPILED_CALC_GUIDE["effective_props"]["eff_max"],
                                      {},
                                      {"phase_1": phase_1,
                                       "phase_n": phase_n,
                                       "alpha_n": alpha_n,
                                       "a_n":     a_n})

        mixing_param = self.ga_params.mixing_param
        effective_prop = mixing_param * effective_prop_max + (1 - mixing_param) * effective_prop_min
        effective_properties.append(effective_prop)

        # Compute concentration factors for load sharing
        vf_weighted_sum_cf_load_i = 0
        vf_weighted_sum_cf_response_i = 0
        for i in range(1, self.opt_params.num_materials):
            phase_i_vol_frac = sorted_vol_fracs[i]
            phase_i = sorted_properties[i]
            if phase_i_vol_frac == 0:
                cf_load_i = 0
                cf_response_i = 0
            else:
                # if cf_load_i has div by zero error, then set cf_load_i to 0
                try:
                    cf_load_i = eval(
                        COMPILED_CALC_GUIDE["concentration_factors"]["cf_load_i"],
                        {},
                        {"n":                self.opt_params.num_materials,
                         "phase_1":          phase_1,
                         "phase_i":          phase_i,
                         "phase_i_vol_frac": phase_i_vol_frac,
                         "eff_prop":         effective_prop}
                    )
                    cf_response_i = eval(
                        COMPILED_CALC_GUIDE["concentration_factors"]["cf_response_i"],
                        {},
                        {"phase_i":   phase_i,
                         "cf_load_i": cf_load_i,
                         "eff_prop":  effective_prop}
                    )
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
            cf_load_1     = eval(COMPILED_CALC_GUIDE["concentration_factors"]["cf_1"],
                                 {},
                                 {"phase_1_vol_frac":    phase_1_vol_frac,
                                  "vf_weighted_sum_cfs": vf_weighted_sum_cf_load_i})
            cf_response_1 = eval(COMPILED_CALC_GUIDE["concentration_factors"]["cf_1"],
                                 {},
                                 {"phase_1_vol_frac":    phase_1_vol_frac,
                                  "vf_weighted_sum_cfs": vf_weighted_sum_cf_response_i})
        concentration_factors.insert(0, cf_response_1)
        concentration_factors.insert(0, cf_load_1)

        return effective_properties, concentration_factors


    def get_elastic_eff_props_and_cfs(self, idx = 0):
        """
        Compute the effective modulus properties and concentration factors of a composite
        material using the Hashin-Shtrikman bounds.

        Args:
            idx (int, optional)

        Returns:
            A tuple containing:
            - effective_properties (list)
            - concentration_factors (list)

        Notes:
            idx is the index in self.values where elastic properties begin
        """

        # Initialize effective property and concentration factor arrays
        # Initialize to zero so as not to contribute to cost if unchanged
        effective_properties  = []
        concentration_factors = []

        # Prepare indices for looping over properties
        # The last num_materials entries are volume fractions, not material properties
        stop = -self.opt_params.num_materials
        # Subtract 1 so as not to include volume fraction
        step = self.opt_params.num_properties - 1

        # Extract bulk moduli and shear moduli from member
        volume_fractions = self.values[-self.opt_params.num_materials:]

        bulk_mods = self.values[idx:stop:step]
        sorted_bulk = np.sort(bulk_mods, axis=0)
        sorted_bulk_indices = np.argsort(bulk_mods, axis=0)
        sorted_vol_fracs = volume_fractions[sorted_bulk_indices]

        shear_mods = self.values[idx+1:stop:step]
        sorted_shear = np.sort(shear_mods, axis=0)
        sorted_shear_indices = np.argsort(shear_mods, axis=0)

        phase_1_bulk  = sorted_bulk[0]
        phase_n_bulk  = sorted_bulk[-1]
        phase_1_shear = sorted_shear[0]
        phase_n_shear = sorted_shear[-1]

        if (phase_1_bulk == 0) and (phase_1_shear == 0):
            bulk_alpha_1 = 0
        else:
            bulk_alpha_1  = eval(COMPILED_CALC_GUIDE["effective_props"]["bulk_alpha_1"],
                                 {},
                                 {"phase_1_bulk":  phase_1_bulk,
                                  "phase_1_shear": phase_1_shear})
        if (phase_n_bulk == 0) and (phase_n_shear == 0):
            bulk_alpha_n = 0
        else:
            bulk_alpha_n  = eval(COMPILED_CALC_GUIDE["effective_props"]["bulk_alpha_n"],
                                 {},
                                 {"phase_n_bulk":  phase_n_bulk,
                                  "phase_n_shear": phase_n_shear})
        if phase_1_shear == 0:
            shear_alpha_1 = 0
        else:
            shear_alpha_1 = eval(COMPILED_CALC_GUIDE["effective_props"]["shear_alpha_1"],
                                 {},
                                 {"phase_1_bulk":  phase_1_bulk,
                                  "phase_1_shear": phase_1_shear})
        if phase_n_shear == 0:
            shear_alpha_n = 0
        else:
            shear_alpha_n = eval(COMPILED_CALC_GUIDE["effective_props"]["shear_alpha_n"],
                                 {},
                                 {"phase_n_bulk":  phase_n_bulk,
                                  "phase_n_shear": phase_n_shear})

        a_1_bulk = 0
        a_1_shear = 0
        for i in range(1, self.opt_params.num_materials):
            phase_i_vol_frac = sorted_vol_fracs[i]
            phase_i_bulk = sorted_bulk[i]
            if phase_1_bulk == phase_i_bulk:
                a_1_term_i = 0
            else:
                a_1_term_i = eval(COMPILED_CALC_GUIDE["effective_props"]["a_1_term_i"],
                                  {},
                                  {"phase_1":          phase_1_bulk,
                                   "phase_i":          phase_i_bulk,
                                   "phase_i_vol_frac": phase_i_vol_frac,
                                   "alpha_1":          bulk_alpha_1})
            a_1_bulk += a_1_term_i

            phase_i_shear = sorted_shear[i]
            if phase_1_shear == phase_i_shear:
                a_1_term_i = 0
            else:
                a_1_term_i = eval(COMPILED_CALC_GUIDE["effective_props"]["a_1_term_i"],
                                  {},
                                  {"phase_1":          phase_1_shear,
                                   "phase_i":          phase_i_shear,
                                   "phase_i_vol_frac": phase_i_vol_frac,
                                   "alpha_1":          shear_alpha_1})
            a_1_shear += a_1_term_i

        a_n_bulk = 0
        a_n_shear = 0
        for i in range(self.opt_params.num_materials - 1):
            phase_i_vol_frac = sorted_vol_fracs[i]
            phase_i_bulk = sorted_bulk[i]
            if phase_i_bulk == phase_n_bulk:
                a_n_term_i = 0
            else:
                a_n_term_i = eval(COMPILED_CALC_GUIDE["effective_props"]["a_n_term_i"],
                                  {},
                                  {"phase_n":          phase_n_bulk,
                                   "phase_i":          phase_i_bulk,
                                   "phase_i_vol_frac": phase_i_vol_frac,
                                   "alpha_n":          bulk_alpha_n})
            a_n_bulk += a_n_term_i

            phase_i_shear = sorted_shear[i]
            if phase_i_shear == phase_n_shear:
                a_n_term_i = 0
            else:
                a_n_term_i = eval(COMPILED_CALC_GUIDE["effective_props"]["a_n_term_i"],
                                  {},
                                  {"phase_n":          phase_n_shear,
                                   "phase_i":          phase_i_shear,
                                   "phase_i_vol_frac": phase_i_vol_frac,
                                   "alpha_n":          shear_alpha_n})
            a_n_shear += a_n_term_i

        # Compute effective bulk modulus bounds with Hashin-Shtrikman
        if phase_1_bulk == phase_n_bulk:
            effective_bulk_min = phase_1_bulk
            effective_bulk_max = phase_n_bulk
        else:
            effective_bulk_min = eval(COMPILED_CALC_GUIDE["effective_props"]["eff_min"],
                                      {},
                                      {"phase_1": phase_1_bulk,
                                       "phase_n": phase_n_bulk,
                                       "alpha_1": bulk_alpha_1,
                                       "a_1":     a_1_bulk})
            effective_bulk_max = eval(COMPILED_CALC_GUIDE["effective_props"]["eff_max"],
                                      {},
                                      {"phase_1": phase_1_bulk,
                                       "phase_n": phase_n_bulk,
                                       "alpha_n": bulk_alpha_n,
                                       "a_n":     a_n_bulk})

        # Compute effective shear modulus bounds with Hashin-Shtrikman
        if phase_1_bulk == phase_n_bulk:
            effective_shear_min = phase_1_shear
            effective_shear_max = phase_n_shear
        else:
            effective_shear_min = eval(COMPILED_CALC_GUIDE["effective_props"]["eff_min"],
                                       {},
                                       {"phase_1": phase_1_shear,
                                        "phase_n": phase_n_shear,
                                        "alpha_1": shear_alpha_1,
                                        "a_1":     a_1_shear})
            effective_shear_max = eval(COMPILED_CALC_GUIDE["effective_props"]["eff_max"],
                                       {},
                                       {"phase_1": phase_1_shear,
                                        "phase_n": phase_n_shear,
                                        "alpha_n": shear_alpha_n,
                                        "a_n":     a_n_shear})

        # Compute concentration factors for load sharing
        mixing_param = self.ga_params.mixing_param
        effective_bulk  = mixing_param * effective_bulk_max +  \
                          (1 - mixing_param) * effective_bulk_min
        effective_properties.append(effective_bulk)
        effective_shear = mixing_param * effective_shear_max + \
                          (1 - mixing_param) * effective_shear_min
        effective_properties.append(effective_shear)

        vf_weighted_sum_cf_bulk_i = 0
        vf_weighted_sum_cf_shear_i = 0
        for i in range(1, self.opt_params.num_materials):
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
                    cf_bulk_i = eval(COMPILED_CALC_GUIDE["concentration_factors"]["cf_elastic_i"],
                                     {},
                                     {"phase_i_vol_frac": phase_i_vol_frac,
                                      "phase_i_elastic":  phase_1_bulk,
                                      "eff_elastic":      effective_bulk,
                                      "phase_1":          phase_1_bulk,
                                      "phase_i":          phase_i_bulk})
                if phase_1_shear == phase_i_shear:
                    cf_shear_i = 0
                else:
                    cf_shear_i = eval(COMPILED_CALC_GUIDE["concentration_factors"]["cf_elastic_i"],
                                      {},
                                      {"phase_i_vol_frac": phase_i_vol_frac,
                                       "phase_i_elastic":  phase_1_shear,
                                       "eff_elastic":      effective_shear,
                                       "phase_1":          phase_1_shear,
                                       "phase_i":          phase_i_shear})

            concentration_factors.append(cf_bulk_i)
            concentration_factors.append(cf_shear_i)
            vf_weighted_sum_cf_bulk_i  += phase_i_vol_frac * cf_bulk_i
            vf_weighted_sum_cf_shear_i += phase_i_vol_frac * cf_shear_i

        phase_1_vol_frac = sorted_vol_fracs[0]
        if phase_1_vol_frac == 0:
            cf_bulk_1 = 0
            cf_shear_1 = 0
        else:
            cf_bulk_1  = eval(COMPILED_CALC_GUIDE["concentration_factors"]["cf_1"],
                              {},
                              {"phase_1_vol_frac":    phase_1_vol_frac,
                               "vf_weighted_sum_cfs": vf_weighted_sum_cf_bulk_i})
            cf_shear_1 = eval(COMPILED_CALC_GUIDE["concentration_factors"]["cf_1"],
                              {},
                              {"phase_1_vol_frac":    phase_1_vol_frac,
                               "vf_weighted_sum_cfs": vf_weighted_sum_cf_shear_i})
        concentration_factors.insert(0, cf_shear_1)
        concentration_factors.insert(0, cf_bulk_1)

        return effective_properties, concentration_factors


    def get_effective_properties(self):
        """
        Computes the effective properties of a material system based on the 
        Hashin-Shtrikman bounds for various property categories.

        Returns:
            effective properties (ndarray)
        """

        # Initialize effective property array
        effective_properties  = []

        # Get Hashin-Shtrikman effective properties for all properties
        idx = 0

        for category in self.opt_params.property_categories:

            if category == "elastic":
                moduli_eff_props, _ = self.get_elastic_eff_props_and_cfs(idx=idx)
                effective_properties.extend(moduli_eff_props)

                eff_univ_aniso, _ = self.get_general_eff_prop_and_cfs(idx=idx+2)
                effective_properties.extend(eff_univ_aniso)

            else:
                # Loop through all properties in the category
                for p in range(idx, idx + len(self.opt_params.property_docs[category])):
                    new_eff_props, _ = self.get_general_eff_prop_and_cfs(idx=p)
                    effective_properties.extend(new_eff_props)

            idx += len(self.opt_params.property_docs[category])

        # Cast effective properties to numpy arrays
        return np.array(effective_properties)
