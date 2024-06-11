# cmember.pyx

# distutils: language=c++
# Define to silence the deprecation warning
cdef extern from *:
    """
    #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
    """

import numpy as np
cimport numpy as cnp
cimport cython
from custom_logger import logger
from genetic_algo import GAParams
from typing import List, Dict, Any
import warnings

cdef class CMember:

    cdef int num_materials
    cdef int num_properties
    cdef cnp.ndarray values
    cdef list property_categories
    cdef dict property_docs
    cdef dict desired_props
    cdef object ga_params
    cdef dict calc_guide

    def __init__(self, 
        int num_materials, 
        int num_properties, 
        cnp.ndarray[double, ndim=1] values, # check for ndim
        list property_categories, 
        dict property_docs, 
        dict desired_props, 
        object ga_params, 
        dict calc_guide):
        
        self.num_materials = num_materials
        self.num_properties = num_properties
        self.values = values if values is not None else np.zeros((num_properties, 1))
        self.property_categories = property_categories
        self.property_docs = property_docs
        self.desired_props = desired_props
        self.ga_params = ga_params
        self.calc_guide = calc_guide

    cpdef double get_cost(self):
        cdef double tolerance = self.ga_params.tolerance
        cdef double weight_eff_prop = self.ga_params.weight_eff_prop
        cdef double weight_conc_factor = self.ga_params.weight_conc_factor

        cdef list effective_properties = []
        cdef list concentration_factors = []
        cdef list cost_func_weights = []

        cdef int idx = 0
        cdef int p
        cdef list moduli_eff_props, moduli_cfs, eff_univ_aniso, cfs_univ_aniso, new_eff_props, new_cfs
        cdef str category
        cdef double factor

        for category in self.property_categories:
            if category == "elastic":
                moduli_eff_props, moduli_cfs = self.get_elastic_eff_props_and_cfs(idx=idx)
                effective_properties.extend(moduli_eff_props)
                concentration_factors.extend(moduli_cfs)
                eff_univ_aniso, cfs_univ_aniso = self.get_general_eff_prop_and_cfs(idx=idx + 2)
                effective_properties.extend(eff_univ_aniso)
                concentration_factors.extend(cfs_univ_aniso)
            else:
                for p in range(idx, idx + len(self.property_docs[category])):
                    new_eff_props, new_cfs = self.get_general_eff_prop_and_cfs(idx=p)
                    effective_properties.extend(new_eff_props)
                    concentration_factors.extend(new_cfs)
            idx += len(self.property_docs[category])

        for factor in concentration_factors:
            if (factor - tolerance) / tolerance > 0:
                cost_func_weights.append(weight_conc_factor)
            else:
                cost_func_weights.append(0)

        cdef cnp.ndarray[cnp.double_t, ndim=1] concentration_factors_np = np.array(concentration_factors, dtype=np.double)
        cdef cnp.ndarray[cnp.double_t, ndim=1] effective_properties_np = np.array(effective_properties, dtype=np.double)
        cdef cnp.ndarray[cnp.double_t, ndim=1] cost_func_weights_np = np.array(cost_func_weights, dtype=np.double)

        des_props = [prop for cat in self.desired_props.values() for prop in cat]
        cdef cnp.ndarray[cnp.double_t, ndim=1] des_props_np = np.array(des_props, dtype=np.double)

        cdef int domains = len(self.property_categories)
        cdef double W = 1.0 / domains
        cdef double cost
        cdef double eff_prop_sum = 0.0
        cdef double conc_factor_sum = 0.0
        cdef int i, n

        n = des_props_np.shape[0]
        for i in range(n):
            eff_prop_sum += abs((des_props_np[i] - effective_properties_np[i]) / effective_properties_np[i])

        n = concentration_factors_np.shape[0]
        for i in range(n):
            conc_factor_sum += cost_func_weights_np[i] * abs((concentration_factors_np[i] - tolerance) / tolerance)

        cost = (weight_eff_prop * W * eff_prop_sum) + conc_factor_sum

        return cost

    cpdef tuple get_general_eff_prop_and_cfs(self, int idx=0):
        cdef double effective_prop = 0.0
        cdef double effective_prop_min = 0.0
        cdef double effective_prop_max = 0.0
        cdef double phase1, phase2, phase1_vol_frac, phase2_vol_frac
        cdef double cf_response1_cf_load1 = 0.0
        cdef double cf_response2_cf_load2 = 0.0
        cdef double mixing_param = self.ga_params.mixing_param
        cdef list effective_properties = []
        cdef list concentration_factors = []

        cdef int stop = -self.num_materials
        cdef int step = self.num_properties - 1
        cdef cnp.ndarray[cnp.double_t, ndim=1] properties = self.values[idx:stop:step]
        cdef int phase1_idx, phase2_idx

        phase1 = np.min(properties)
        phase2 = np.max(properties)
        phase1_idx = np.argmin(properties)
        phase2_idx = np.argmax(properties)
        phase1_vol_frac = self.values[-self.num_materials + phase1_idx]
        phase2_vol_frac = self.values[-self.num_materials + phase2_idx]

        if phase1 == phase2:
            effective_prop_min = phase1
            effective_prop_max = phase2
        else:
            effective_prop_min = eval(self.calc_guide['effective_props']['eff_min'].format(phase1=phase1, phase2=phase2, phase1_vol_frac=phase1_vol_frac, phase2_vol_frac=phase2_vol_frac))
            effective_prop_max = eval(self.calc_guide['effective_props']['eff_max'].format(phase1=phase1, phase2=phase2, phase1_vol_frac=phase1_vol_frac, phase2_vol_frac=phase2_vol_frac))

        effective_prop = mixing_param * effective_prop_max + (1 - mixing_param) * effective_prop_min
        effective_properties.append(effective_prop)

        if phase1_vol_frac == 0:
            cf_response1_cf_load1 = (1 / phase2_vol_frac) ** 2
            cf_response2_cf_load2 = (1 / phase2_vol_frac) ** 2
        elif phase2_vol_frac == 0:
            cf_response1_cf_load1 = (1 / phase1_vol_frac) ** 2
            cf_response2_cf_load2 = (1 / phase1_vol_frac) ** 2
        elif phase1 == phase2:
            cf_response1_cf_load1 = (1 / phase1_vol_frac) ** 2
            cf_response2_cf_load2 = (1 / phase2_vol_frac) ** 2
        else:
            cf_response1_cf_load1 = eval(self.calc_guide['concentration_factors']['cf_1'].format(phase1=phase1, phase2=phase2, phase1_vol_frac=phase1_vol_frac, effective_property=effective_prop))
            cf_response2_cf_load2 = eval(self.calc_guide['concentration_factors']['cf_2'].format(phase1=phase1, phase2=phase2, phase2_vol_frac=phase2_vol_frac, effective_property=effective_prop))

        concentration_factors.append(cf_response1_cf_load1)
        concentration_factors.append(cf_response2_cf_load2)

        return effective_properties, concentration_factors

    cpdef tuple get_elastic_eff_props_and_cfs(self, int idx=0):
        cdef double effective_bulk_mod_min = 0.0
        cdef double effective_bulk_mod_max = 0.0
        cdef double effective_shear_mod_min = 0.0
        cdef double effective_shear_mod_max = 0.0
        cdef double phase1_bulk, phase2_bulk, phase1_shear, phase2_shear
        cdef double phase1_vol_frac, phase2_vol_frac, bulk_mod_eff = 0.0
        cdef double shear_mod_eff = 0.0
        cdef double cf_phase1_bulk = 0.0
        cdef double cf_phase2_bulk = 0.0
        cdef double cf_phase1_shear = 0.0
        cdef double cf_phase2_shear = 0.0
        cdef double mixing_param = self.ga_params.mixing_param
        cdef list effective_properties = []
        cdef list concentration_factors = []

        cdef int stop = idx + (self.num_properties - 1) * self.num_materials
        cdef int step = self.num_properties

        bulk_mods = self.values[idx:stop + 1:step]
        phase1_bulk = np.min(bulk_mods)
        phase2_bulk = np.max(bulk_mods)
        phase1_bulk_idx = np.argmin(bulk_mods)
        phase2_bulk_idx = np.argmax(bulk_mods)

        shear_mods = self.values[idx + 1:stop + 1:step]
        phase1_shear = np.min(shear_mods)
        phase2_shear = np.max(shear_mods)
        phase1_shear_idx = np.argmin(shear_mods)
        phase2_shear_idx = np.argmax(shear_mods)

        if (phase1_bulk_idx != phase1_shear_idx) or (phase2_bulk_idx != phase2_shear_idx):
            warnings.warn("Cannot perform optimization when for bulk modulus phase 1 > phase 2 and for shear modulus phase 2 > phase 1 or vice versa.")

        phase1_vol_frac = self.values[-self.num_materials + phase1_bulk_idx]
        phase2_vol_frac = self.values[-self.num_materials + phase2_bulk_idx]

        if phase1_bulk == phase2_bulk:
            effective_bulk_mod_min = phase1_bulk
            effective_bulk_mod_max = phase2_bulk
        else:
            effective_bulk_mod_min = eval(self.calc_guide['effective_props']['bulk_mod_min'].format(phase1_bulk=phase1_bulk, phase2_bulk=phase2_bulk, phase1_shear=phase1_shear, phase2_shear=phase2_shear, phase1_vol_frac=phase1_vol_frac, phase2_vol_frac=phase2_vol_frac))
            effective_bulk_mod_max = eval(self.calc_guide['effective_props']['bulk_mod_max'].format(phase1_bulk=phase1_bulk, phase2_bulk=phase2_bulk, phase1_shear=phase1_shear, phase2_shear=phase2_shear, phase1_vol_frac=phase1_vol_frac, phase2_vol_frac=phase2_vol_frac))

        if phase1_shear == phase2_shear:
            effective_shear_mod_min = phase1_shear
            effective_shear_mod_max = phase2_shear
        else:
            effective_shear_mod_min = eval(self.calc_guide['effective_props']['shear_mod_min'].format(phase1_bulk=phase1_bulk, phase2_bulk=phase2_bulk, phase1_shear=phase1_shear, phase2_shear=phase2_shear, phase1_vol_frac=phase1_vol_frac, phase2_vol_frac=phase2_vol_frac))
            effective_shear_mod_max = eval(self.calc_guide['effective_props']['shear_mod_max'].format(phase1_bulk=phase1_bulk, phase2_bulk=phase2_bulk, phase1_shear=phase1_shear, phase2_shear=phase2_shear, phase1_vol_frac=phase1_vol_frac, phase2_vol_frac=phase2_vol_frac))

        bulk_mod_eff = mixing_param * effective_bulk_mod_max + (1 - mixing_param) * effective_bulk_mod_min
        shear_mod_eff = mixing_param * effective_shear_mod_max + (1 - mixing_param) * effective_shear_mod_min

        effective_properties.append(bulk_mod_eff)
        effective_properties.append(shear_mod_eff)

        if phase1_vol_frac == 0:
            cf_phase2_bulk = 1 / phase2_vol_frac
            cf_phase1_bulk = 1 / phase2_vol_frac
            cf_phase2_shear = 1 / phase2_vol_frac
            cf_phase1_shear = 1 / phase2_vol_frac
        elif phase2_vol_frac == 0:
            cf_phase2_bulk = 1 / phase1_vol_frac
            cf_phase1_bulk = 1 / phase1_vol_frac
            cf_phase2_shear = 1 / phase1_vol_frac
            cf_phase1_shear = 1 / phase1_vol_frac
        elif phase1_bulk == phase2_bulk:
            cf_phase2_bulk = 1 / phase2_vol_frac
            cf_phase1_bulk = 1 / phase1_vol_frac
        else:
            cf_phase2_bulk = eval(self.calc_guide['concentration_factors']['cf_2_elastic'].format(phase2_vol_frac=phase2_vol_frac, phase1=phase1_bulk, phase2=phase2_bulk, effective_property=bulk_mod_eff))
            cf_phase1_bulk = eval(self.calc_guide['concentration_factors']['cf_1_elastic'].format(phase1_vol_frac=phase1_vol_frac, phase2_vol_frac=phase2_vol_frac, cf_2_elastic=cf_phase2_bulk))

        if phase1_vol_frac == 0:
            cf_phase2_shear = 1 / phase2_vol_frac
            cf_phase1_shear = 1 / phase2_vol_frac
        elif phase2_vol_frac == 0:
            cf_phase2_shear = 1 / phase1_vol_frac
            cf_phase1_shear = 1 / phase1_vol_frac
        elif phase1_shear == phase2_shear:
            cf_phase2_shear = 1 / phase2_vol_frac
            cf_phase1_shear = 1 / phase1_vol_frac
        else:
            cf_phase2_shear = eval(self.calc_guide['concentration_factors']['cf_2_elastic'].format(phase2_vol_frac=phase2_vol_frac, phase1=phase1_shear, phase2=phase2_shear, effective_property=shear_mod_eff))
            cf_phase1_shear = eval(self.calc_guide['concentration_factors']['cf_1_elastic'].format(phase1_vol_frac=phase1_vol_frac, phase2_vol_frac=phase2_vol_frac, cf_2_elastic=cf_phase2_shear))

        concentration_factors.append(cf_phase1_bulk)
        concentration_factors.append(cf_phase1_shear)
        concentration_factors.append(cf_phase2_bulk)
        concentration_factors.append(cf_phase2_shear)

        return effective_properties, concentration_factors