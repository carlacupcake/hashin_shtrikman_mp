# From MPRester
import itertools
import re
import warnings
import json
from functools import lru_cache
from json import loads
from os import environ
from typing import Any, Dict, List, Literal, Optional, Union

from emmet.core.mpid import MPID
from emmet.core.settings import EmmetSettings
from emmet.core.summary import HasProps
from packaging import version
from requests import Session, get

from mp_api.client.core.settings import MAPIClientSettings
from mp_api.client.core.utils import validate_ids

# Custom Classes
from ga_params_class import GAParams
from member_class import Member
from population_class import Population
from user_input_class import UserInput

# Other
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
from mp_api.client import MPRester
from mpcontribs.client import Client
from tabulate import tabulate
from hs_logger import logger


_DEPRECATION_WARNING = (
    "MPRester is being modernized. Please use the new method suggested and "
    "read more about these changes at https://docs.materialsproject.org/api. The current "
    "methods will be retained until at least January 2022 for backwards compatibility."
)

_EMMET_SETTINGS = EmmetSettings()
_MAPI_SETTINGS = MAPIClientSettings()

# HashinShtrikman class defaults
DEFAULT_API_KEY       = environ.get("MP_API_KEY", None)
DEFAULT_ENDPOINT      = environ.get("MP_API_ENDPOINT", "https://api.materialsproject.org/")
DEFAULT_PROPERTY_DOCS = ["carrier-transport", "dielectric", "elastic", "magnetic", "piezoelectric"]
DEFAULT_DESIRED_PROPS = {"carrier-transport": [],
                         "dielectric": [],
                         "elastic": [],
                         "magnetic": [],
                         "piezoelectric": []}
DEFAULT_HAS_PROPS     = [HasProps.dielectric, HasProps.elasticity]
DEFAULT_FIELDS        = {"material_id": [], 
                         "is_stable": [], 
                         "band_gap": [], 
                         "is_metal": [],
                         "formula": [],}
DEFAULT_LOWER_BOUNDS  = {"mat1": DEFAULT_DESIRED_PROPS,
                         "mat2": DEFAULT_DESIRED_PROPS}
DEFAULT_UPPER_BOUNDS  = {"mat1": DEFAULT_DESIRED_PROPS,
                         "mat2": DEFAULT_DESIRED_PROPS} # could generalize to more materials later

class HashinShtrikman:

    #------ Initialization method ------#
    def __init__(
            self,
            api_key:              Optional[str] = None,
            mp_contribs_project:  Optional[str] = None,
            endpoint:             str           = DEFAULT_ENDPOINT,
            user_input:           UserInput     = UserInput(),
            property_docs:        list          = DEFAULT_PROPERTY_DOCS,
            desired_props:        dict          = DEFAULT_DESIRED_PROPS,
            has_props:            list          = DEFAULT_HAS_PROPS,
            fields:               dict          = DEFAULT_FIELDS,
            num_properties:       int           = 0,
            lower_bounds:         dict          = DEFAULT_LOWER_BOUNDS,
            upper_bounds:         dict          = DEFAULT_UPPER_BOUNDS, 
            ga_params:            GAParams      = GAParams(),
            final_population:     Population    = Population(),
            cost_history:         np.ndarray    = np.empty,   
            lowest_costs:         np.ndarray    = np.empty,          
            avg_parent_costs:     np.ndarray    = np.empty, 
        ):
            
            self.api_key              = api_key 
            self.mp_contribs_project  = mp_contribs_project
            self.endpoint             = endpoint
            self.user_input           = user_input
            self.property_docs        = property_docs
            self.desired_props        = desired_props 
            self.has_props            = has_props
            self.fields               = fields
            self.num_properties       = num_properties  
            self.lower_bounds         = lower_bounds 
            self.upper_bounds         = upper_bounds
            self.ga_params            = ga_params 
            self.final_population     = final_population
            self.cost_history         = cost_history            
            self.lowest_costs         = lowest_costs
            self.avg_parent_costs     = avg_parent_costs

            # Update from default based on self.user_input
            self.set_desired_props_from_user_input()
            self.set_num_properties_from_desired_props()
            self.set_lower_bounds_from_user_input()
            self.set_upper_bounds_from_user_input() 

            try:
                from mpcontribs.client import Client
                self.contribs = Client(api_key, project="carrier_transport")
            except ImportError:
                self.contribs = None
                warnings.warn(
                    "mpcontribs-client not installed."
                    "Install the package to query MPContribs data:"
                    "pip install mpcontribs-client"
                )
            except Exception as error:
                self.contribs = None
                warnings.warn(f"Problem loading MPContribs client: {error}")

            if not self.endpoint.endswith("/"):
                self.endpoint += "/"


    #------ Getter Methods ------#

    def get_lower_bounds(self):
        return self.lower_bounds
    
    def get_upper_bounds(self):
        return self.upper_bounds
    
    def get_retain_parents(self):
        return self.retain_parents
    
    def get_allow_mutations(self):
        return self.allow_mutations

    def get_property_docs(self):
        return self.property_docs
    
    def get_desired_props(self):
        return self.desired_props
    
    def get_has_props(self):
        return self.has_props
    
    def get_fields(self):
        return self.fields
    
    def get_num_properties(self):
        return self.num_properties
    
    def get_ga_params(self):
        return self.ga_params
    
    def get_final_population(self):
        return self.final_population
    
    def get_cost_history(self):
        return self.cost_history
    
    def get_lowest_costs(self):
        return self.lowest_costs           

    def get_avg_parent_costs(self):
        return self.avg_parent_costs
    
    def get_headers(self, include_mpids=False):

        headers = []
        if include_mpids:
            headers.append("Material 1 MP-ID")
            headers.append("Material 2 MP-ID")
        if "carrier-transport" in self.property_docs:
            headers.append("(Phase 1) Electrical conductivity, [S/m]")
            headers.append("(Phase 1) Thermal conductivity, [W/m/K]")
            headers.append("(Phase 2) Electrical conductivity, [S/m]")
            headers.append("(Phase 2) Thermal conductivity, [W/m/K]")
        if "dielectric" in self.property_docs:
            headers.append("(Phase 1) Total dielectric constant, [F/m]")
            headers.append("(Phase 1) Ionic contrib dielectric constant, [F/m]")
            headers.append("(Phase 1) Electronic contrib dielectric constant, [F/m]")
            headers.append("(Phase 1) Dielectric n, [F/m]")
            headers.append("(Phase 2) Total dielectric constant, [F/m]")
            headers.append("(Phase 2) Ionic contrib dielectric constant, [F/m]")
            headers.append("(Phase 2) Electronic contrib dielectric constant, [F/m]")
            headers.append("(Phase 2) Dielectric n, [F/m]")
        if "elastic" in self.property_docs:
            headers.append("(Phase 1) Bulk modulus, [GPa]")
            headers.append("(Phase 1) Shear modulus, [GPa]")
            headers.append("(Phase 1) Universal anisotropy, []")
            headers.append("(Phase 2) Bulk modulus, [GPa]")
            headers.append("(Phase 2) Shear modulus, [GPa]")
            headers.append("(Phase 2) Universal anisotropy, []")
        if "magnetic" in self.property_docs:
            headers.append("(Phase 1) Total magnetization, []")
            headers.append("(Phase 1) Total magnetization normalized volume, []")
            headers.append("(Phase 2) Total magnetization, []")
            headers.append("(Phase 2) Total magnetization normalized volume, []")
        if "piezoelectric" in self.property_docs:
            headers.append("(Phase 1) Piezoelectric constant, [C/N or m/V]")
            headers.append("(Phase 2) Piezoelectric constant, [C/N or m/V]")

        headers.extend(["Mixing paramter, []",
                        "(Phase 1) Volume fraction, [] ",
                        "Cost, []"])
        
        return headers
    
    def get_unique_designs(self):

        # Costs are often equal to >10 decimal points, truncate to obtain a richer set of suggestions
        self.final_population.set_costs()
        final_costs = self.final_population.get_costs()
        rounded_costs = np.round(final_costs, decimals=3)
    
        # Obtain unique members and costs
        [unique_costs, unique_indices] = np.unique(rounded_costs, return_index=True)
        unique_members = self.final_population.values[unique_indices]

        return [unique_members, unique_costs] 

    def get_table_of_best_designs(self):

        [unique_members, unique_costs] = self.get_unique_designs()     
        table_data = np.hstack((unique_members[0:self.user_input.num_results, :], unique_costs[0:self.user_input.num_results].reshape(-1, 1))) 

        return table_data

    def get_dict_of_best_designs(self):

        # Initialize a general dictionary for each material
        material_property_dict = {}
        if "carrier-transport" in self.property_docs:
            material_property_dict["carrier-transport"] = {}
            material_property_dict["carrier-transport"]["elec_cond_300K_low_doping"] = []
            material_property_dict["carrier-transport"]["therm_cond_300K_low_doping"] = []
        if "dielectric" in self.property_docs:
            material_property_dict["dielectric"] = {}
            material_property_dict["dielectric"]["e_total"] = []
            material_property_dict["dielectric"]["e_ionic"] = []
            material_property_dict["dielectric"]["e_electronic"] = []
            material_property_dict["dielectric"]["n"] = []
        if "elastic" in self.property_docs:
            material_property_dict["elastic"] = {}
            material_property_dict["elastic"]["bulk_modulus"] = []
            material_property_dict["elastic"]["shear_modulus"] = []
            material_property_dict["elastic"]["universal_anisotropy"] = []
        if "magnetic" in self.property_docs:
            material_property_dict["magnetic"] = {}
            material_property_dict["magnetic"]["total_magnetization"] = []
            material_property_dict["magnetic"]["total_magnetization_normalized_volume"] = []
        if "piezoelectric" in self.property_docs:
            material_property_dict["piezoelectric"] = {}
            material_property_dict["piezoelectric"]["e_ij"] = []

        # Use the general material dictionary to create a nested dictionary
        best_designs_dict = {"mat1": material_property_dict,
                             "mat2": material_property_dict} # could generalize to more materials later 

        [unique_members, unique_costs] = self.get_unique_designs()   
        
        for i in range(len(unique_costs)):
            idx = 0

            if "carrier-transport" in self.property_docs:
                best_designs_dict["mat1"]["carrier-transport"]["elec_cond_300K_low_doping"].append(unique_members[i, idx])
                idx += 1
                best_designs_dict["mat1"]["carrier-transport"]["therm_cond_300K_low_doping"].append(unique_members[i, idx])
                idx += 1
                best_designs_dict["mat2"]["carrier-transport"]["elec_cond_300K_low_doping"].append(unique_members[i, idx])
                idx += 1
                best_designs_dict["mat2"]["carrier-transport"]["therm_cond_300K_low_doping"].append(unique_members[i, idx])
                idx += 1

            if "dielectric" in self.property_docs:
                best_designs_dict["mat1"]["dielectric"]["e_total"].append(unique_members[i, idx])
                idx += 1
                best_designs_dict["mat1"]["dielectric"]["e_ionic"].append(unique_members[i, idx])
                idx += 1
                best_designs_dict["mat1"]["dielectric"]["e_electronic"].append(unique_members[i, idx])
                idx += 1
                best_designs_dict["mat1"]["dielectric"]["n"].append(unique_members[i, idx])
                idx += 1
                best_designs_dict["mat2"]["dielectric"]["e_total"].append(unique_members[i, idx])
                idx += 1
                best_designs_dict["mat2"]["dielectric"]["e_ionic"].append(unique_members[i, idx])
                idx += 1
                best_designs_dict["mat2"]["dielectric"]["e_electronic"].append(unique_members[i, idx])
                idx += 1
                best_designs_dict["mat2"]["dielectric"]["n"].append(unique_members[i, idx])
                idx += 1

            if "elastic" in self.property_docs:
                best_designs_dict["mat1"]["elastic"]["bulk_modulus"].append(unique_members[i, idx])
                idx += 1
                best_designs_dict["mat1"]["elastic"]["shear_modulus"].append(unique_members[i, idx])
                idx += 1
                best_designs_dict["mat1"]["elastic"]["universal_anisotropy"].append(unique_members[i, idx])
                idx += 1
                best_designs_dict["mat2"]["elastic"]["bulk_modulus"].append(unique_members[i, idx])
                idx += 1
                best_designs_dict["mat2"]["elastic"]["shear_modulus"].append(unique_members[i, idx])
                idx += 1
                best_designs_dict["mat2"]["elastic"]["universal_anisotropy"].append(unique_members[i, idx])
                idx += 1

            if "magnetic" in self.property_docs:
                best_designs_dict["mat1"]["magnetic"]["total_magnetization"].append(unique_members[i, idx])
                idx += 1  
                best_designs_dict["mat1"]["magnetic"]["total_magnetization_normalized_volume"].append(unique_members[i, idx])
                idx += 1
                best_designs_dict["mat2"]["magnetic"]["total_magnetization"].append(unique_members[i, idx])
                idx += 1
                best_designs_dict["mat2"]["magnetic"]["total_magnetization_normalized_volume"].append(unique_members[i, idx])
                idx += 1

            if "piezoelectric" in self.property_docs:
                best_designs_dict["mat1"]["piezoelectric"]["e_ij"].append(unique_members[i, idx])
                idx += 1
                best_designs_dict["mat2"]["piezoelectric"]["e_ij"].append(unique_members[i, idx])
                idx += 1

        return best_designs_dict        
    
    def get_material_matches(self, consolidated_dict: dict = {}): 

        best_designs_dict = self.get_dict_of_best_designs()
        mat1_matches: set = set()
        mat2_matches: set = set()
        
        # TODO get from latest final_dict file: change this to a method that reads from the latest MP database
        if consolidated_dict == {}:
            with open("test_final_dict") as f:
                consolidated_dict = json.load(f)

        # Carrier transport extrema
        if "carrier-transport" in self.property_docs:
            mat_1_elec_cond_lower_bound  = min(best_designs_dict["mat1"]["carrier-transport"]["elec_cond_300K_low_doping"])
            mat_1_elec_cond_upper_bound  = max(best_designs_dict["mat1"]["carrier-transport"]["elec_cond_300K_low_doping"])
            mat_2_elec_cond_lower_bound  = min(best_designs_dict["mat2"]["carrier-transport"]["elec_cond_300K_low_doping"])
            mat_2_elec_cond_upper_bound  = max(best_designs_dict["mat2"]["carrier-transport"]["elec_cond_300K_low_doping"])

            mat_1_therm_cond_lower_bound  = min(best_designs_dict["mat1"]["carrier-transport"]["therm_cond_300K_low_doping"])
            mat_1_therm_cond_upper_bound  = max(best_designs_dict["mat1"]["carrier-transport"]["therm_cond_300K_low_doping"])
            mat_2_therm_cond_lower_bound  = min(best_designs_dict["mat2"]["carrier-transport"]["therm_cond_300K_low_doping"])
            mat_2_therm_cond_upper_bound  = max(best_designs_dict["mat2"]["carrier-transport"]["therm_cond_300K_low_doping"])

        # Dielectric extrema
        if "dielectric" in self.property_docs:
            mat_1_e_total_lower_bound = min(best_designs_dict["mat1"]["dielectric"]["e_total"])
            mat_1_e_total_upper_bound = max(best_designs_dict["mat1"]["dielectric"]["e_total"])
            mat_2_e_total_lower_bound = min(best_designs_dict["mat2"]["dielectric"]["e_total"])
            mat_2_e_total_upper_bound = max(best_designs_dict["mat2"]["dielectric"]["e_total"])

            mat_1_e_ionic_lower_bound = min(best_designs_dict["mat1"]["dielectric"]["e_ionic"])
            mat_1_e_ionic_upper_bound = max(best_designs_dict["mat1"]["dielectric"]["e_ionic"])
            mat_2_e_ionic_lower_bound = min(best_designs_dict["mat2"]["dielectric"]["e_ionic"])
            mat_2_e_ionic_upper_bound = max(best_designs_dict["mat2"]["dielectric"]["e_ionic"])

            mat_1_e_elec_lower_bound  = min(best_designs_dict["mat1"]["dielectric"]["e_electronic"])
            mat_1_e_elec_upper_bound  = max(best_designs_dict["mat1"]["dielectric"]["e_electronic"])
            mat_2_e_elec_lower_bound  = min(best_designs_dict["mat2"]["dielectric"]["e_electronic"])
            mat_2_e_elec_upper_bound  = max(best_designs_dict["mat2"]["dielectric"]["e_electronic"])

            mat_1_n_lower_bound       = min(best_designs_dict["mat1"]["dielectric"]["n"])
            mat_1_n_upper_bound       = max(best_designs_dict["mat1"]["dielectric"]["n"])
            mat_2_n_lower_bound       = min(best_designs_dict["mat2"]["dielectric"]["n"])
            mat_2_n_upper_bound       = max(best_designs_dict["mat2"]["dielectric"]["n"])

        # Elastic extrema
        if "elastic" in self.property_docs:
            mat_1_bulk_mod_lower_bound   = min(best_designs_dict["mat1"]["elastic"]["bulk_modulus"])
            mat_1_bulk_mod_upper_bound   = max(best_designs_dict["mat1"]["elastic"]["bulk_modulus"])
            mat_2_bulk_mod_lower_bound   = min(best_designs_dict["mat2"]["elastic"]["bulk_modulus"])
            mat_2_bulk_mod_upper_bound   = max(best_designs_dict["mat2"]["elastic"]["bulk_modulus"])

            mat_1_shear_mod_lower_bound  = min(best_designs_dict["mat1"]["elastic"]["shear_modulus"])
            mat_1_shear_mod_upper_bound  = max(best_designs_dict["mat1"]["elastic"]["shear_modulus"])
            mat_2_shear_mod_lower_bound  = min(best_designs_dict["mat2"]["elastic"]["shear_modulus"])
            mat_2_shear_mod_upper_bound  = max(best_designs_dict["mat2"]["elastic"]["shear_modulus"])

            mat_1_univ_aniso_lower_bound = min(best_designs_dict["mat1"]["elastic"]["universal_anisotropy"])
            mat_1_univ_aniso_upper_bound = max(best_designs_dict["mat1"]["elastic"]["universal_anisotropy"])
            mat_2_univ_aniso_lower_bound = min(best_designs_dict["mat2"]["elastic"]["universal_anisotropy"])
            mat_2_univ_aniso_upper_bound = max(best_designs_dict["mat2"]["elastic"]["universal_anisotropy"])

        # Magnetic extrema
        if "magnetic" in self.property_docs:
            mat_1_tot_mag_lower_bound      = min(best_designs_dict["mat1"]["magnetic"]["total_magnetization"])
            mat_1_tot_mag_upper_bound      = max(best_designs_dict["mat1"]["magnetic"]["total_magnetization"])
            mat_2_tot_mag_lower_bound      = min(best_designs_dict["mat2"]["magnetic"]["total_magnetization"])
            mat_2_tot_mag_upper_bound      = max(best_designs_dict["mat2"]["magnetic"]["total_magnetization"])

            mat_1_tot_mag_norm_lower_bound = min(best_designs_dict["mat1"]["magnetic"]["total_magnetization_normalized_volume"])
            mat_1_tot_mag_norm_upper_bound = max(best_designs_dict["mat1"]["magnetic"]["total_magnetization_normalized_volume"])
            mat_2_tot_mag_norm_lower_bound = min(best_designs_dict["mat2"]["magnetic"]["total_magnetization_normalized_volume"])
            mat_2_tot_mag_norm_upper_bound = max(best_designs_dict["mat2"]["magnetic"]["total_magnetization_normalized_volume"])

        # Piezoelectric extrema
        if "piezoelectric" in self.property_docs:
            mat_1_e_ij_lower_bound = min(best_designs_dict["mat1"]["piezoelectric"]["e_ij"])
            mat_1_e_ij_upper_bound = max(best_designs_dict["mat1"]["piezoelectric"]["e_ij"])
            mat_2_e_ij_lower_bound = min(best_designs_dict["mat2"]["piezoelectric"]["e_ij"])
            mat_2_e_ij_upper_bound = max(best_designs_dict["mat2"]["piezoelectric"]["e_ij"])        

        # Get materials that fall within the above extrema
        # Get intersections of material matches for each property to populate overall matches
        # Check for materials that meet carrier transport criteria
        mat_1_elec_idx = []
        mat_2_elec_idx = []

        mat_1_therm_idx = [] 
        mat_2_therm_idx = [] 
        
        # I want to convert length of list to a new list of indices starting from 0 to len(list) 
        mat1_matches = set(range(len(consolidated_dict["material_id"])))
        mat2_matches = set(range(len(consolidated_dict["material_id"])))

        if "carrier-transport" in self.property_docs:
            for i, elec_cond in enumerate(consolidated_dict["elec_cond_300K_low_doping"]):
                if (elec_cond >= mat_1_elec_cond_lower_bound) and (elec_cond <= mat_1_elec_cond_upper_bound):  
                    mat_1_elec_idx.append(i)
                if (elec_cond >= mat_2_elec_cond_lower_bound) and (elec_cond <= mat_2_elec_cond_upper_bound):  
                    mat_2_elec_idx.append(i) 

            for i, therm_cond in enumerate(consolidated_dict["therm_cond_300K_low_doping"]):
                if (therm_cond >= mat_1_therm_cond_lower_bound) and (therm_cond <= mat_1_therm_cond_upper_bound):  
                    mat_1_therm_idx.append(i)
                if (therm_cond >= mat_2_therm_cond_lower_bound) and (therm_cond <= mat_2_therm_cond_upper_bound):  
                    mat_2_therm_idx.append(i)

            mat1_matches = mat1_matches & set(mat_1_elec_idx) & set(mat_1_therm_idx)
            mat2_matches = mat2_matches & set(mat_2_elec_idx) & set(mat_2_therm_idx)

        # Check for materials that meet dielectric criteria
        mat_1_e_total_idx = []
        mat_2_e_total_idx = []

        mat_1_e_ionic_idx = []
        mat_2_e_ionic_idx = []

        mat_1_e_elec_idx = []
        mat_2_e_elec_idx = []

        mat_1_n_idx = []
        mat_2_n_idx = []

        if "dielectric" in self.property_docs:
            for i, e_total in enumerate(consolidated_dict["e_total"]):
                if (e_total >= mat_1_e_total_lower_bound) and (e_total <= mat_1_e_total_upper_bound):       
                    mat_1_e_total_idx.append(i)
                if (e_total >= mat_2_e_total_lower_bound) and (e_total <= mat_2_e_total_upper_bound):       
                    mat_2_e_total_idx.append(i)

            for i, e_ionic in enumerate(consolidated_dict["e_ionic"]):
                if (e_ionic >= mat_1_e_ionic_lower_bound) and (e_ionic <= mat_1_e_ionic_upper_bound):       
                    mat_1_e_ionic_idx.append(i)
                if (e_ionic >= mat_2_e_ionic_lower_bound) and (e_ionic <= mat_2_e_ionic_upper_bound):       
                    mat_2_e_ionic_idx.append(i)
            
            for i, e_elec in enumerate(consolidated_dict["e_electronic"]):
                if (e_elec >= mat_1_e_elec_lower_bound) and (e_ionic <= mat_1_e_elec_upper_bound):       
                    mat_1_e_elec_idx.append(i)
                if (e_elec >= mat_2_e_elec_lower_bound) and (e_ionic <= mat_2_e_elec_upper_bound):       
                    mat_2_e_elec_idx.append(i)

            for i, n in enumerate(consolidated_dict["n"]):
                if (n >= mat_1_n_lower_bound) and (n <= mat_1_n_upper_bound):       
                    mat_1_n_idx.append(i)
                if (n >= mat_2_n_lower_bound) and (n <= mat_2_n_upper_bound):       
                    mat_2_n_idx.append(i)
            
            mat1_matches = mat1_matches & set(mat_1_e_total_idx) & set(mat_1_e_ionic_idx) & set(mat_1_e_elec_idx) & set(mat_1_n_idx)
            mat2_matches = mat2_matches & set(mat_2_e_total_idx) & set(mat_2_e_ionic_idx) & set(mat_2_e_elec_idx) & set(mat_2_n_idx)

        # Check for materials that meet elastic criteria
        mat_1_bulk_idx = []
        mat_2_bulk_idx = []

        mat_1_shear_idx = []
        mat_2_shear_idx = []

        mat_1_univ_aniso_idx = []
        mat_2_univ_aniso_idx = []

        if "elastic" in self.property_docs:
        
            for i, bulk_mod in enumerate(consolidated_dict["bulk_modulus"]):
                if (bulk_mod >= mat_1_bulk_mod_lower_bound) and (bulk_mod <= mat_1_bulk_mod_upper_bound):       
                    mat_1_bulk_idx.append(i)
                if (bulk_mod >= mat_2_bulk_mod_lower_bound) and (bulk_mod <= mat_2_bulk_mod_upper_bound):       
                    mat_2_bulk_idx.append(i)
            
            for i, shear_mod in enumerate(consolidated_dict["shear_modulus"]):
                if (shear_mod >= mat_1_shear_mod_lower_bound) and (shear_mod <= mat_1_shear_mod_upper_bound): 
                    mat_1_shear_idx.append(i)
                if (shear_mod >= mat_2_shear_mod_lower_bound) and (shear_mod <= mat_2_shear_mod_upper_bound): 
                    mat_2_shear_idx.append(i) 

            for i, univ_aniso in enumerate(consolidated_dict["universal_anisotropy"]):
                if (univ_aniso >= mat_1_univ_aniso_lower_bound) and (univ_aniso <= mat_1_univ_aniso_upper_bound): 
                    mat_1_univ_aniso_idx.append(i)
                if (univ_aniso >= mat_2_univ_aniso_lower_bound) and (univ_aniso <= mat_2_univ_aniso_upper_bound): 
                    mat_2_univ_aniso_idx.append(i) 

            mat1_matches = mat1_matches & set(mat_1_bulk_idx) & set(mat_1_shear_idx) & set(mat_1_univ_aniso_idx)
            mat2_matches = mat2_matches & set(mat_2_bulk_idx) & set(mat_2_shear_idx) & set(mat_2_univ_aniso_idx)

        # Check for materials that meet magnetic criteria
        mat_1_tot_mag_idx = []
        mat_2_tot_mag_idx = []

        mat_1_tot_mag_norm_idx = []
        mat_2_tot_mag_norm_idx = []

        if "magnetic" in self.property_docs:
            for i, tot_mag in enumerate(consolidated_dict["total_magnetization"]):
                if (tot_mag >= mat_1_tot_mag_lower_bound) and (tot_mag <= mat_1_tot_mag_upper_bound):       
                    mat_1_tot_mag_idx.append(i)
                if (tot_mag >= mat_2_tot_mag_lower_bound) and (tot_mag <= mat_2_tot_mag_upper_bound):       
                    mat_2_tot_mag_idx.append(i)
            
            for i, tot_mag_norm in enumerate(consolidated_dict["total_magnetization_normalized_vol"]):
                if (tot_mag_norm >= mat_1_tot_mag_norm_lower_bound) and (tot_mag_norm <= mat_1_tot_mag_norm_upper_bound):       
                    mat_1_tot_mag_norm_idx.append(i)
                if (tot_mag_norm >= mat_2_tot_mag_norm_lower_bound) and (tot_mag_norm <= mat_2_tot_mag_norm_upper_bound):       
                    mat_2_tot_mag_norm_idx.append(i)

            mat1_matches = mat1_matches & set(mat_1_tot_mag_idx) & set(mat_1_tot_mag_norm_idx)
            mat2_matches = mat2_matches & set(mat_2_tot_mag_idx) & set(mat_2_tot_mag_norm_idx)

        # Check for materials that meet piezoelectric criteria
        mat_1_e_ij_idx = []
        mat_2_e_ij_idx = []

        if "piezoelectric" in self.property_docs:
            for i, e_ij in enumerate(consolidated_dict["total_magnetization"]):
                if (e_ij >= mat_1_e_ij_lower_bound) and (e_ij <= mat_1_e_ij_upper_bound):       
                    mat_1_e_ij_idx.append(i)
                if (e_ij >= mat_2_e_ij_lower_bound) and (e_ij <= mat_2_e_ij_upper_bound):       
                    mat_2_e_ij_idx.append(i)
            
            mat1_matches = mat1_matches & set(mat_1_e_ij_idx)
            mat2_matches = mat2_matches & set(mat_2_e_ij_idx)

        # Extract mp-ids
        # mat_1_matches is a set of indices, extract the corresponding mp-ids
        
        mat_1_ids = [consolidated_dict["material_id"][i] for i in mat1_matches]
        mat_2_ids = [consolidated_dict["material_id"][i] for i in mat2_matches]

        return mat_1_ids, mat_2_ids 
    
    def get_material_match_costs(self, mat_1_ids, mat_2_ids, consolidated_dict: dict = {}):

        if consolidated_dict == {}:
            with open("test_final_dict") as f: # TODO change to get most recent consolidated dict
                consolidated_dict = json.load(f)

        for m1 in mat_1_ids:
            for m2 in mat_2_ids:

                m1_idx = mat_1_ids.index(m1)
                m2_idx = mat_2_ids.index(m2)
                
                material_values = []
                if "carrier-transport" in self.property_docs:
                    material_values.append(consolidated_dict["elec_cond_300K_low_doping"][m1_idx])
                    material_values.append(consolidated_dict["therm_cond_300K_low_doping"][m1_idx])
                    material_values.append(consolidated_dict["elec_cond_300K_low_doping"][m2_idx])
                    material_values.append(consolidated_dict["therm_cond_300K_low_doping"][m2_idx])
                if "dielectric" in self.property_docs:
                    material_values.append(consolidated_dict["e_total"][m1_idx])
                    material_values.append(consolidated_dict["e_ionic"][m1_idx])
                    material_values.append(consolidated_dict["e_electronic"][m1_idx])
                    material_values.append(consolidated_dict["n"][m1_idx])
                    material_values.append(consolidated_dict["e_total"][m2_idx])
                    material_values.append(consolidated_dict["e_ionic"][m2_idx])
                    material_values.append(consolidated_dict["e_electronic"][m2_idx])
                    material_values.append(consolidated_dict["n"][m2_idx])
                if "elastic" in self.property_docs:
                    material_values.append(consolidated_dict["bulk_modulus"][m1_idx])
                    material_values.append(consolidated_dict["shear_modulus"][m1_idx])
                    material_values.append(consolidated_dict["universal_anisotropy"][m1_idx])
                    material_values.append(consolidated_dict["bulk_modulus"][m2_idx])
                    material_values.append(consolidated_dict["shear_modulus"][m2_idx])
                    material_values.append(consolidated_dict["universal_anisotropy"][m2_idx])
                if "magnetic" in self.property_docs:
                    material_values.append(consolidated_dict["total_magnetization"][m1_idx])
                    material_values.append(consolidated_dict["total_magnetization_normalized_volume"][m1_idx])
                    material_values.append(consolidated_dict["total_magnetization"][m2_idx])
                    material_values.append(consolidated_dict["total_magnetization_normalized_volume"][m2_idx])
                if "piezoelectric" in self.property_docs:
                    material_values.append(consolidated_dict["e_ij"][m1_idx])
                    material_values.append(consolidated_dict["e_ij"][m2_idx])

                # Create population of same properties for all members based on material match pair
                values = np.reshape(material_values*self.ga_params.get_num_members(), (self.ga_params.get_num_members(), len(material_values))) 
                population = np.reshape(values, (self.ga_params.get_num_members(), len(material_values)))

                # Only the vary the mixing parameter and volume fraction across the population
                mixing_param = np.random.rand(self.ga_params.get_num_members(), 1)
                phase1_vol_frac = np.random.rand(self.ga_params.get_num_members(), 1)

                # Include the random mixing parameters and volume fractions in the population
                values = np.c_[population, mixing_param, phase1_vol_frac]    

                # Instantiate the population and find the best performers
                population = Population(num_properties=self.num_properties, values=values, property_docs=self.property_docs, desired_props=self.desired_props, ga_params=self.ga_params)
                population.set_costs()
                [sorted_costs, sorted_indices] = population.sort_costs()
                population.set_order_by_costs(sorted_indices)
                sorted_costs = np.reshape(sorted_costs, (len(sorted_costs), 1))

                # Assemble a table for printing
                mat1_id = np.reshape([m1]*self.ga_params.get_num_members(), (self.ga_params.get_num_members(),1))
                mat2_id = np.reshape([m2]*self.ga_params.get_num_members(), (self.ga_params.get_num_members(),1))
                table_data = np.c_[mat1_id, mat2_id, population.values, sorted_costs] 
                print("\nMATERIALS PROJECT PAIRS AND HASHIN-SHTRIKMAN RECOMMENDED VOLUME FRACTION")
                print(tabulate(table_data[0:5, :], headers=self.get_headers(include_mpids=True))) # hardcoded to be 5 rows, could change
    
    #------ Setter Methods ------#

    def set_lower_bounds(self, lower_bounds):
        self.lower_bounds = lower_bounds
        return self

    def set_lower_bounds_from_user_input(self):

        self.lower_bounds = {"mat1": {}, "mat2": {}}

        # Carrier transport
        if "carrier-transport" in self.property_docs:
            self.lower_bounds["mat1"]["carrier-transport"] = []
            self.lower_bounds["mat1"]["carrier-transport"].append(self.user_input.mat1_lower_elec_cond_300k_low_doping)
            self.lower_bounds["mat1"]["carrier-transport"].append(self.user_input.mat1_lower_therm_cond_300k_low_doping)
            self.lower_bounds["mat2"]["carrier-transport"] = []
            self.lower_bounds["mat2"]["carrier-transport"].append(self.user_input.mat2_lower_elec_cond_300k_low_doping)
            self.lower_bounds["mat2"]["carrier-transport"].append(self.user_input.mat2_lower_therm_cond_300k_low_doping)

        # Dielectric
        if "dielectric" in self.property_docs:
            self.lower_bounds["mat1"]["dielectric"] = []
            self.lower_bounds["mat1"]["dielectric"].append(self.user_input.mat1_lower_e_total)
            self.lower_bounds["mat1"]["dielectric"].append(self.user_input.mat1_lower_e_ionic)
            self.lower_bounds["mat1"]["dielectric"].append(self.user_input.mat1_lower_e_electronic)
            self.lower_bounds["mat1"]["dielectric"].append(self.user_input.mat1_lower_n)
            self.lower_bounds["mat2"]["dielectric"] = []
            self.lower_bounds["mat2"]["dielectric"].append(self.user_input.mat2_lower_e_total)
            self.lower_bounds["mat2"]["dielectric"].append(self.user_input.mat2_lower_e_ionic)
            self.lower_bounds["mat2"]["dielectric"].append(self.user_input.mat2_lower_e_electronic)
            self.lower_bounds["mat2"]["dielectric"].append(self.user_input.mat2_lower_n)

        # Elastic
        if "elastic" in self.property_docs:
            self.lower_bounds["mat1"]["elastic"] = []
            self.lower_bounds["mat1"]["elastic"].append(self.user_input.mat1_lower_bulk_modulus)
            self.lower_bounds["mat1"]["elastic"].append(self.user_input.mat1_lower_shear_modulus)
            self.lower_bounds["mat1"]["elastic"].append(self.user_input.mat1_lower_universal_anisotropy)
            self.lower_bounds["mat2"]["elastic"] = []
            self.lower_bounds["mat2"]["elastic"].append(self.user_input.mat2_lower_bulk_modulus)
            self.lower_bounds["mat2"]["elastic"].append(self.user_input.mat2_lower_shear_modulus)
            self.lower_bounds["mat2"]["elastic"].append(self.user_input.mat2_lower_universal_anisotropy)

        # Magnetic
        if "magnetic" in self.property_docs:
            self.lower_bounds["mat1"]["magnetic"] = []
            self.lower_bounds["mat1"]["magnetic"].append(self.user_input.mat1_lower_total_magnetization)
            self.lower_bounds["mat1"]["magnetic"].append(self.user_input.mat1_lower_total_magnetization_normalized_volume)
            self.lower_bounds["mat2"]["magnetic"] = []
            self.lower_bounds["mat2"]["magnetic"].append(self.user_input.mat2_lower_total_magnetization)
            self.lower_bounds["mat2"]["magnetic"].append(self.user_input.mat2_lower_total_magnetization_normalized_volume)

        # Piezoelectric
        if "piezoelectric" in self.property_docs:
            self.lower_bounds["mat1"]["piezoelectric"] = []
            self.lower_bounds["mat1"]["piezoelectric"].append(self.user_input.mat1_lower_e_ij)
            self.lower_bounds["mat2"]["piezoelectric"] = []
            self.lower_bounds["mat2"]["piezoelectric"].append(self.user_input.mat2_lower_e_ij)

        return self
    
    def set_upper_bounds(self, upper_bounds):
        self.upper_bounds = upper_bounds
        return self
    
    def set_upper_bounds_from_user_input(self):

        self.upper_bounds = {"mat1": {}, "mat2": {}}

        # Carrier transport
        if "carrier-transport" in self.property_docs:
            self.upper_bounds["mat1"]["carrier-transport"] = []
            self.upper_bounds["mat1"]["carrier-transport"].append(self.user_input.mat1_upper_elec_cond_300k_low_doping)
            self.upper_bounds["mat1"]["carrier-transport"].append(self.user_input.mat1_upper_therm_cond_300k_low_doping)
            self.upper_bounds["mat2"]["carrier-transport"] = []
            self.upper_bounds["mat2"]["carrier-transport"].append(self.user_input.mat2_upper_elec_cond_300k_low_doping)
            self.upper_bounds["mat2"]["carrier-transport"].append(self.user_input.mat2_upper_therm_cond_300k_low_doping)

        # Dielectric
        if "dielectric" in self.property_docs:
            self.upper_bounds["mat1"]["dielectric"] = []
            self.upper_bounds["mat1"]["dielectric"].append(self.user_input.mat1_upper_e_total)
            self.upper_bounds["mat1"]["dielectric"].append(self.user_input.mat1_upper_e_ionic)
            self.upper_bounds["mat1"]["dielectric"].append(self.user_input.mat1_upper_e_electronic)
            self.upper_bounds["mat1"]["dielectric"].append(self.user_input.mat1_upper_n)
            self.upper_bounds["mat2"]["dielectric"] = []
            self.upper_bounds["mat2"]["dielectric"].append(self.user_input.mat2_upper_e_total)
            self.upper_bounds["mat2"]["dielectric"].append(self.user_input.mat2_upper_e_ionic)
            self.upper_bounds["mat2"]["dielectric"].append(self.user_input.mat2_upper_e_electronic)
            self.upper_bounds["mat2"]["dielectric"].append(self.user_input.mat2_upper_n)

        # Elastic
        if "elastic" in self.property_docs:
            self.upper_bounds["mat1"]["elastic"] = []
            self.upper_bounds["mat1"]["elastic"].append(self.user_input.mat1_upper_bulk_modulus)
            self.upper_bounds["mat1"]["elastic"].append(self.user_input.mat1_upper_shear_modulus)
            self.upper_bounds["mat1"]["elastic"].append(self.user_input.mat1_upper_universal_anisotropy)
            self.upper_bounds["mat2"]["elastic"] = []
            self.upper_bounds["mat2"]["elastic"].append(self.user_input.mat2_upper_bulk_modulus)
            self.upper_bounds["mat2"]["elastic"].append(self.user_input.mat2_upper_shear_modulus)
            self.upper_bounds["mat2"]["elastic"].append(self.user_input.mat2_upper_universal_anisotropy)

        # Magnetic
        if "magnetic" in self.property_docs:
            self.upper_bounds["mat1"]["magnetic"] = []
            self.upper_bounds["mat1"]["magnetic"].append(self.user_input.mat1_upper_total_magnetization)
            self.upper_bounds["mat1"]["magnetic"].append(self.user_input.mat1_upper_total_magnetization_normalized_volume)
            self.upper_bounds["mat2"]["magnetic"] = []
            self.upper_bounds["mat2"]["magnetic"].append(self.user_input.mat2_upper_total_magnetization)
            self.upper_bounds["mat2"]["magnetic"].append(self.user_input.mat2_upper_total_magnetization_normalized_volume)

        # Piezoelectric
        if "piezoelectric" in self.property_docs:
            self.upper_bounds["mat1"]["piezoelectric"] = []
            self.upper_bounds["mat1"]["piezoelectric"].append(self.user_input.mat1_upper_e_ij)
            self.upper_bounds["mat2"]["piezoelectric"] = []
            self.upper_bounds["mat2"]["piezoelectric"].append(self.user_input.mat2_upper_e_ij)

        return self

    def set_retain_parents(self, retain_parents):
        self.retain_parents = retain_parents
        return self
    
    def set_allow_mutations(self, allow_mutations):
        self.allow_mutations = allow_mutations
        return self

    def set_property_docs(self, property_docs):
        self.property_docs = property_docs
        return self
    
    def set_desired_props(self, desired_props):
        self.desired_props = desired_props
        return self
    
    def set_desired_props_from_user_input(self):

        self.desired_props = {}

        # Carrier transport
        if "carrier-transport" in self.property_docs:
            self.desired_props["carrier-transport"] = []
            self.desired_props["carrier-transport"].append(self.user_input.desired_elec_cond_300k_low_doping)
            self.desired_props["carrier-transport"].append(self.user_input.desired_therm_cond_300k_low_doping)

        # Dielectric
        if "dielectric" in self.property_docs:
            self.desired_props["dielectric"] = []
            self.desired_props["dielectric"].append(self.user_input.desired_e_total)
            self.desired_props["dielectric"].append(self.user_input.desired_e_ionic)
            self.desired_props["dielectric"].append(self.user_input.desired_e_electronic)
            self.desired_props["dielectric"].append(self.user_input.desired_n)

        # Elastic
        if "elastic" in self.property_docs:
            self.desired_props["elastic"] = []
            self.desired_props["elastic"].append(self.user_input.desired_bulk_modulus)
            self.desired_props["elastic"].append(self.user_input.desired_shear_modulus)
            self.desired_props["elastic"].append(self.user_input.desired_universal_anisotropy)

        # Magnetic
        if "magnetic" in self.property_docs:
            self.desired_props["magnetic"] = []
            self.desired_props["magnetic"].append(self.user_input.desired_total_magnetization)
            self.desired_props["magnetic"].append(self.user_input.desired_total_magnetization_normalized_volume)

        # Piezoelectric
        if "piezoelectric" in self.property_docs:
            self.desired_props["piezoelectric"] = []
            self.desired_props["piezoelectric"].append(self.user_input.desired_e_ij)

        return self
    
    def set_has_props(self, has_props):
        self.has_props = has_props
        return self
    
    def set_fields(self, fields):
        self.fields = fields
        return self
    
    def set_num_properties(self, num_properties):
        self.num_properties = num_properties
        return num_properties
    
    def set_num_properties_from_desired_props(self):

        num_properties = 0
        
        # Add variables to members for each property doc
        if "carrier-transport" in self.property_docs:
            num_properties += len(self.desired_props["carrier-transport"])
        if "dielectric" in self.property_docs:
            num_properties += len(self.desired_props["dielectric"])
        if "elastic" in self.property_docs:
            num_properties += len(self.desired_props["elastic"])
        if "magnetic" in self.property_docs:
            num_properties += len(self.desired_props["magnetic"])
        if "piezoelectric" in self.property_docs:
            num_properties += len(self.desired_props["piezoelectric"])
        
        # Make sure each material in the composite has variables for each property doc
        # (Multiply by the number of materials in the composite)
        num_materials = len(self.lower_bounds)
        num_properties = num_properties * num_materials

        # Add variables for mixing parameter and volume fraction
        num_properties += 2

        self.num_properties = num_properties
        return self
    
    def set_ga_params(self, ga_params):
        self.ga_params = ga_params
        return self
    
    def set_final_population(self, final_pop):
        self.final_population = final_pop
        return self
    
    def set_cost_history(self, cost_history):
        self.cost_history = cost_history
        return self
    
    def set_lowest_costs(self, lowest_costs):
        self.lowest_costs = lowest_costs
        return self           

    def set_avg_parent_costs(self, avg_parent_costs):
        self.avg_parent_costs = avg_parent_costs
        return self
    
    def set_HS_optim_params(self):
        
        # MAIN OPTIMIZATION FUNCTION

        # Unpack necessary attributes from self
        num_parents = self.ga_params.get_num_parents()
        num_kids = self.ga_params.get_num_kids()
        num_generations = self.ga_params.get_num_generations()
        num_members = self.ga_params.get_num_members()
        
        # Initialize arrays to store the cost and original indices of each generation
        all_costs = np.ones((num_generations, num_members))
        
        # Initialize arrays to store best performer and parent avg 
        lowest_costs = np.zeros(num_generations)     # best cost
        avg_parent_costs = np.zeros(num_generations) # avg cost of parents
        
        # Generation counter
        g = 0

        # Initialize array to store costs for current generation
        costs = np.zeros(num_members)

        # Randomly populate first generation  
        population = Population(num_properties=self.num_properties, property_docs=self.property_docs, desired_props=self.desired_props, ga_params=self.ga_params)
        population.set_initial_random(self.lower_bounds, self.upper_bounds)

        # Calculate the costs of the first generation
        population.set_costs()    

        # Sort the costs of the first generation
        [sorted_costs, sorted_indices] = population.sort_costs()  
        all_costs[g, :] = sorted_costs.reshape(1, num_members) 

        # Store the cost of the best performer and average cost of the parents 
        lowest_costs[g] = np.min(sorted_costs)
        avg_parent_costs[g] = np.mean(sorted_costs[0:num_parents])
        
        # Update population based on sorted indices
        population.set_order_by_costs(sorted_indices)
        
        # Perform all later generations    
        while g < num_generations:

            print(f"Generation {g} of {num_generations}")
            costs[0:num_parents] = sorted_costs[0:num_parents] # retain the parents from the previous generation
            
            # Select top parents from population to be breeders
            for p in range(0, num_parents, 2):
                phi1, phi2 = np.random.rand(2)
                kid1 = phi1 * population.values[p, :] + (1-phi1) * population.values[p+1, :]
                kid2 = phi2 * population.values[p, :] + (1-phi2) * population.values[p+1, :]
                
                # Append offspring to population, overwriting old population members 
                population.values[num_parents+p,   :] = kid1
                population.values[num_parents+p+1, :] = kid2
            
                # Cast offspring to members and evaluate costs
                kid1 = Member(num_properties=self.num_properties, values=kid1, property_docs=self.property_docs, desired_props=self.desired_props, ga_params=self.ga_params)
                kid2 = Member(num_properties=self.num_properties, values=kid2, property_docs=self.property_docs, desired_props=self.desired_props, ga_params=self.ga_params)
                costs[num_parents+p]   = kid1.get_cost()
                costs[num_parents+p+1] = kid2.get_cost()
                        
            # Randomly generate new members to fill the rest of the population
            members_minus_parents_minus_kids = num_members - num_parents - num_kids
            population.set_new_random(members_minus_parents_minus_kids, self.lower_bounds, self.upper_bounds)

            # Calculate the costs of the gth generation
            population.set_costs()

            # Sort the costs for the gth generation
            [sorted_costs, sorted_indices] = population.sort_costs()  
            all_costs[g, :] = sorted_costs.reshape(1, num_members) 
        
            # Store the cost of the best performer and average cost of the parents 
            lowest_costs[g] = np.min(sorted_costs)
            avg_parent_costs[g] = np.mean(sorted_costs[0:num_parents])
        
            # Update population based on sorted indices
            population.set_order_by_costs(sorted_indices)

            # Update the generation counter
            g = g + 1 

        # Update self attributes following optimization
        self.final_population = population
        self.cost_history = all_costs
        self.lowest_costs = lowest_costs
        self.avg_parent_costs = avg_parent_costs     
        
        return self         

    #------ Other Methods ------#

    def print_table_of_best_designs(self):

        table_data = self.get_table_of_best_designs()
        print("\nHASHIN-SHTRIKMAN + GENETIC ALGORITHM RECOMMENDED MATERIAL PROPERTIES")
        print(tabulate(table_data, headers=self.get_headers()))
    
    def plot_optimization_results(self):
        fig, ax = plt.subplots(figsize=(10,6))
        ax.plot(range(self.ga_params.get_num_generations()), self.avg_parent_costs, label="Avg. of top 10 performers")
        ax.plot(range(self.ga_params.get_num_generations()), self.lowest_costs, label="Best costs")
        plt.xlabel("Generation", fontsize= 20)
        plt.ylabel("Cost", fontsize=20)
        plt.title("Genetic Algorithm Results", fontsize = 24)
        plt.legend(fontsize = 14)
        plt.show()  

    def generate_material_property_dict(self):

        material_property_dict = {"mp-ids": [],
                                  "mp-ids-contrib": [], 
                                  "formula": [],
                                  "metal": []} 

        if "carrier-transport" in self.property_docs:
            material_property_dict["elec_cond_300K_low_doping"] = []
            material_property_dict["therm_cond_300K_low_doping"] = []

        if "dielectric" in self.property_docs:
            material_property_dict["e_total"] = []
            material_property_dict["e_ionic"] = []
            material_property_dict["e_electronic"] = []
            material_property_dict["n"] = []

        if "elastic" in self.property_docs:
            material_property_dict["bulk_modulus"] = []
            material_property_dict["shear_modulus"] = []
            material_property_dict["universal_anisotropy"] = []

        if "magnetic" in self.property_docs:
            material_property_dict["total_magnetization"] = []
            material_property_dict["total_magnetization_normalized_volume"] = []
            
        if "piezoelectric" in self.property_docs:
            material_property_dict["e_ij"] = []

        return material_property_dict
    
    def generate_consolidated_dict(self, total_docs = None):

        # MAIN FUNCTION USED TO GENERATE MATRIAL PROPERTY DICTIONARY DEPENDING ON USER REQUEST

        if "carrier-transport" in self.property_docs:
            client = Client(apikey=self.api_key, project=self.mp_contribs_project)
        else:
            client = Client(apikey=self.api_key)
        
        new_fields = self.fields

        # MP-contribs data
        if "carrier-transport" in self.property_docs:
           new_fields["mp-ids-contrib"] = []
           new_fields["elec_cond_300K_low_doping"] = []
           new_fields["therm_cond_300K_low_doping"] = []

        # Dielectric
        if "dielectric" in self.property_docs:
            new_fields["e_electronic"] = []
            new_fields["e_ionic"] = []
            new_fields["e_total"] = []
            new_fields["n"] = []

        # Elastic
        if "elastic" in self.property_docs:
            new_fields["bulk_modulus"] = []
            new_fields["shear_modulus"] = []
            new_fields["universal_anisotropy"] = []
        
        # Magnetic
        if "magnetic" in self.property_docs:
            new_fields["total_magnetization"] = []
            new_fields["total_magnetization_normalized_vol"] = []

        # Piezoelectric
        if "piezoelectric" in self.property_docs:
            new_fields["e_ij_max"] = []

        self.set_fields(new_fields)

        logger.info(f"self.fields: {self.fields}")

        with MPRester(self.api_key) as mpr:
            
            docs = mpr.materials.summary.search(fields=self.fields)
            from mpi4py import MPI

            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()

            # Calculate the size of each chunk
            if total_docs is None:
                total_docs = len(docs)
                chunk_size = len(docs) // size
            elif isinstance(total_docs, int):
                chunk_size = total_docs // size

            # Calculate the start and end indices for this process"s chunk
            start = rank * chunk_size
            end = start + chunk_size if rank != size - 1 else total_docs  # The last process gets the remainder

            # Each process gets a different chunk
            chunk = docs[start:end]

            # for i, doc in enumerate(docs):
            for i, doc in enumerate(chunk[0:total_docs]):

                logger.info(f"Process {rank}: {i} of {len(chunk[0:total_docs])}")

                required_fields = [doc.material_id, doc.is_stable, doc.is_metal]

                if "dielectric" in self.property_docs:
                    required_fields.append(doc.e_electronic)
                    required_fields.append(doc.e_ionic)
                    required_fields.append(doc.e_total)
                    required_fields.append(doc.n)
                if "magnetic" in self.property_docs:
                    required_fields.append(doc.total_magnetization)
                    required_fields.append(doc.total_magnetization_normalized_vol)
                if "piezoelectric" in self.property_docs:
                    required_fields.append(doc.e_ij_max)
                if "elastic" in self.property_docs:
                    required_fields.append(doc.bulk_modulus)
                    required_fields.append(doc.shear_modulus)
                    required_fields.append(doc.universal_anisotropy)
            
                if "carrier-transport" in self.property_docs:
                    try:
                        mp_id = doc.material_id                           
                        query = {"identifier": mp_id}
                        my_dict = client.download_contributions(query=query, include=["tables"])[0]
                        required_fields.append(my_dict["identifier"])
                    except IndexError:
                        continue

                if all(field is not None for field in required_fields):
                    self.fields["material_id"].append(mp_id)
                    self.fields["formula"].append(my_dict["formula"])
                    self.fields["is_stable"].append(doc.is_stable)
                    self.fields["is_metal"].append(doc.is_metal)
                    self.fields["band_gap"].append(doc.band_gap)
                    
                    # Carrier transport
                    if "carrier-transport" in self.property_docs:
                        self.fields["mp-ids-contrib"].append(my_dict["identifier"])
                        thermal_cond_str = my_dict["tables"][7].iloc[2, 0].replace(",", "")

                        if "×10" in thermal_cond_str:
                            # Extract the numeric part before the "±" symbol and the exponent
                            thermal_cond_str, thermal_cond_exponent_str = re.search(r"\((.*?) ±.*?\)×10(.*)", thermal_cond_str).groups()
                            # Convert the exponent part to a format that Python can understand
                            thermal_cond_exponent = self.superscript_to_int(thermal_cond_exponent_str.strip())
                            # Combine the numeric part and the exponent part, and convert the result to a float
                            thermal_cond = float(f"{thermal_cond_str}e{thermal_cond_exponent}") * 1e-14  # multply by relaxation time, 10 fs
                            logger.info(f"thermal_cond_if_statement = {thermal_cond}")
                        else:
                            thermal_cond = float(thermal_cond_str) * 1e-14  # multply by relaxation time, 10 fs
                            logger.info(f"thermal_cond_else_statement = {thermal_cond}")

                        elec_cond_str = my_dict["tables"][5].iloc[2, 0].replace(",", "")

                        if "×10" in elec_cond_str:
                            # Extract the numeric part before the "±" symbol and the exponent
                            elec_cond_str, elec_cond_exponent_str = re.search(r"\((.*?) ±.*?\)×10(.*)", elec_cond_str).groups()
                            # Convert the exponent part to a format that Python can understand
                            elec_cond_exponent = self.superscript_to_int(elec_cond_exponent_str.strip())
                            # Combine the numeric part and the exponent part, and convert the result to a float
                            elec_cond = float(f"{elec_cond_str}e{elec_cond_exponent}") * 1e-14  # multply by relaxation time, 10 fs
                            logger.info(f"elec_cond_if_statement = {elec_cond}")
                        else:
                            elec_cond = float(elec_cond_str) * 1e-14  # multply by relaxation time, 10 fs
                            logger.info(f"elec_cond_else_statement = {elec_cond}")


                        self.fields["therm_cond_300K_low_doping"].append(thermal_cond)
                        self.fields["elec_cond_300K_low_doping"].append(elec_cond)   
                    
                    # Dielectric
                    if "dielectric" in self.property_docs:
                        self.fields["e_electronic"].append(doc.e_electronic)
                        self.fields["e_ionic"].append(doc.e_ionic)
                        self.fields["e_total"].append(doc.e_total)
                        self.fields["n"].append(doc.n)
                    
                    # Elastic
                    if "elastic" in self.property_docs:
                        self.fields["bulk_modulus"].append(doc.bulk_modulus["voigt"])
                        self.fields["shear_modulus"].append(doc.shear_modulus["voigt"])
                        self.fields["universal_anisotropy"].append(doc.universal_anisotropy)
                        logger.info(f"bulk_modulus = {doc.bulk_modulus['voigt']}")
                
                    # Magnetic
                    if "magnetic" in self.property_docs:
                        self.fields["total_magnetization"].append(doc.total_magnetization)
                        self.fields["total_magnetization_normalized_vol"].append(doc.total_magnetization_normalized_vol)
                    
                    # Piezoelectric
                    if "piezoelectric" in self.property_docs:
                        self.fields["e_ij_max"].append(doc.e_ij_max)

        
            # comm.gather the self.fields data after the for loop
            gathered_fields = comm.gather(self.fields, root=0)

            # On process 0, consolidate self.fields data into a single dictionary -- consolidated_dict
            if rank == 0:
                consolidated_dict: Dict[str, List[Any]] = {}
                if gathered_fields is not None:
                    for fields in gathered_fields:
                        for key, value in fields.items():
                            if key in consolidated_dict:
                                consolidated_dict[key].extend(value)
                            else:
                                consolidated_dict[key] = value

                # Save the consolidated results to a JSON file
                now = datetime.now()
                my_file_name = "consolidated_dict_" + now.strftime("%m_%d_%Y_%H_%M_%S")
                with open(my_file_name, "w") as my_file:
                    json.dump(consolidated_dict, my_file)

        return consolidated_dict
    
    def superscript_to_int(self, superscript_str):
        superscript_to_normal = {
            "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
            "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9"
        }
        normal_str = "".join(superscript_to_normal.get(char, char) for char in superscript_str)
        return int(normal_str)
    
    

        
    

    