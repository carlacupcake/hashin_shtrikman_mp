# From MPRester
import itertools
import warnings
from functools import lru_cache
from json import loads
from os import environ
from typing import Dict, List, Literal, Optional, Union

from emmet.core.mpid import MPID
from emmet.core.settings import EmmetSettings
from emmet.core.summary import HasProps
from packaging import version
from requests import Session, get

from mp_api.client.core.settings import MAPIClientSettings
from mp_api.client.core.utils import validate_ids

# Custom Classes
from ga_params_class import GAParams
from genetic_string_class import GeneticString
from population_class import Population

# Other
import numpy as np
import matplotlib.pyplot as plt
import json
import datetime
from mp_api.client import MPRester
from mpcontribs.client import Client
from tabulate import tabulate

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
DEFAULT_LOWER_BOUNDS  = [0] * 26
DEFAULT_UPPER_BOUNDS  = [np.inf] * 24 + [1] * 2
DEFAULT_PROPERTY_DOCS = ["carrier-transport", "dielectric", "elastic", "magnetic", "piezoelectric"]
DEFAULT_DESIRED_PROPS = [0] * 12
DEFAULT_HAS_PROPS     = [HasProps.dielectric, HasProps.elasticity]
DEFAULT_FIELDS        = ["material_id", "is_stable", "band_gap", "is_metal"]

class HashinShtrikman:

    #------ Initialization method ------#
    def __init__(
            self,
            api_key:              Optional[str] = None,
            endpoint:             str  = DEFAULT_ENDPOINT,
            lower_bounds:         list = DEFAULT_LOWER_BOUNDS,
            upper_bounds:         list = DEFAULT_UPPER_BOUNDS, 
            property_docs:        list = DEFAULT_PROPERTY_DOCS,
            desired_props:        list = DEFAULT_DESIRED_PROPS,
            has_props:            list = DEFAULT_HAS_PROPS,
            fields:               list = DEFAULT_FIELDS,
            dv:                   int  = 26,
            ga_params:            GAParams = GAParams(),
            final_population:     Population = Population(),
            cost_history:         np.ndarray = np.empty,   
            lowest_costs:         np.ndarray = np.empty,          
            parent_average_costs: np.ndarray = np.empty, 
        ):
            
            self.api_key              = api_key 
            self.endpoint             = endpoint
            self.lower_bounds         = lower_bounds 
            self.upper_bounds         = upper_bounds
            self.property_docs        = property_docs
            self.desired_props        = desired_props 
            self.has_props            = has_props
            self.fields               = fields
            self.dv                   = dv   # dimension of a gentic strng, initialize with 14 to account for g and v1
            self.ga_params            = ga_params 
            self.final_population     = final_population
            self.cost_history         = cost_history            
            self.lowest_costs         = lowest_costs
            self.parent_average_costs = parent_average_costs

            try:
                from mpcontribs.client import Client
                self.contribs = Client(api_key, project="carrier_transport")
            except ImportError:
                self.contribs = None
                warnings.warn(
                    "mpcontribs-client not installed. "
                    "Install the package to query MPContribs data:"
                    "'pip install mpcontribs-client'"
                )
            except Exception as error:
                self.contribs = None
                warnings.warn(f"Problem loading MPContribs client: {error}")

            # # Check if emmet version of server os compatible
            # emmet_version = version.parse(self.get_emmet_version())

            # if version.parse(emmet_version.base_version) < version.parse(
            #     _MAPI_SETTINGS.MIN_EMMET_VERSION
            # ):
            #     warnings.warn(
            #         "The installed version of the mp-api client may not be compatible with the API server. "
            #         "Please install a previous version if any problems occur."
            #     )

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
    
    def get_dv(self):
        return self.dv
    
    def get_ga_params(self):
        return self.ga_params
    
    def get_final_population(self):
        return self.final_population
    
    def get_cost_history(self):
        return self.cost_history
    
    def get_lowest_costs(self):
        return self.lowest_costs           

    def get_parent_average_costs(self):
        return self.parent_average_costs
    
    def get_unique_designs(self):
        
        # Costs are often equal to >10 decimal points, truncate to obtain a richer set of suggestions
        self.final_population.set_costs()
        final_costs = self.final_population.get_costs()
        rounded_costs = np.round(final_costs, decimals=3)
    
        # Obtain Unique Strings and Costs
        [unique_costs, iuniq] = np.unique(rounded_costs, return_index=True)
        unique_strings = self.final_population.values[iuniq]

        return [unique_strings, unique_costs] 

    def get_table_of_best_designs(self):
        [unique_strings, unique_costs] = self.get_unique_designs()     
        table_data = np.hstack((unique_strings[0:20,:], unique_costs[0:20].reshape(-1,1)))
        return table_data         
    
    def get_material_matches(self): # TODO UNDER CONSTRUCTION

        table_data = self.get_table_of_best_designs()
        final_dict = self.generate_final_dict()

        # (Material 1) Filter the dictionary extracted from MP-API based on GA results
        mat_1_dict = {"mp-ids": [],
                      "mp-ids-contrib": [], 
                      "formula": [],
                      "metal": [],
                      "elec_cond_300K_low_doping": [],
                      "therm_cond_300K_low_doping": [],
                      "e_total": [],
                      "e_ionic": [],
                      "e_electronic": [],
                      "n": [],
                      "bulk_modulus": [],
                      "shear_modulus": [],
                      "universal_anisotropy": [],
                      "total_magnetization": [],
                      "total_magnetization_normalized_volume": [],
                      "e_ij": [],
                      }
        
        # (Material 2) Filter the dictionary extracted from MP-API based on GA results
        mat_2_dict = {"mp-ids": [],
                      "mp-ids-contrib": [], 
                      "formula": [],
                      "metal": [],
                      "elec_cond_300K_low_doping": [],
                      "therm_cond_300K_low_doping": [],
                      "e_total": [],
                      "e_ionic": [],
                      "e_electronic": [],
                      "n": [],
                      "bulk_modulus": [],
                      "shear_modulus": [],
                      "universal_anisotropy": [],
                      "total_magnetization": [],
                      "total_magnetization_normalized_volume": [],
                      "e_ij": [],
                      }

        # All indices below based on genetic string class
        # Carrier transport extrema
        mat_1_elec_cond_lower_bound  = min(table_data[:,0])
        mat_1_elec_cond_upper_bound  = max(table_data[:,0])
        mat_2_elec_cond_lower_bound  = min(table_data[:,12])
        mat_2_elec_cond_upper_bound  = max(table_data[:,12])

        mat_1_therm_cond_lower_bound = min(table_data[:,1])
        mat_1_therm_cond_upper_bound = max(table_data[:,1])
        mat_2_therm_cond_lower_bound = min(table_data[:,13])
        mat_2_therm_cond_upper_bound = max(table_data[:,13])

        # Dielectric extrema
        mat_1_e_total_lower_bound = min(table_data[:,2])
        mat_1_e_total_upper_bound = max(table_data[:,2])
        mat_2_e_total_lower_bound = min(table_data[:,14])
        mat_2_e_total_upper_bound = max(table_data[:,14])

        mat_1_e_ionic_lower_bound = min(table_data[:,3])
        mat_1_e_ionic_upper_bound = max(table_data[:,3])
        mat_2_e_ionic_lower_bound = min(table_data[:,15])
        mat_2_e_ionic_upper_bound = max(table_data[:,15])

        mat_1_e_elec_lower_bound  = min(table_data[:,4])
        mat_1_e_elec_upper_bound  = max(table_data[:,4])
        mat_2_e_elec_lower_bound  = min(table_data[:,16])
        mat_2_e_elec_upper_bound  = max(table_data[:,16])

        mat_1_n_lower_bound       = min(table_data[:,5])
        mat_1_n_upper_bound       = max(table_data[:,5])
        mat_2_n_lower_bound       = min(table_data[:,17])
        mat_2_n_upper_bound       = max(table_data[:,17])

        # Elastic extrema
        mat_1_bulk_mod_lower_bound   = min(table_data[:,6])
        mat_1_bulk_mod_upper_bound   = max(table_data[:,6])
        mat_2_bulk_mod_lower_bound   = min(table_data[:,18])
        mat_2_bulk_mod_upper_bound   = max(table_data[:,18])

        mat_1_shear_mod_lower_bound  = min(table_data[:,7])
        mat_1_shear_mod_upper_bound  = max(table_data[:,7])
        mat_2_shear_mod_lower_bound  = min(table_data[:,19])
        mat_2_shear_mod_upper_bound  = max(table_data[:,19])

        mat_1_univ_aniso_lower_bound = min(table_data[:,8])
        mat_1_univ_aniso_upper_bound = max(table_data[:,8])
        mat_2_univ_aniso_lower_bound = min(table_data[:,20])
        mat_2_univ_aniso_upper_bound = max(table_data[:,20])

        # Magnetic extrema
        mat_1_tot_mag_lower_bound      = min(table_data[:,9])
        mat_1_tot_mag_upper_bound      = max(table_data[:,9])
        mat_2_tot_mag_lower_bound      = min(table_data[:,21])
        mat_2_tot_mag_upper_bound      = max(table_data[:,21])

        mat_1_tot_mag_norm_lower_bound = min(table_data[:,10])
        mat_1_tot_mag_norm_upper_bound = max(table_data[:,10])
        mat_2_tot_mag_norm_lower_bound = min(table_data[:,22])
        mat_2_tot_mag_norm_upper_bound = max(table_data[:,22])

        # Piezoelectric extrema
        mat_1_e_ij_lower_bound = min(table_data[:,11])
        mat_1_e_ij_upper_bound = max(table_data[:,11])
        mat_2_e_ij_lower_bound = min(table_data[:,23])
        mat_2_e_ij_upper_bound = max(table_data[:,23])        

        # Get materials that fall within the above extrema
        
        # Check for materials that meet carrier transport criteria
        mat_1_elec_idx = []
        mat_2_elec_idx = []
        for i, elec_cond in enumerate(final_dict["elec_cond_300K_low_doping"]):
            if (elec_cond >= mat_1_elec_cond_lower_bound) and (elec_cond <= mat_1_elec_cond_upper_bound):  
                mat_1_elec_idx.append(i)
            if (elec_cond >= mat_2_elec_cond_lower_bound) and (elec_cond <= mat_2_elec_cond_upper_bound):  
                mat_2_elec_idx.append(i)
        
        mat_1_therm_idx = [] 
        mat_2_therm_idx = []      
        for i, therm_cond in enumerate(final_dict["therm_cond_300K_low_doping"]):
            if (therm_cond >= mat_1_therm_cond_lower_bound) and (therm_cond <= mat_1_therm_cond_upper_bound):  
                mat_1_therm_idx.append(i)
            if (therm_cond >= mat_2_therm_cond_lower_bound) and (therm_cond <= mat_2_therm_cond_upper_bound):  
                mat_2_therm_idx.append(i)

        # Check for materials that meet dielectric criteria
        mat_1_e_total_idx = []
        mat_2_e_total_idx = []
        for i, e_total in enumerate(final_dict["e_total"]):
            if (e_total >= mat_1_e_total_lower_bound) and (e_total <= mat_1_e_total_upper_bound):       
                mat_1_e_total_idx.append(i)
            if (e_total >= mat_2_e_total_lower_bound) and (e_total <= mat_2_e_total_upper_bound):       
                mat_2_e_total_idx.append(i)

        mat_1_e_ionic_idx = []
        mat_2_e_ionic_idx = []
        for i, e_ionic in enumerate(final_dict["e_ionic"]):
            if (e_ionic >= mat_1_e_ionic_lower_bound) and (e_ionic <= mat_1_e_ionic_upper_bound):       
                mat_1_e_ionic_idx.append(i)
            if (e_ionic >= mat_2_e_ionic_lower_bound) and (e_ionic <= mat_2_e_ionic_upper_bound):       
                mat_2_e_ionic_idx.append(i)

        mat_1_e_elec_idx = []
        mat_2_e_elec_idx = []
        for i, e_elec in enumerate(final_dict["e_elec"]):
            if (e_elec >= mat_1_e_elec_lower_bound) and (e_ionic <= mat_1_e_elec_upper_bound):       
                mat_1_e_elec_idx.append(i)
            if (e_elec >= mat_2_e_elec_lower_bound) and (e_ionic <= mat_2_e_elec_upper_bound):       
                mat_2_e_elec_idx.append(i)

        mat_1_n_idx = []
        mat_2_n_idx = []
        for i, n in enumerate(final_dict["n"]):
            if (n >= mat_1_n_lower_bound) and (n <= mat_1_n_upper_bound):       
                mat_1_n_idx.append(i)
            if (n >= mat_2_n_lower_bound) and (n <= mat_2_n_upper_bound):       
                mat_2_n_idx.append(i)

        # Check for materials that meet elastic criteria
        mat_1_bulk_idx = []
        mat_2_bulk_idx = []
        for i, bulk_mod in enumerate(final_dict["bulk_modulus"]):
            if (bulk_mod >= mat_1_bulk_mod_lower_bound) and (bulk_mod <= mat_1_bulk_mod_upper_bound):       
                mat_1_bulk_idx.append(i)
            if (bulk_mod >= mat_2_bulk_mod_lower_bound) and (bulk_mod <= mat_2_bulk_mod_upper_bound):       
                mat_2_bulk_idx.append(i)
    
        mat_1_shear_idx = []
        mat_2_shear_idx = []
        for i, shear_mod in enumerate(final_dict["shear_modulus"]):
            if (shear_mod >= mat_1_shear_mod_lower_bound) and (shear_mod <= mat_1_shear_mod_upper_bound): 
                mat_1_shear_idx.append(i)
            if (shear_mod >= mat_2_shear_mod_lower_bound) and (shear_mod <= mat_2_shear_mod_upper_bound): 
                mat_2_shear_idx.append(i) 

        mat_1_univ_aniso_idx = []
        mat_2_univ_aniso_idx = []
        for i, univ_aniso in enumerate(final_dict["universal_anisotropy"]):
            if (univ_aniso >= mat_1_univ_aniso_lower_bound) and (univ_aniso <= mat_1_univ_aniso_upper_bound): 
                mat_1_univ_aniso_idx.append(i)
            if (univ_aniso >= mat_2_univ_aniso_lower_bound) and (univ_aniso <= mat_2_univ_aniso_upper_bound): 
                mat_2_univ_aniso_idx.append(i) 

        # Check for materials that meet magnetic criteria
        mat_1_tot_mag_idx = []
        mat_2_tot_mag_idx = []
        for i, tot_mag in enumerate(final_dict["total_magnetization"]):
            if (tot_mag >= mat_1_tot_mag_lower_bound) and (tot_mag <= mat_1_tot_mag_upper_bound):       
                mat_1_tot_mag_idx.append(i)
            if (tot_mag >= mat_2_tot_mag_lower_bound) and (tot_mag <= mat_2_tot_mag_upper_bound):       
                mat_2_tot_mag_idx.append(i)

        mat_1_tot_mag_norm_idx = []
        mat_2_tot_mag_norm_idx = []
        for i, tot_mag_norm in enumerate(final_dict["total_magnetization_normalized_volume"]):
            if (tot_mag_norm >= mat_1_tot_mag_norm_lower_bound) and (tot_mag_norm <= mat_1_tot_mag_norm_upper_bound):       
                mat_1_tot_mag_norm_idx.append(i)
            if (tot_mag_norm >= mat_2_tot_mag_norm_lower_bound) and (tot_mag_norm <= mat_2_tot_mag_norm_upper_bound):       
                mat_2_tot_mag_norm_idx.append(i)

        # Check for materials that meet piezoelectric criteria
        mat_1_e_ij_idx = []
        mat_2_e_ij_idx = []
        for i, e_ij in enumerate(final_dict["total_magnetization"]):
            if (e_ij >= mat_1_e_ij_lower_bound) and (e_ij <= mat_1_e_ij_upper_bound):       
                mat_1_e_ij_idx.append(i)
            if (e_ij >= mat_2_e_ij_lower_bound) and (e_ij <= mat_2_e_ij_upper_bound):       
                mat_2_e_ij_idx.append(i)
                
        # Find intersection
        mat_1_indices = list(set(mat_1_elec_idx)         &\
                             set(mat_1_therm_idx)        &\
                             set(mat_1_e_total_idx)      &\
                             set(mat_1_e_ionic_idx)      &\
                             set(mat_1_e_elec_idx)       &\
                             set(mat_1_bulk_idx)         &\
                             set(mat_1_shear_idx)        &\
                             set(mat_1_univ_aniso_idx)   &\
                             set(mat_1_tot_mag_idx)      &\
                             set(mat_1_tot_mag_norm_idx) &\
                             set(mat_1_e_ij_idx))
        
        mat_2_indices = list(set(mat_2_elec_idx)         &\
                             set(mat_2_therm_idx)        &\
                             set(mat_2_e_total_idx)      &\
                             set(mat_2_e_ionic_idx)      &\
                             set(mat_2_e_elec_idx)       &\
                             set(mat_2_bulk_idx)         &\
                             set(mat_2_shear_idx)        &\
                             set(mat_2_univ_aniso_idx)   &\
                             set(mat_2_tot_mag_idx)      &\
                             set(mat_2_tot_mag_norm_idx) &\
                             set(mat_2_e_ij_idx))
        
        # Extract mp-ids
        mat_1_ids = [mat_1_dict["mp-ids"][i] for i in mat_1_indices]
        mat_2_ids = [mat_2_dict["mp-ids"][i] for i in mat_2_indices]

        return mat_1_ids, mat_2_ids
    
    #------ Setter Methods ------#

    def set_lower_bounds(self, new_lower_bounds):
        self.lower_bounds = new_lower_bounds
        return self
    
    def set_upper_bounds(self, new_upper_bounds):
        self.upper_bounds = new_upper_bounds
        return self
    
    def set_retain_parents(self, new_retain_parents):
        self.retain_parents = new_retain_parents
        return self
    
    def set_allow_mutations(self, new_allow_mutations):
        self.allow_mutations = new_allow_mutations
        return self

    def set_property_docs(self, new_property_docs):
        self.property_docs = new_property_docs
        return self
    
    def set_desired_props(self, new_desired_props):
        self.desired_props = new_desired_props
        return self
    
    def set_has_props(self, new_has_props):
        self.has_props = new_has_props
        return self
    
    def set_fields(self, new_fields):
        self.fields = new_fields
        return self
    
    def set_dv(self, new_dv):
        self.dv = new_dv
        return self
    
    def set_ga_params(self, new_ga_params):
        self.ga_params = new_ga_params
        return self
    
    def set_final_population(self, new_final_pop):
        self.final_population = new_final_pop
        return self
    
    def set_cost_history(self, new_cost_history):
        self.cost_history = new_cost_history
        return self
    
    def set_lowest_costs(self, new_lowest_costs):
        self.lowest_costs = new_lowest_costs
        return self           

    def set_parent_average_costs(self, new_par_avg_costs):
        self.parent_average_costs = new_par_avg_costs
        return self
        
    def set_material_matches():
        return
    
    def set_HS_optim_params(self):

        '''
        MAIN OPTIMIZATION FUNCTION
        '''

        # Unpack necessary attributes from self
        P = self.ga_params.get_P()
        K = self.ga_params.get_K()
        G = self.ga_params.get_G()
        S = self.ga_params.get_S()
        lower_bounds = self.lower_bounds
        upper_bounds = self.upper_bounds
        
        # Initialize arrays to store the cost and original indices of each generation
        PI = np.ones((G, S))
        Orig = np.ones((G, S))
        
        # Initialize arrays to store best performer and parent avg 
        Pi_min = np.zeros(G)     # best cost
        Pi_par_avg = np.zeros(G) # avg cost of parents
        
        # Generation counter
        g = 0

        # Initialize array to store costs for current generation
        costs = np.zeros(S)

        # Randomly populate first generation  
        Lambda = Population(dv=self.dv, material_properties=self.property_docs, desired_properties=self.desired_props, ga_params=self.ga_params)
        print(f"lower bounds: {lower_bounds}")
        print(f"upper bounds: {upper_bounds}")
        Lambda.set_initial_random(lower_bounds, upper_bounds)

        # Calculate the costs of the first generation
        Lambda.set_costs()
                
        # Sort the costs of the first generation
        [sorted_costs, ind] = Lambda.sort_costs()  
        PI[g, :] = sorted_costs.reshape(1,S) 
        
        # Store the cost of the best performer and average cost of the parents 
        Pi_min[g] = np.min(sorted_costs)
        Pi_par_avg[g] = np.mean(sorted_costs[0:P])
        
        # Update Lambda based on sorted indices
        Lambda.set_order_by_costs(ind)
        
        # Perform all later generations    
        while g < G:
            
            print(g)
            costs[0:P] = sorted_costs[0:P] # retain the parents from the previous generation
            
            # Select top parents P from Lambda to be breeders
            for p in range(0, P, 2):
                phi1, phi2 = np.random.rand(2)
                kid1 = phi1*Lambda.values[p, :] + (1-phi1)*Lambda.values[p+1, :]
                kid2 = phi2*Lambda.values[p, :] + (1-phi2)*Lambda.values[p+1, :]
                
                # Append offspring to Lambda, overwriting old population members 
                Lambda.values[P+p,   :] = kid1
                Lambda.values[P+p+1, :] = kid2
            
                # Cast offspring to genetic strings and evaluate costs
                kid1 = GeneticString(dv=self.dv, values=kid1, material_properties=self.property_docs, desired_properties=self.desired_props, ga_params=self.ga_params)
                kid2 = GeneticString(dv=self.dv, values=kid2, material_properties=self.property_docs, desired_properties=self.desired_props, ga_params=self.ga_params)
                costs[P+p]   = kid1.get_cost()
                costs[P+p+1] = kid2.get_cost()
                        
            # Randomly generate new design strings to fill the rest of the population
            for i in range(S-P-K):
                upper_bounds = [1e9 if np.isinf(x) else x for x in upper_bounds]
                Lambda.values[P+K+i, :] = np.random.uniform(lower_bounds, upper_bounds)

            # Calculate the costs of the gth generation
            Lambda.set_costs()

            # Sort the costs for the gth generation
            [sorted_costs, ind] = Lambda.sort_costs()  
            PI[g, :] = sorted_costs.reshape(1,S) 
        
            # Store the cost of the best performer and average cost of the parents 
            Pi_min[g] = np.min(sorted_costs)
            Pi_par_avg[g] = np.mean(sorted_costs[0:P])
        
            # Update Lambda based on sorted indices
            Lambda.set_order_by_costs(ind)

            # Update the generation counter
            g = g + 1 

        # Update self attributes following optimization
        self.final_population = Lambda
        self.cost_history = PI
        print(f"g = {g}")
        print(f"G = {G}")
        self.lowest_costs = Pi_min
        print(f"Pi_min: {Pi_min}")
        self.parent_average_costs = Pi_par_avg     
        print(f"Pi_par_avg: {Pi_par_avg}")
        
        return self         

    #------ Other Methods ------#

    def print_table_of_best_designs(self):

        headers = ['(Phase 1) Electrical conductivity, [S/m]',
                   '(Phase 2) Electrical conductivity, [S/m]',
                   '(Phase 1) Thermal conductivity, [W/m/K]',
                   '(Phase 2) Thermal conductivity, [W/m/K]',
                   '(Phase 1) Total dielectric constant, [F/m]',
                   '(Phase 2) Total dielectric constant, [F/m]',
                   '(Phase 1) Ionic contrib dielectric constant, [F/m]',
                   '(Phase 2) Ionic contrib dielectric constant, [F/m]',
                   '(Phase 1) Electronic contrib dielectric constant, [F/m]',
                   '(Phase 2) Electronic contrib dielectric constant, [F/m]',
                   '(Phase 1) Dielectric n, [F/m]',
                   '(Phase 2) Dielectric n, [F/m]',
                   '(Phase 1) Bulk modulus, [GPa]',
                   '(Phase 2) Bulk modulus, [GPa]',
                   '(Phase 1) Shear modulus, [GPa]',
                   '(Phase 2) Shear modulus, [GPa]',
                   '(Phase 1) Universal anisotropy, []',
                   '(Phase 2) Universal anisotropy, []',
                   '(Phase 1) Total magnetization, []',
                   '(Phase 2) Total magnetization, []',
                   '(Phase 1) Total magnetization normalized volume, []',
                   '(Phase 2) Total magnetization normalized volume, []',
                   '(Phase 1) Piezoelectric constant, [C/N or m/V]',
                   '(Phase 2) Piezoelectric constant, [C/N or m/V]',
                   'Gamma, the avergaing parameter, []',
                   '(Phase 1) Volume fraction, [] ',
                   'Cost']
     
        table_data = self.get_table_of_best_designs()
        print('\nHASHIN-SHTRIKMAN + GENETIC ALGORITHM RECOMMENDED MATERIAL PROPERTIES')
        print(tabulate(table_data, headers=headers))
    
    def plot_optimization_results(self):
        fig, ax = plt.subplots(figsize=(10,6))
        ax.plot(range(self.ga_params.get_G()), self.parent_average_costs, label='Avg. of top 10 performers')
        ax.plot(range(self.ga_params.get_G()), self.lowest_costs, label="Best costs")
        plt.xlabel('Generation', fontsize= 20)
        plt.ylabel('Cost', fontsize=20)
        plt.title('Genetic Algorithm Results, Case A', fontsize = 24)
        plt.legend(fontsize = 14)
        plt.show()   

    def generate_final_dict(self):

        '''
        MAIN FUNCTION USED TO GENERATE MATERIAL PROPERTY DICTIONARY DEPENDING ON USER REQUEST
        '''

        # Initialize local variables
        get_band_gap = True

        get_elec_cond      = False
        get_therm_cond     = False
        get_mp_ids_contrib = False

        get_e_electronic = True
        get_e_ionic      = True
        get_e_total      = True
        get_n            = True
        
        get_bulk_modulus         = True
        get_shear_modulus        = True 
        get_universal_anisotropy = True

        get_total_magnetization                = False
        get_total_magnetization_normalized_vol = False
        
        get_e_ij_max = True

        if "carrier-transport" in self.property_docs:
            get_elec_cond      = True
            get_therm_cond     = True
            get_mp_ids_contrib = True
        else:
            get_elec_cond      = False
            get_therm_cond     = False
            get_mp_ids_contrib = False

        if "dielectric" in self.property_docs:
            get_e_electronic = True
            get_e_ionic      = True
            get_e_total      = True
            get_n            = True
        else:
            get_e_electronic = False
            get_e_ionic      = False
            get_e_total      = False
            get_n            = False

        if "elastic" in self.property_docs:
            get_bulk_modulus         = True
            get_shear_modulus        = True 
            get_universal_anisotropy = True
        else:
            get_bulk_modulus         = False
            get_shear_modulus        = False
            get_universal_anisotropy = False

        if "magnetic" in self.property_docs:
            get_total_magnetization                = True
            get_total_magnetization_normalized_vol = True
        else:
            get_total_magnetization                = False
            get_total_magnetization_normalized_vol = False

        if "piezoelectric" in self.property_docs:
            get_e_ij_max = True
        else:
            get_e_ij_max = False


        if get_mp_ids_contrib:
            client = Client(apikey="uJpFxJJGKCSp9s1shwg9HmDuNjCDfWbM", project="carrier_transport")
        else:
            client = Client(apikey="uJpFxJJGKCSp9s1shwg9HmDuNjCDfWbM")

        # Assemble dictionary of values needed for Hashin-Shtrikman analysis
        final_dict = {"mp-ids": [],
                      "mp-ids-contrib": [], 
                      "formula": [],
                      "metal": [],
                      "bulk_modulus": [],
                      "shear_modulus": [],
                      "universal_anisotropy": [],
                      "e_total": [],
                      "e_ionic": [],
                      "e_electronic": [],
                      "n": [],
                      "e_ij_max": [],
                      "therm_cond_300K_low_doping": [],
                      "elec_cond_300K_low_doping": []}
    
        new_fields = self.fields
        if get_band_gap not in self.fields:
            new_fields.append("band_gap")
        
        # Dielectric
        if get_e_electronic not in self.fields:
            new_fields.append("e_electronic")        
        if get_e_ionic not in self.fields:
            new_fields.append("e_ionic")
        if get_e_total not in self.fields:
            new_fields.append("e_total")
        if get_n not in self.fields:
            new_fields.append("n")

        # Elastic
        if get_bulk_modulus not in self.fields:
            new_fields.append("bulk_modulus")
        if get_shear_modulus not in self.fields:
            new_fields.append("shear_modulus")
        if get_universal_anisotropy not in self.fields:
            new_fields.append("universal_anisotropy")
        
        # Magnetic
        if get_total_magnetization not in self.fields:
            new_fields.append("total_magnetization")
        if get_total_magnetization_normalized_vol not in self.fields:
            new_fields.append("total_magnetization_normalized_vol")

        # Piezoelectric
        if get_e_ij_max not in self.fields:
            new_fields.append("e_ij_max")

        self.set_fields(new_fields)


        with MPRester(self.api_key) as mpr:
            
            docs = mpr.materials.summary.search(fields=self.fields)
            from mpi4py import MPI

            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()

            # Calculate the size of each chunk
            chunk_size = len(docs) // size

            # Calculate the start and end indices for this process's chunk
            start = rank * chunk_size
            end = start + chunk_size if rank != size - 1 else len(docs)  # The last process gets the remainder

            # Each process gets a different chunk
            chunk = docs[start:end]

            # for i, doc in enumerate(docs):
            for i, doc in enumerate(chunk):

                # print(f"{i} of {len(docs)}")
                print(f"Process {rank}: {i} of {len(chunk)}")

                try:

                    mp_id = doc.material_id                           
                    query = {"identifier": mp_id}
                    my_dict = client.download_contributions(query=query, include=["tables"])[0]
                    final_dict["mp-ids"].append(mp_id)    
                    final_dict["formula"].append(my_dict["formula"])
                    final_dict["metal"].append(my_dict["data"]["metal"])                  
                    final_dict["is_stable"].append(doc.is_stable)
                    final_dict["is_metal"].append(doc.is_metal) 

                    if get_band_gap:
                        final_dict["band_gap"].append(doc.band_gap)

                    # Dielectric
                    if get_e_electronic:
                        final_dict["e_electronic"].append(doc.e_electronic)
                    if get_e_ionic:
                        final_dict["e_ionic"].append(doc.e_ionic)
                    if get_e_total:
                        final_dict["e_total"].append(doc.e_total)
                    if get_n:
                        final_dict["n"].append(doc.n)

                    # Elastic
                    if get_bulk_modulus:
                        bulk_modulus_voigt = doc.bulk_modulus["voigt"]
                        final_dict["bulk_modulus"].append(bulk_modulus_voigt)
                    if get_shear_modulus:
                        shear_modulus_voigt = doc.shear_modulus["voigt"]
                        final_dict["shear_modulus"].append(shear_modulus_voigt)
                    if get_universal_anisotropy:
                        final_dict["universal_anisotropy"].append(doc.universal_anisotropy)                   

                    # Magnetic
                    if get_total_magnetization:
                        final_dict["total_magnetization"].append(doc.total_magnetization)
                    if get_total_magnetization_normalized_vol:
                        final_dict["total_magnetization_normalized_vol"].append(doc.total_magnetization_normalized_vol)

                    # Piezoelectric
                    if get_e_ij_max:
                        final_dict["e_ij_max"].append(doc.e_ij_max)
                    
                    # Carrier transport
                    if get_mp_ids_contrib:

                        try:
                            final_dict["mp-ids-contrib"].append(my_dict["identifier"])
                            thermal_cond = my_dict["tables"][7].iloc[2, 1] * 1e-14  # multply by relaxation time, 10 fs
                            elec_cond = my_dict["tables"][5].iloc[2, 1] * 1e-14 # multply by relaxation time, 10 fs   
                            final_dict["therm_cond_300K_low_doping"].append(thermal_cond)
                            final_dict["elec_cond_300K_low_doping"].append(elec_cond)              

                        except:
                            IndexError

                except:
                    TypeError

        
        # After the for loop
        final_dicts = comm.gather(final_dict, root=0)

        # On process 0, consolidate the results
        if rank == 0:
            consolidated_dict = {
                "mp-ids-contrib": [],
                "therm_cond_300K_low_doping": [],
                "elec_cond_300K_low_doping": [],
                # Add other keys as needed
            }

            for final_dict in final_dicts:
                for key in consolidated_dict:
                    consolidated_dict[key].extend(final_dict[key])

            # Save the consolidated results to a JSON file
            now = datetime.now()
            my_file_name = "final_dict_" + now.strftime("%m/%d/%Y, %H:%M:%S")
            with open(my_file_name, "w") as my_file:
                json.dump(consolidated_dict, my_file)

        # now = datetime.now()
        # my_file_name = "final_dict_" + now.strftime("%m/%d/%Y, %H:%M:%S")
        # with open(my_file_name, "w") as my_file:
        #     json.dump(final_dict, my_file)

        return final_dicts
        
    

    