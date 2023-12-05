import itertools
import warnings
from functools import lru_cache
from json import loads
from os import environ
from typing import Dict, List, Literal, Optional, Union
import numpy as np

from emmet.core.mpid import MPID
from emmet.core.settings import EmmetSettings
from emmet.core.summary import HasProps
from packaging import version
from requests import Session, get

from mp_api.client.core.settings import MAPIClientSettings
from mp_api.client.core.utils import validate_ids

_DEPRECATION_WARNING = (
    "MPRester is being modernized. Please use the new method suggested and "
    "read more about these changes at https://docs.materialsproject.org/api. The current "
    "methods will be retained until at least January 2022 for backwards compatibility."
)

_EMMET_SETTINGS = EmmetSettings()
_MAPI_SETTINGS = MAPIClientSettings()

DEFAULT_API_KEY = environ.get("MP_API_KEY", None)
DEFAULT_ENDPOINT = environ.get("MP_API_ENDPOINT", "https://api.materialsproject.org/")

class HashinShtrikman:
    def __init__(
            self,
            api_key: Optional[str] = None,
            endpoint: str = DEFAULT_ENDPOINT,
            retain_parents: bool=True, 
            allow_mutations: bool=False,
            property_docs: list=["carrier-transport", "dielectric", "elastic", "magnetic", "piezoelectric"],
            desired_props: list=[],
        ):
            """
            Args:
            api_key
            retain_parents
            allow_mutations
            property_docs
            desired_props
            """

            #------ User-defined inputs ------#
            self.api_key = api_key or DEFAULT_API_KEY
            self.endpoint = endpoint or DEFAULT_ENDPOINT
            self.retain_parents = retain_parents
            self.allow_mutations = allow_mutations
            self.property_docs = property_docs
            self.desired_props = desired_props   

            #------ Other attributes ------# 
            self.fields = ["material_id", "is_stable", "band_gap", "is_metal"]
            self.dv = 2 # dimension of a gentic strng, initialize with 2 to account for g and v1

            # Search bounds
            self.Lam0_min  = 0    # Lower bound for electrical conductivity, [S/m]
            self.Lam0_max  = 1e9  # Upper bound for electrical conductivity, [S/m]
            self.Lam1_min  = 0    # Lower bound for thermal conductivity, [W/m/K]
            self.Lam1_max  = 1e9  # Upper bound for thermal conductivity, [W/m/K]
            self.Lam2_min  = 0    # Lower bound for total dielectric constant, [F/m]  < -- TODO check units !!, just make one max and min ??
            self.Lam2_max  = 1e9  # Upper bound for total dielectric constant, [F/m]
            self.Lam3_min  = 0    # Lower bound for ionic contrib dielectric constant, [F/m]
            self.Lam3_max  = 1e9  # Upper bound for ionic contrib dielectric constant, [F/m]
            self.Lam4_min  = 0    # Lower bound for electronic contrib dielectric constant, [F/m]
            self.Lam4_max  = 1e9  # Upper bound for electronic contrib dielectric constant, [F/m]
            self.Lam5_min  = 0    # Lower bound for dielectric n, [F/m]
            self.Lam5_max  = 1e9  # Upper bound for dielectric n, [F/m]
            self.Lam6_min  = 0    # Lower bound for bulk modulus, [GPa]
            self.Lam6_max  = 1e9  # Upper bound for bulk modulus, [GPa]
            self.Lam7_min  = 0    # Lower bound for shear modulus, [GPa]
            self.Lam7_max  = 1e9  # Upper bound for shear modulus, [GPa]
            self.Lam8_min  = 0    # Lower bound for universal anisotropy, []
            self.Lam8_max  = 1e9  # Upper bound for universal anisotropy, []
            self.Lam9_min  = 0    # Lower bound for total magnetization, []
            self.Lam9_max  = 1e9  # Upper bound for total magnetization, []
            self.Lam10_min = 0    # Lower bound for total magnetization normalized volume, []
            self.Lam10_max = 1e9  # Upper bound for total magnetization normalized volume, []
            self.Lam11_min = 0    # Lower bound for piezoelectric constant, [C/N or m/V]
            self.Lam11_max = 1e9  # Upper bound for piezoelectric constant, [C/N or m/V]
            self.Lam12_min = 0    # Lower bound for gamma the avergaing parameter, []
            self.Lam12_max = 1    # Upper bound for gamma the avergaing parameter, []
            self.Lam13_min = 0    # Lower bound for volume fraction of phase 1, []
            self.Lam13_max = 1    # Upper bound for volume fraction of phase 1, []    
            
            # Genetic algorithm parameters
            self.P = 10   # number of design strings to breed
            self.K = 10   # number of offspring design strings 
            self.G = 5000 # maximum number of generations
            self.S = 200  # total number of design strings per generation

            # Concentration factor tolerances
            self.TOL = 0.5 # property tolerance for convergence

            # Given Cost Function Weights
            '''TODO !!
            W1 = 1/3 # Electrical weight,            []
            W2 = 1/3 # Thermal weight,               []
            W3 = 1/3 # Mechanical weight,            []
            w1 = 1   # Material property weights,    []
            wj = 0.5 # Concentration tensor weights, []
            '''

            #------ Check API connections are working properly ------# 
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

            # Check if emmet version of server os compatible
            emmet_version = version.parse(self.get_emmet_version())

            if version.parse(emmet_version.base_version) < version.parse(
                _MAPI_SETTINGS.MIN_EMMET_VERSION
            ):
                warnings.warn(
                    "The installed version of the mp-api client may not be compatible with the API server. "
                    "Please install a previous version if any problems occur."
                )

            if not self.endpoint.endswith("/"):
                self.endpoint += "/"


    def assemble_search_bound_vectors(self):
        """
        Structure of full genetic string:
        Lambda = [elec_cond, therm_cond, e_total, e_ionic, e_elec, n,    bulk_mod, shear_mod, univ_aniso, tot_mag, tot_mag_norm_vol, piezo, g,     v1]
        Lambda = [Lam0,      Lam1,       Lam2,    Lam3,    Lam4,   Lam5, Lam6,     Lam7,      Lam8,       Lam9,    Lam10,            Lam11, Lam12, Lam13]
        Each element in the genetic string has its own search bounds, comprised of a min and max          
        """

        lower_bounds = []
        upper_bounds = []
        if "carrier-transport" in self.property_docs:
            lower_bounds.append(self.Lam0_min)
            upper_bounds.append(self.Lam0_max)
            lower_bounds.append(self.Lam1_min)
            upper_bounds.append(self.Lam1_max)
        if "dielectric" in self.property_docs:
            lower_bounds.append(self.Lam2_min)
            upper_bounds.append(self.Lam2_max)
            lower_bounds.append(self.Lam3_min)
            upper_bounds.append(self.Lam3_max)
            lower_bounds.append(self.Lam4_min)
            upper_bounds.append(self.Lam4_max)
            lower_bounds.append(self.Lam5_min)
            upper_bounds.append(self.Lam5_max)
        if "elastic" in self.property_docs:
            lower_bounds.append(self.Lam6_min)
            upper_bounds.append(self.Lam6_max)
            lower_bounds.append(self.Lam7_min)
            upper_bounds.append(self.Lam7_max)
            lower_bounds.append(self.Lam8_min)
            upper_bounds.append(self.Lam8_max)
        if "magnetic" in self.property_docs:
            lower_bounds.append(self.Lam9_min)
            upper_bounds.append(self.Lam9_max)
            lower_bounds.append(self.Lam10_min)
            upper_bounds.append(self.Lam10_max)
        if "piezoelectric" in self.property_docs:
            lower_bounds.append(self.Lam11_min)
            upper_bounds.append(self.Lam11_max)

        # Add gamma and volume fraction no matter what
        lower_bounds.append(self.Lam12_min)
        upper_bounds.append(self.Lam12_max)
        lower_bounds.append(self.Lam13_min)
        upper_bounds.append(self.Lam13_max)
        
        # Cast to numpy array
        self.lower_bounds = np.array(lower_bounds)
        self.upper_bounds = np.array(upper_bounds)

        return self.lower_bounds, self.upper_bounds
    
    def append_fields(self):

        fields = self.fields
        if "dielectric" in self.property_docs:
            fields.append("e_total")
            fields.append("e_ionic")
            fields.append("e_electronic")
            fields.append("n")
        if "elastic" in self.property_docs:
            fields.append("bulk_modulus")
            fields.append("shear_modulus")
            fields.append("universal_anisotropy")
        if "magnetic" in self.property_docs:
            fields.append("total_magnetization")
            fields.append("total_magnetization_normalized_vol") 
        if "piezoelectric" in self.property_docs:
            fields.append("e_ij_max")

        self.fields = fields
        return self

    def append_has_props(self):
        has_props = []
        if "dielectric" in self.property_docs:
            has_props.append(HasProps.dielectric)
        if "elastic" in self.property_docs:
            has_props.append(HasProps.elasticity)

        self.has_props = has_props
        return self

    def get_genetic_string_dimensions(self):

        dv = self.dv
        if "carrier-transport" in self.property_docs:
            dv = dv + 2
        if "dielectric" in self.property_docs:
            dv = dv + 4
        if "elastic" in self.property_docs:
            dv = dv + 3
        if "magnetic" in self.property_docs:
            dv = dv + 2
        if "piezoelectric" in self.property_docs:
            dv = dv + 1

        self.dv = dv
        return self.dv