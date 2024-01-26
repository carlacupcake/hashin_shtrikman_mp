from ga_params_class import GAParams
from hashin_sktrikman_class import HashinShtrikman
from population_class import Population
from genetic_string_class import GeneticString

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

# DEFAULT_UPPER_BOUNDS = [1000] * 24 + [1] * 2 
# DEFAULT_LOWER_BOUNDS = [1.1e6, 1.1e6, 0.1, 0.5, 1, 1, 1, 1, 1, 1, 1, 1, 100, 110, 10, 10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
DEFAULT_UPPER_BOUNDS = [2.2e7, 2.2e7, 10, 10, 100, 100, 100, 100, 100, 100, 100, 100, 150, 160, 60, 60, 100, 100, 100, 100, 100, 100, 100, 100, 1, 1]
DEFAULT_DESIRED_PROPS = [2.0e7, 5, 2, 2, 2, 2, 135, 40, 2, 2, 2, 2]

HS = HashinShtrikman(upper_bounds=DEFAULT_UPPER_BOUNDS,
                     desired_props=DEFAULT_DESIRED_PROPS,
                     property_docs=["carrier-transport", "elastic"])

HS.generate_final_dict(api_key="uJpFxJJGKCSp9s1shwg9HmDuNjCDfWbM",
                       mp_contribs_project="carrier_transport",
                       total_docs=100)