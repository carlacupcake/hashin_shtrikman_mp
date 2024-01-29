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

lower_bounds = [0] * 5
upper_bounds = [1e9] * 5

HS = HashinShtrikman(property_docs=["carrier-transport", "elastic"],
                     api_key="uJpFxJJGKCSp9s1shwg9HmDuNjCDfWbM",
                     mp_contribs_project="carrier_transport",)

HS.generate_final_dict(total_docs=50)
HS.set_HS_optim_params()
HS.plot_optimization_results()
HS.print_table_of_best_designs()