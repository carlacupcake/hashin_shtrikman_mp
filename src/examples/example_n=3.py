import json
import matplotlib.pyplot as plt
import sys    
sys.path.insert(1, '../core')
from optimization import HashinShtrikman
from user_input import MaterialProperty, Material, MixtureProperty, Mixture
from genetic_algo import GAParams
from member import Member
from population import Population

# Testing without calls to generate final dict (faster)
consolidated_dict = {}
with open("consolidated_dict_02_11_2024_23_45_58") as f:
    consolidated_dict = json.load(f)

# Define properties for each material, testing with a subset of property categories
properties_mat_1 = [
    MaterialProperty(prop='elec_cond_300k_low_doping', upper_bound=20, lower_bound=1),
    MaterialProperty(prop='therm_cond_300k_low_doping', upper_bound=0.0001, lower_bound=1e-5),
    MaterialProperty(prop='bulk_modulus', upper_bound=100, lower_bound=50),
    MaterialProperty(prop='shear_modulus', upper_bound=100, lower_bound=80),
    MaterialProperty(prop='universal_anisotropy', upper_bound=2, lower_bound=1),
]

properties_mat_2 = [
    MaterialProperty(prop='elec_cond_300k_low_doping', upper_bound=5, lower_bound=2),
    MaterialProperty(prop='therm_cond_300k_low_doping', upper_bound=0.009, lower_bound=1e-4),
    MaterialProperty(prop='bulk_modulus', upper_bound=400, lower_bound=20),
    MaterialProperty(prop='shear_modulus', upper_bound=200, lower_bound=100),
    MaterialProperty(prop='universal_anisotropy', upper_bound=2.3, lower_bound=1.3),
]

properties_mat_3 = [
    MaterialProperty(prop='elec_cond_300k_low_doping', upper_bound=10, lower_bound=1),
    MaterialProperty(prop='therm_cond_300k_low_doping', upper_bound=0.005, lower_bound=1e-4),
    MaterialProperty(prop='bulk_modulus', upper_bound=300, lower_bound=20),
    MaterialProperty(prop='shear_modulus', upper_bound=300, lower_bound=100),
    MaterialProperty(prop='universal_anisotropy', upper_bound=2.1, lower_bound=0.9),
]

# Define properties for the mixture
properties_mixture = [
    MixtureProperty(prop='elec_cond_300k_low_doping', desired_prop=9),
    MixtureProperty(prop='therm_cond_300k_low_doping', desired_prop=0.007),
    MixtureProperty(prop='bulk_modulus', desired_prop=234),
    MixtureProperty(prop='shear_modulus', desired_prop=150),
    MixtureProperty(prop='universal_anisotropy', desired_prop=1.5),
]

# Create Material & Mixture instances
mat_1 = Material(name='mat_1', properties=properties_mat_1)
mat_2 = Material(name='mat_2', properties=properties_mat_2)
mat_3 = Material(name='mat_3', properties=properties_mat_3)
mixture = Mixture(name='mixture', properties=properties_mixture)
aggregate = [mat_1, mat_2, mat_3, mixture]

user_input = {}
for entity in aggregate:
    entity_name = entity.name
    properties = entity.properties
    user_input[entity_name] = {}
    for property in properties:  
        property_name = property.prop
        user_input[entity_name][property_name] = {}
        if type(property) == MaterialProperty:
            user_input[entity_name][property_name]['upper_bound'] = property.upper_bound  
            user_input[entity_name][property_name]['lower_bound'] = property.lower_bound   
        if type(property) == MixtureProperty:
            user_input[entity_name][property_name]['desired_prop'] = property.desired_prop


if __name__ == "__main__":
    HS = HashinShtrikman(api_key="uJpFxJJGKCSp9s1shwg9HmDuNjCDfWbM", user_input=user_input)
    HS.set_HS_optim_params()
    print(f'lowest costs: {HS.lowest_costs}')

    HS.print_table_of_best_designs()
    HS.plot_optimization_results()
    HS.get_material_matches()

    matches_dict = HS.get_material_matches(consolidated_dict=consolidated_dict)
    print("Material Matches:")
    print(matches_dict)
