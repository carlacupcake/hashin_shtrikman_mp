from hashin_shtrikman_mp.core.user_input import MaterialProperty, Material, MixtureProperty, Mixture, UserInput
from hashin_shtrikman_mp.core import GeneticAlgorithmParams, GeneticAlgorithm
from hashin_shtrikman_mp.core.genetic_algorithm import OptimizationParams
from hashin_shtrikman_mp.core.match_finder import MatchFinder


def test_optimization_flow():

    # Define properties for each material
    properties_mat_1 = [
        MaterialProperty(prop='elec_cond_300k_low_doping', upper_bound=120, lower_bound=1e-7), # upper_bound=20, lower_bound=1 
        MaterialProperty(prop='therm_cond_300k_low_doping', upper_bound=2, lower_bound=1e-7), # upper_bound=0.0001, lower_bound=1e-5 
        MaterialProperty(prop='bulk_modulus', upper_bound=500, lower_bound=50),
        MaterialProperty(prop='shear_modulus', upper_bound=500, lower_bound=80),
        MaterialProperty(prop='universal_anisotropy', upper_bound=6, lower_bound=1),
    ]

    properties_mat_2 = [
        MaterialProperty(prop='elec_cond_300k_low_doping', upper_bound=78, lower_bound=1e-7), # upper_bound=5, lower_bound=2
        MaterialProperty(prop='therm_cond_300k_low_doping', upper_bound=2, lower_bound=1e-7), # upper_bound=0.009, lower_bound=1e-4
        MaterialProperty(prop='bulk_modulus', upper_bound=400, lower_bound=20),
        MaterialProperty(prop='shear_modulus', upper_bound=500, lower_bound=100),
        MaterialProperty(prop='universal_anisotropy', upper_bound=4.3, lower_bound=1.3),
    ]

    properties_mat_3 = [
        MaterialProperty(prop='elec_cond_300k_low_doping', upper_bound=78, lower_bound=1e-7), # upper_bound=10, lower_bound=1
        MaterialProperty(prop='therm_cond_300k_low_doping', upper_bound=2, lower_bound=1e-7), # upper_bound=0.005, lower_bound=1e-4
        MaterialProperty(prop='bulk_modulus', upper_bound=700, lower_bound=20),
        MaterialProperty(prop='shear_modulus', upper_bound=600, lower_bound=100),
        MaterialProperty(prop='universal_anisotropy', upper_bound=2.1, lower_bound=0.9),
    ]

    # Define properties for the mixture
    properties_mixture = [
        MixtureProperty(prop='elec_cond_300k_low_doping', desired_prop=9),
        MixtureProperty(prop='therm_cond_300k_low_doping', desired_prop=0.9),
        MixtureProperty(prop='bulk_modulus', desired_prop=280),
        MixtureProperty(prop='shear_modulus', desired_prop=230),
        MixtureProperty(prop='universal_anisotropy', desired_prop=1.5),
    ]

    # Create Material & Mixture instances
    mat_1 = Material(name='mat_1', properties=properties_mat_1)
    mat_2 = Material(name='mat_2', properties=properties_mat_2)
    mat_3 = Material(name='mat_3', properties=properties_mat_3)
    mixture = Mixture(name='mixture', properties=properties_mixture)
    aggregate = [mat_1, mat_2, mat_3, mixture]

    # Initialize UserInput instance with materials and mixtures
    user_input= UserInput(materials=[mat_1, mat_2, mat_3], mixtures=[mixture])
    print("User Input: ", user_input)

    # Initialize dictionaries to store the overall upper and lower bounds for each property
    overall_bounds = {}

    # Iterate over materials
    for entity in aggregate:
        # Skip the mixture as it doesn't have upper and lower bounds
        if isinstance(entity, Material):
            for property in entity.properties:
                prop_name = property.prop

                # Initialize the overall bounds if they are not already present for the property
                if prop_name not in overall_bounds:
                    overall_bounds[prop_name] = {'upper_bound': property.upper_bound, 'lower_bound': property.lower_bound}
                else:
                    # Update overall upper and lower bounds by comparing with existing values
                    overall_bounds[prop_name]['upper_bound'] = max(overall_bounds[prop_name]['upper_bound'], property.upper_bound)
                    overall_bounds[prop_name]['lower_bound'] = min(overall_bounds[prop_name]['lower_bound'], property.lower_bound)

    # Print the overall bounds for each property
    print("Overall Upper & Lower Bounds:")
    for prop, bounds in overall_bounds.items():
        print(f"Property: {prop}, Upper Bound: {bounds['upper_bound']}, Lower Bound: {bounds['lower_bound']}")

    # Step 1: Create the consolidated_dict
    overall_bounds_dict = {}
    for prop, bounds in overall_bounds.items():
        overall_bounds_dict[prop] = {
            'upper_bound': bounds['upper_bound'],
            'lower_bound': bounds['lower_bound']
        }

    optimizer = GeneticAlgorithm()
    ga_result = optimizer.run(user_input, gen_counter=True)    


    match_finder = MatchFinder(ga_result)
    matches_dict = match_finder.get_material_matches(overall_bounds_dict)
    assert len(matches_dict['mat1']) > 1
    assert len(matches_dict['mat2']) > 1
    assert len(matches_dict['mat3']) > 1

    test_consolidated_dict = match_finder.generate_consolidated_dict(overall_bounds_dict=overall_bounds_dict)    
