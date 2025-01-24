"""test_optimization_flow.py."""
from hashin_shtrikman_mp.core.user_input import MaterialProperty, Material, MixtureProperty, Mixture, UserInput
from hashin_shtrikman_mp.core import GeneticAlgorithm
from hashin_shtrikman_mp.core.genetic_algorithm import OptimizationParams
from hashin_shtrikman_mp.core.match_finder import MatchFinder

def test_optimization_flow():   

    """
    Integration test for the optimization flow of the Hashin-Shtrikman bounds optimization problem.

    Notes:
    - Run `pip install -e .` in the root directory to install the package in editable mode.

    """ 

    # Define properties for each material
    print("Defining properties for each material...")
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
    print("Creating Material & Mixture instances...")
    mat_1 = Material(name='mat_1', properties=properties_mat_1)
    mat_2 = Material(name='mat_2', properties=properties_mat_2)
    mat_3 = Material(name='mat_3', properties=properties_mat_3)
    mixture = Mixture(name='mixture', properties=properties_mixture)
    aggregate = [mat_1, mat_2, mat_3, mixture]

    # Initialize UserInput instance with materials and mixtures
    print("Initializing UserInput instance...")
    user_input= UserInput(materials=[mat_1, mat_2, mat_3], mixtures=[mixture])

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

    # Create the consolidated_dict
    print("Creating the consolidated dictionary...")
    overall_bounds_dict = {}
    for prop, bounds in overall_bounds.items():
        overall_bounds_dict[prop] = {
            'upper_bound': bounds['upper_bound'],
            'lower_bound': bounds['lower_bound']
        }

    # Initialize optimization parameters and genetic algorithm
    print("Initializing optimization parameters and genetic algorithm...")
    optimization_parameters = OptimizationParams.from_user_input(user_input)
    ga = GeneticAlgorithm()

    assert optimization_parameters.property_categories, "There are no property categories in optimization parameters."
    assert optimization_parameters.property_docs, "There are no property docs in optimization parameters."
    assert optimization_parameters.lower_bounds, "There are no lower bounds in optimization parameters."
    assert optimization_parameters.upper_bounds, "There are no upper bounds in optimization parameters."
    assert optimization_parameters.num_materials, "The number of materials is not defined in optimization parameters."
    assert optimization_parameters.num_properties, "The number of properties is not defined in optimization parameters."

    # Run the optimization
    print("Running the optimization...")
    ga_result = ga.run(user_input, gen_counter=True)

    # Create an instance of the MatchFinder class using the genetic algorithm result
    match_finder = MatchFinder(ga_result)

    # Get material matches
    print("Getting material matches...")
    matches_dict = match_finder.get_material_matches(overall_bounds_dict)

    # Check that the matches_dict is non-empty
    assert len(matches_dict['mat1']) > 1
    assert len(matches_dict['mat2']) > 1
    assert len(matches_dict['mat3']) > 1

    # Test that `generate_consolidated_dict` method works as expected
    test_consolidated_dict = match_finder.generate_consolidated_dict(overall_bounds_dict=overall_bounds_dict)
    assert test_consolidated_dict, "The generated dictionary is empty."

    # Test complete
    print("Test completed successfully.")


# Perform the test
if __name__ == "__main__":
    test_optimization_flow()
