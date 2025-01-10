from hashin_shtrikman_mp.core.utilities import load_property_categories
from hashin_shtrikman_mp.core.user_input import UserInput, Material, MaterialProperty, MixtureProperty, Mixture

def test_load_property_categories():

    properties_mat_1 = [
        MaterialProperty(prop='elec_cond_300k_low_doping', upper_bound=120, lower_bound=1e-7),
        MaterialProperty(prop='e_electronic', upper_bound=2, lower_bound=1e-7),
    ]

    properties_mat_2 = [
        MaterialProperty(prop='elec_cond_300k_low_doping', upper_bound=78, lower_bound=1e-7),
        MaterialProperty(prop='e_electronic', upper_bound=2, lower_bound=1e-7),
    ]

    # Define properties for the mixture
    properties_mixture = [
        MixtureProperty(prop='elec_cond_300k_low_doping', desired_prop=9),
        MixtureProperty(prop='e_electronic', desired_prop=0.9),
    ]

    # Create Material & Mixture instances
    mat_1 = Material(name='mat_1', properties=properties_mat_1)
    mat_2 = Material(name='mat_2', properties=properties_mat_2)
    mixture = Mixture(name='mixture', properties=properties_mixture)
    aggregate = [mat_1, mat_2, mixture]

    # Initialize UserInput instance with materials and mixtures
    user_input= UserInput(materials=[mat_1, mat_2], mixtures=[mixture])    

    categories, _ = load_property_categories(user_input=user_input)
    assert "dielectric" in categories
    assert "carrier-transport" in categories