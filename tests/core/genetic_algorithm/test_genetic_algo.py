"""test_genetic_algo.py"""
from hashin_shtrikman_mp.core.genetic_algorithm.genetic_algorithm import GeneticAlgorithm
from hashin_shtrikman_mp.core.genetic_algorithm.genetic_algorithm_parameters import GeneticAlgorithmParams
from hashin_shtrikman_mp.core.genetic_algorithm.genetic_algorithm_result import GeneticAlgorithmResult
from hashin_shtrikman_mp.core.genetic_algorithm.population import Population
from hashin_shtrikman_mp.core.user_input import UserInput
import numpy as np
from pydantic import ValidationError
import pytest
from unittest.mock import MagicMock

mock_num_parents     = 10
mock_num_generations = 100
mock_num_members     = 200

# Test running of the genetic algorithm optimization
def test_genetic_algorithm_run():
    # Mock the user input
    mock_user_input = MagicMock(spec=UserInput)

    # Mock optimization parameters
    mock_optimization_params = MagicMock()
    mock_optimization_params.property_categories = ['carrier-transport', 'elastic']
    mock_optimization_params.property_docs = {
        'carrier-transport': {
            'elec_cond_300k_low_doping':  None, 
            'therm_cond_300k_low_doping': None
        }, 
        'dielectric': {
            'e_electronic': None, 
            'e_ionic':      None, 
            'e_total':      None, 
            'n':            None
        }, 
        'elastic': {
            'bulk_modulus':         'vrh', 
            'shear_modulus':        'vrh', 
            'universal_anisotropy': None
        }, 
        'magnetic': {
            'total_magnetization':                None, 
            'total_magnetization_normalized_vol': None
        }, 
        'piezoelectric': {
            'e_ij_max': None
        }
    }
    mock_optimization_params.lower_bounds = {
        'mat_1': {
            'carrier-transport': [0.0, 0.0], 
            'elastic':           [0.0, 0.0, 0.0]
        },
        'mat_2': {
            'carrier-transport': [0.0, 0.0], 
            'elastic':           [0.0, 0.0, 0.0]
        },
        'mat_3': {
            'carrier-transport': [0.0, 0.0], 
            'elastic':           [0.0, 0.0, 0.0]
        },
        'volume-fractions': [0.01, 0.01, 0.01]
    }
    mock_optimization_params.upper_bounds = {
        'mat_1': {
            'carrier-transport': [1.0, 1.0], 
            'elastic':           [1.0, 1.0, 1.0]
        },
        'mat_2': {
            'carrier-transport': [1.0, 1.0], 
            'elastic':           [1.0, 1.0, 1.0]
        },
        'mat_3': {
            'carrier-transport': [1.0, 1.0], 
            'elastic':           [1.0, 1.0, 1.0]
        },
        'volume-fractions': [1.0, 1.0, 1.0]
    }
    mock_optimization_params.desired_props = {
        'carrier-transport': [0.5, 0.5], 
        'elastic':           [0.5, 0.5, 0.5]
    }

    mock_optimization_params.num_materials  = 3
    mock_optimization_params.num_properties = 6
    mock_optimization_params.indices_elastic_moduli = [2, 3]

    # Patch OptimizationParams.from_user_input
    with pytest.MonkeyPatch().context() as m:
        m.setattr('hashin_shtrikman_mp.core.genetic_algorithm.optimization_params.OptimizationParams.from_user_input', 
                  lambda x: mock_optimization_params)

        # Mock genetic algorithm parameters
        ga_params = GeneticAlgorithmParams(num_parents=mock_num_parents, num_kids=mock_num_parents, num_generations=mock_num_generations, num_members=mock_num_members)

        # Mock Population class
        mock_population = MagicMock(spec=Population)
        mock_population.sort_costs.return_value = (np.array([1.0, 2.0, 3.0, 4.0]), np.array([0, 1, 2, 3]))

        with pytest.MonkeyPatch().context() as m:
            m.setattr('hashin_shtrikman_mp.core.genetic_algorithm.population.Population', lambda *args, **kwargs: mock_population)

            # Instantiate the GeneticAlgorithm
            ga = GeneticAlgorithm()

            # Run the algorithm
            result = ga.run(mock_user_input, ga_params, gen_counter=False)

            # Assertions
            assert isinstance(result, GeneticAlgorithmResult)
            assert result.algo_parameters == ga_params
            assert result.final_population != None
            assert result.lowest_costs.shape == (mock_num_generations,)
            assert result.avg_parent_costs.shape == (mock_num_generations,)

# Test for default values
def test_default_values():
    # Creating an instance with no arguments (will use the default values)
    ga_params = GeneticAlgorithmParams()

    # Check if default values are correctly set
    assert ga_params.num_parents        == 10
    assert ga_params.num_kids           == 10
    assert ga_params.num_generations    == 100
    assert ga_params.num_members        == 200
    assert ga_params.mixing_param       == 0.5
    assert ga_params.tolerance          == 1.0
    assert ga_params.weight_eff_prop    == 1.0
    assert ga_params.weight_conc_factor == 1.0

# Test for custom values
def test_custom_values():
    # Customizing the parameters
    ga_params = GeneticAlgorithmParams(
        num_parents=20,
        num_kids=15,
        num_generations=50,
        num_members=100,
        mixing_param=0.7,
        tolerance=0.8,
        weight_eff_prop=1.5,
        weight_conc_factor=2.0
    )

    # Check if the custom values are correctly set
    assert ga_params.num_parents        == 20
    assert ga_params.num_kids           == 15
    assert ga_params.num_generations    == 50
    assert ga_params.num_members        == 100
    assert ga_params.mixing_param       == 0.7
    assert ga_params.tolerance          == 0.8
    assert ga_params.weight_eff_prop    == 1.5
    assert ga_params.weight_conc_factor == 2.0

# Test for validation errors (invalid values)
def test_invalid_values():
    # Test invalid value for num_parents (should be a positive integer)
    with pytest.raises(ValidationError):
        GeneticAlgorithmParams(num_parents=-1)

    # Test invalid value for num_kids (should be a positive integer)
    with pytest.raises(ValidationError):
        GeneticAlgorithmParams(num_kids=-1)

    # Test invalid value for mixing_param (should be a float between 0 and 1)
    with pytest.raises(ValidationError):
        GeneticAlgorithmParams(mixing_param=1.5)

    # Test invalid value for tolerance (should be a positive float)
    with pytest.raises(ValidationError):
        GeneticAlgorithmParams(tolerance=-0.1)

    # Test invalid value for weight_eff_prop (should be a positive float)
    with pytest.raises(ValidationError):
        GeneticAlgorithmParams(weight_eff_prop=-1.0)

    # Test invalid value for weight_conc_factor (should be a positive float)
    with pytest.raises(ValidationError):
        GeneticAlgorithmParams(weight_conc_factor=-1.0)

# Test for boundary values
def test_boundary_values():
    # Test mixing_param on boundary (should be a float between 0 and 1)
    ga_params_min = GeneticAlgorithmParams(mixing_param=0.0)
    ga_params_max = GeneticAlgorithmParams(mixing_param=1.0)

    assert ga_params_min.mixing_param == 0.0
    assert ga_params_max.mixing_param == 1.0

    # Test tolerance at the boundary
    ga_params_tolerance = GeneticAlgorithmParams(tolerance=0.0)
    assert ga_params_tolerance.tolerance == 0.0

# Test for missing required fields (should raise a ValidationError)
def test_missing_required_fields():
    with pytest.raises(ValidationError):
        # Missing some required fields should raise an error
        GeneticAlgorithmParams(num_parents=None)
