"""test_genetic_algorithm.py"""
import numpy as np
import pytest

from unittest.mock import MagicMock

from hashin_shtrikman_mp.core.genetic_algorithm.genetic_algorithm import GeneticAlgorithm
from hashin_shtrikman_mp.core.genetic_algorithm.genetic_algorithm_parameters import GeneticAlgorithmParams
from hashin_shtrikman_mp.core.genetic_algorithm.genetic_algorithm_result import GeneticAlgorithmResult
from hashin_shtrikman_mp.core.genetic_algorithm.population import Population
from hashin_shtrikman_mp.core.user_input import UserInput

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
