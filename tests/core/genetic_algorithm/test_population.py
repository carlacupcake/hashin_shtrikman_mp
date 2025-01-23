"""test_population.py"""
from hashin_shtrikman_mp.core.genetic_algorithm.population import Population
from hashin_shtrikman_mp.core.genetic_algorithm.genetic_algorithm_parameters import GeneticAlgorithmParams
import numpy as np
import pytest

mock_num_materials  = 3
mock_num_properties = 6
mock_num_members    = 4

@pytest.fixture
def mock_ga_params():
    return GeneticAlgorithmParams(num_members=mock_num_members)

@pytest.fixture
def mock_property_docs():
    return {
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

@pytest.fixture
def mock_opt_params(mock_property_docs):
    class MockOptimizationParams:
        num_materials  = mock_num_materials
        num_properties = mock_num_properties
        property_categories = ['carrier-transport', 'elastic']
        desired_props = {'carrier-transport': [0.5, 0.5], 'elastic': [0.5, 0.5, 0.5]}
        property_docs = mock_property_docs
    return MockOptimizationParams()

@pytest.fixture
def mock_population(mock_opt_params, mock_ga_params):
    values = np.array([[1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 1.10, 1.11, 1.12, 1.13, 1.14, 1.15, 0.33, 0.33, 0.34],
                       [2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 2.10, 2.11, 2.12, 2.13, 2.14, 2.15, 0.33, 0.33, 0.34],
                       [3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 3.10, 3.11, 3.12, 3.13, 3.14, 3.15, 0.33, 0.33, 0.34],
                       [4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 4.10, 4.11, 4.12, 4.13, 4.14, 4.15, 0.33, 0.33, 0.34]])
    costs = np.array([10, 20, 30, 40])
    return Population(mock_opt_params, mock_ga_params, values, costs)

def test_initialization(mock_population):
    assert mock_population.values.shape == (mock_num_members, mock_num_materials * mock_num_properties)
    assert mock_population.costs.shape  == (mock_num_members,)

def test_get_unique_designs(mock_population, mocker):
    mocker.patch.object(mock_population, 'set_costs', return_value=None)
    unique_designs, unique_costs = mock_population.get_unique_designs()
    assert unique_designs.shape[0] == len(np.unique(mock_population.costs))
    assert unique_costs.shape[0]   == len(np.unique(mock_population.costs))

def test_get_effective_properties(mock_population, mocker):
    result = mock_population.get_effective_properties()
    assert result.shape == (mock_num_members, mock_num_properties - 1) # subtract one for volume fraction

def test_set_random_values(mock_population):
    lower_bounds = {
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
    upper_bounds = {
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

    mock_population.set_random_values(lower_bounds, upper_bounds)
    assert mock_population.values.shape == (mock_num_members, mock_num_materials * mock_num_properties)

def test_set_costs(mock_population):
    mock_population.set_costs()
    assert np.all(mock_population.costs != None)

def test_set_order_by_costs(mock_population):
    sorted_indices = np.array([3, 2, 1, 0])
    mock_population.set_order_by_costs(sorted_indices)
    assert np.array_equal(mock_population.values[0][0], 4.1)
