"""test_optimization_params.py"""
from hashin_shtrikman_mp.core.genetic_algorithm.optimization_params import OptimizationParams
import pytest

mock_num_materials  = 3
mock_num_properties = 6

@pytest.fixture
def mock_user_input():
    return  {
        'mat_1': {
            'elec_cond_300k_low_doping': {
                'upper_bound': 1.0, 
                'lower_bound': 0.0
            }, 
            'therm_cond_300k_low_doping': {
                'upper_bound': 1.0, 
                'lower_bound': 0.0
            }, 
            'bulk_modulus': {
                'upper_bound': 1.0, 
                'lower_bound': 0.0
            }, 
            'shear_modulus': {
                'upper_bound': 1.0, 
                'lower_bound': 0.0
            }, 
            'universal_anisotropy': {
                'upper_bound': 1.0, 
                'lower_bound': 0.0
            }
        }, 
        'mat_2': {
            'elec_cond_300k_low_doping': {
                'upper_bound': 1.0, 
                'lower_bound': 0.0
            }, 
            'therm_cond_300k_low_doping': {
                'upper_bound': 1.0, 
                'lower_bound': 0.0
            }, 
            'bulk_modulus': {
                'upper_bound': 1.0, 
                'lower_bound': 0.0
            }, 
            'shear_modulus': {
                'upper_bound': 1.0, 
                'lower_bound': 0.0
            }, 
            'universal_anisotropy': {
                'upper_bound': 1.0, 
                'lower_bound': 0.0
            }
        }, 
        'mat_3': {
            'elec_cond_300k_low_doping': {
                'upper_bound': 1.0, 
                'lower_bound': 0.0
            }, 
            'therm_cond_300k_low_doping': {
                'upper_bound': 1.0, 
                'lower_bound': 0.0
            }, 
            'bulk_modulus': {
                'upper_bound': 1.0, 
                'lower_bound': 0.0
            }, 
            'shear_modulus': {
                'upper_bound': 1.0, 
                'lower_bound': 0.0
            }, 
            'universal_anisotropy': {
                'upper_bound': 1.0, 
                'lower_bound': 0.0
            }
        }, 
        'mixture': {
            'elec_cond_300k_low_doping': {
                'desired_prop': 0.5
            }, 
            'therm_cond_300k_low_doping': {
                'desired_prop': 0.5
            }, 
            'bulk_modulus': {
                'desired_prop': 0.5
            }, 
            'shear_modulus': {
                'desired_prop': 0.5
            }, 
            'universal_anisotropy': {
                'desired_prop': 0.5
            }
        }
    }

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

def test_from_user_input(mock_user_input, mock_property_docs, monkeypatch):
    monkeypatch.setattr("hashin_shtrikman_mp.core.genetic_algorithm.optimization_params.load_property_categories",
                        lambda user_input: (["carrier-transport", "elastic"], mock_property_docs))
    params = OptimizationParams.from_user_input(mock_user_input)
    assert params.num_materials == mock_num_materials
    assert params.num_properties == mock_num_properties
    assert params.desired_props == {"carrier-transport": [0.5, 0.5], "elastic": [0.5, 0.5, 0.5]}

def test_get_num_properties_from_desired_props():
    desired_props = {"carrier-transport": [0.5, 0.5], "elastic": [0.5, 0.5, 0.5]}
    result = OptimizationParams.get_num_properties_from_desired_props(desired_props)
    assert result == mock_num_properties

def test_get_bounds_from_user_input(mock_user_input, mock_property_docs):
    lower_bounds = OptimizationParams.get_bounds_from_user_input(mock_user_input, "lower_bound", mock_property_docs, mock_num_materials)
    assert lower_bounds["mat_1"]["carrier-transport"] == [0.0, 0.0]
    assert lower_bounds["mat_1"]["elastic"] == [0.0, 0.0, 0.0]
    assert lower_bounds["volume-fractions"] == [0.01, 0.01, 0.01]

def test_get_desired_props_from_user_input(mock_user_input, mock_property_docs):
    desired_props = OptimizationParams.get_desired_props_from_user_input(mock_user_input, ["carrier-transport", "elastic"], mock_property_docs)
    assert desired_props == {"carrier-transport": [0.5, 0.5], "elastic": [0.5, 0.5, 0.5]}

def test_get_elastic_idx_from_user_input():
    upper_bounds = {
        "mat_1": {"elastic": [1.0, 1.0]},
        "volume-fractions": [0.99]
    }
    indices = OptimizationParams.get_elastic_idx_from_user_input(upper_bounds, ["elastic"])
    assert indices == [0, 1]
