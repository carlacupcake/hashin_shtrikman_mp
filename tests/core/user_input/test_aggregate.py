"""test_aggregate.py"""
import pytest
from pydantic import ValidationError

from hashin_shtrikman_mp.core.user_input.material_property import MaterialProperty
from hashin_shtrikman_mp.core.user_input.material import Material
from hashin_shtrikman_mp.core.user_input.mixture import Mixture
from hashin_shtrikman_mp.core.user_input.aggregate import Aggregate

# Test valid Aggregate initialization
def test_valid_aggregate():
    properties_mat_1 = [
        MaterialProperty(prop="elec_cond_300k_low_doping", upper_bound=1.0, lower_bound=0.0),
        MaterialProperty(prop="therm_cond_300k_low_doping", upper_bound=3.0, lower_bound=2.0),
    ]
    properties_mat_2 = [
        MaterialProperty(prop="elec_cond_300k_low_doping", upper_bound=2.0, lower_bound=0.5),
    ]
    
    mat_1 = Material(name="mat_1", properties=properties_mat_1)
    mat_2 = Material(name="mat_2", properties=properties_mat_2)
    mixture = Mixture(name="mixture", properties=[])  # Mixture is ignored in bounds calculation
    aggregate = Aggregate(name="aggregate", components=[mat_1, mat_2, mixture])
    
    bounds_dict = aggregate.get_bounds_dict()
    expected_output = {
        "elec_cond_300k_low_doping": {"upper_bound": 2.0, "lower_bound": 0.0},
        "therm_cond_300k_low_doping": {"upper_bound": 3.0, "lower_bound": 2.0},
    }
    
    assert bounds_dict == expected_output

# Test Aggregate with empty components list
def test_empty_aggregate():
    aggregate = Aggregate(name="empty_aggregate", components=[])
    assert aggregate.get_bounds_dict() == {}

# Test Aggregate with invalid component type
def test_invalid_component_type():
    with pytest.raises(ValidationError):
        Aggregate(name="invalid_aggregate", components=["not a material or mixture"])

# Test Aggregate with only mixtures (should return empty bounds_dict)
def test_aggregate_with_only_mixtures():
    mixture_1 = Mixture(name="mixture_1", properties=[])
    mixture_2 = Mixture(name="mixture_2", properties=[])
    aggregate = Aggregate(name="mixture_only_aggregate", components=[mixture_1, mixture_2])
    
    assert aggregate.get_bounds_dict() == {}
