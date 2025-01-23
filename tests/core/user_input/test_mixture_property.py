"""test_mixture_property.py"""
from hashin_shtrikman_mp.core.user_input.mixture_property import MixtureProperty
from pydantic import ValidationError
import pytest

# Test valid MixtureProperty initialization
def test_valid_mixture_property():
    mixture = MixtureProperty(prop="e_ij_max", desired_prop=0.5)
    assert mixture.prop == "e_ij_max"
    assert mixture.desired_prop == 0.5

# Test missing required fields
def test_missing_fields():
    with pytest.raises(ValidationError):
        MixtureProperty(prop="e_ij_max")  # Missing 'desired_prop'

    with pytest.raises(ValidationError):
        MixtureProperty(desired_prop=0.5)  # Missing 'prop'

# Test empty string for prop
def test_empty_prop():
    with pytest.raises(ValidationError):
        MixtureProperty(prop="", desired_prop=0.5)

# Test invalid desired_prop type
def test_invalid_desired_prop_type():
    with pytest.raises(ValidationError):
        MixtureProperty(prop="invalid", desired_prop="high")
