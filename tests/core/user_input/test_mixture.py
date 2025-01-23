"""test_mixture.py"""
import pytest
from pydantic import ValidationError
from hashin_shtrikman_mp.core.user_input.mixture import Mixture
from hashin_shtrikman_mp.core.user_input.mixture_property import MixtureProperty

# Test valid Mixture initialization
def test_valid_mixture():
    properties = [
        MixtureProperty(prop="bulk_modulus",  desired_prop=1.0),
        MixtureProperty(prop="shear_modulus", desired_prop=2.0),
    ]
    mixture = Mixture(name="mixture", properties=properties)

    assert mixture.name == "mixture"
    assert len(mixture.properties) == 2
    assert mixture.properties[0].prop == "bulk_modulus"
    assert mixture.properties[1].desired_prop == 2.0

# Test Mixture's custom_dict method
def test_mixture_custom_dict():
    properties = [
        MixtureProperty(prop="bulk_modulus",  desired_prop=1.0),
        MixtureProperty(prop="shear_modulus", desired_prop=2.0),
    ]
    mixture = Mixture(name="mixture", properties=properties)
    
    expected_dict = {
        "mixture": {
            "bulk_modulus":  {"desired_prop": 1.0},
            "shear_modulus": {"desired_prop": 2.0},
        }
    }
    assert mixture.custom_dict() == expected_dict

# Test missing required fields
def test_missing_fields():
    with pytest.raises(ValidationError):
        Mixture(properties=[])  # Missing 'name'

    with pytest.raises(ValidationError):
        Mixture(name="mixture")  # Missing 'properties'

# Test invalid property type in list
def test_invalid_property_type():
    with pytest.raises(ValidationError):
        Mixture(name="mixture", properties=["invalid"])

# Test empty properties list
def test_empty_properties_list():
    mixture = Mixture(name="mixture", properties=[])
    assert mixture.name == "mixture"
    assert mixture.properties == []

# Test invalid name type
def test_invalid_name_type():
    with pytest.raises(ValidationError):
        Mixture(name=123, properties=[])
