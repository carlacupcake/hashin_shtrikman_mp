"""test_material.py"""
import pytest

from pydantic import ValidationError

from hashin_shtrikman_mp.core.user_input.material_property import MaterialProperty
from hashin_shtrikman_mp.core.user_input.material import Material


# Test valid Material initialization
def test_valid_material():
    properties = [
        MaterialProperty(prop="elec_cond_300k_low_doping",  upper_bound=1.0, lower_bound=0.0),
        MaterialProperty(prop="therm_cond_300k_low_doping", upper_bound=1.0, lower_bound=0.0),
    ]
    material = Material(name="mat_1", properties=properties)

    assert material.name                      == "mat_1"
    assert len(material.properties)           == 2
    assert material.properties[0].prop        == "elec_cond_300k_low_doping"
    assert material.properties[1].upper_bound == 1.0


# Test invalid MaterialProperty inside Material
def test_invalid_material_property():
    with pytest.raises(ValidationError):
        Material(
            name="mat_2",
            properties=[
                {"prop": "e_ij_max", "upper_bound": 0.0, "lower_bound": 1.0} # invalid bounds
            ],
        )


# Test Material with missing properties
def test_missing_properties():
    with pytest.raises(ValidationError):
        Material(name="mat_1")  # Missing required 'properties' field


# Test Material with incorrect property type (should be a list of MaterialProperty)
def test_invalid_properties_type():
    with pytest.raises(ValidationError):
        Material(name="mat_1", properties="not a list")


# Test Material with empty properties list
def test_empty_properties_list():
    material = Material(name="mat_1", properties=[])
    assert material.name == "mat_1"
    assert material.properties == []


# Test custom_dict method for correct transformation
def test_custom_dict():
    properties = [
        MaterialProperty(prop="elec_cond_300k_low_doping",  upper_bound=1.0, lower_bound=0.0),
        MaterialProperty(prop="therm_cond_300k_low_doping", upper_bound=3.0, lower_bound=2.0),
    ]
    material = Material(name="mat_1", properties=properties)
    
    expected_output = {
        "mat_1": {
            "elec_cond_300k_low_doping":  {"upper_bound": 1.0, "lower_bound": 0.0},
            "therm_cond_300k_low_doping": {"upper_bound": 3.0, "lower_bound": 2.0},
        }
    }
    
    assert material.custom_dict() == expected_output
