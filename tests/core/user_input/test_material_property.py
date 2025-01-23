"""test_material_property.py"""
from hashin_shtrikman_mp.core.user_input.material_property import MaterialProperty
import pytest

# Test valid data input
def test_valid_data():
    material = MaterialProperty(prop="e_ij_max", upper_bound=1.0, lower_bound=0.0)

    assert material.prop == "e_ij_max"
    assert material.upper_bound == 1.0
    assert material.lower_bound == 0.0


# Test invalid data (e.g., lower_bound greater than upper_bound)
def test_invalid_data():
    with pytest.raises(ValueError):
        MaterialProperty(prop="e_ij_max", upper_bound=0.0, lower_bound=1.0)


# Test missing required fields
def test_missing_fields():
    with pytest.raises(ValueError):
        MaterialProperty(prop="e_ij_max", upper_bound=1.0)
