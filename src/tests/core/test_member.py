import numpy as np
import pytest
from hashin_shtrikman_mp.core.genetic_algo import GAParams
from hashin_shtrikman_mp.core.member import Member
from pydantic import ValidationError


# Test for default values
def test_default_values():
    member = Member()

    # Check default values
    assert member.num_materials == 0
    assert member.num_properties == 0
    assert member.values.shape == (0, 1)  # Expecting an empty numpy array with shape (0, 1)
    assert member.property_categories == []
    assert member.property_docs == {}
    assert member.desired_props == {}
    assert member.ga_params is None
    assert member.calc_guide is None


# Test for custom values initialization
def test_custom_values():
    ga_params = GAParams(
        num_parents=20,
        num_kids=15,
        num_generations=50,
        num_members=100,
        mixing_param=0.7,
        tolerance=0.8,
        weight_eff_prop=1.5,
        weight_conc_factor=2.0
    )

    member = Member(
        num_materials=2,
        num_properties=3,
        values=np.array([[1.0], [2.0], [3.0]]),
        property_categories=["elastic", "mechanical"],
        property_docs={"elastic": {"E": "elastic_modulus"}, "mechanical": {"yield_strength": "strength"}},
        desired_props={"elastic": [1.5], "mechanical": [1.0]},
        ga_params=ga_params,
        calc_guide={"effective_props": {"alpha_1": "phase_1 * 1.1"}}  # Example guide for testing
    )

    assert member.num_materials == 2
    assert member.num_properties == 3
    assert np.array_equal(member.values, np.array([[1.0], [2.0], [3.0]]))
    assert member.property_categories == ["elastic", "mechanical"]
    assert member.property_docs == {"elastic": {"E": "elastic_modulus"}, "mechanical": {"yield_strength": "strength"}}
    assert member.desired_props == {"elastic": [1.5], "mechanical": [1.0]}
    assert member.ga_params == ga_params


# Test for validation when invalid values are provided
def test_invalid_values():
    # Test invalid value for num_materials (should be >= 0)
    with pytest.raises(ValidationError):
        Member(num_materials=-10)

    # Test invalid value for num_properties (should be >= 0)
    with pytest.raises(ValidationError):
        Member(num_properties=-10)

    # Test invalid values for values (should be numpy array of appropriate shape)
    with pytest.raises(ValidationError):
        Member(values="invalid_value")


# Test for empty values array initialization
def test_check_and_initialize_arrays():
    # Initialize with empty values
    member = Member(num_properties=3, values=np.empty(0))

    # Check if values are initialized to zeros
    assert np.array_equal(member.values, np.zeros((3, 1)))


# Test for the Config class (allowing arbitrary types)
def test_arbitrary_types_allowed():
    member = Member(num_materials=2, num_properties=3, values=np.array([[1.0], [2.0], [3.0]]))
    assert isinstance(member.values, np.ndarray)  # Ensure that values is an ndarray
