"""test_member.py"""
import numpy as np
import pytest

from pydantic import ValidationError

from hashin_shtrikman_mp.core.genetic_algorithm import GeneticAlgorithmParams
from hashin_shtrikman_mp.core.genetic_algorithm import OptimizationParams, Member


@pytest.fixture(scope="module")
def basic_opt_params():
    return OptimizationParams(
        num_materials=2,
        num_properties=3,
        property_categories=["elastic", "mechanical"],
        property_docs={"elastic": {"E": "elastic_modulus"}, "mechanical": {"yield_strength": "strength"}},
        desired_props={"elastic": [1.5], "mechanical": [1.0]},        
    )


# Test for default values
def test_opt_params_default_values():
    opt_params = OptimizationParams()

    # Check default values
    assert opt_params.num_materials == 0
    assert opt_params.num_properties == 0
    assert opt_params.property_categories == []
    assert opt_params.property_docs == {}
    assert opt_params.desired_props == {}


# Test for custom values initialization
def test_custom_values(basic_opt_params):
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

    member = Member(
        values=np.array([[1.0], [2.0], [3.0]]),
        optimization_params=basic_opt_params,
        ga_params=ga_params
    )

    assert member.opt_params.num_materials == 2
    assert member.opt_params.num_properties == 3
    assert np.array_equal(member.values, np.array([[1.0], [2.0], [3.0]]))
    assert member.opt_params.property_categories == ["elastic", "mechanical"]
    assert member.opt_params.property_docs == {"elastic": {"E": "elastic_modulus"}, "mechanical": {"yield_strength": "strength"}}
    assert member.opt_params.desired_props == {"elastic": [1.5], "mechanical": [1.0]}
    assert member.ga_params == ga_params


# Test for validation when invalid values are provided
@pytest.mark.skip
def test_invalid_values():
    # Test invalid value for num_materials (should be >= 0)
    with pytest.raises(ValidationError):
        OptimizationParams(num_materials=-10)

    # Test invalid value for num_properties (should be >= 0)
    with pytest.raises(ValidationError):
        OptimizationParams(num_properties=-10)

    # Test invalid values for values (should be numpy array of appropriate shape)
    with pytest.raises(ValidationError):
        Member(values="invalid_value")


# Test for empty values array initialization
def test_check_and_initialize_arrays(basic_opt_params):
    # Initialize with empty values
    member = Member(optimization_params=basic_opt_params, ga_params=GeneticAlgorithmParams(), values=np.empty(0))

    # Check if values are initialized to zeros
    assert np.array_equal(member.values, np.zeros((3, 1)))


# Test for the Config class (allowing arbitrary types)
def test_arbitrary_types_allowed(basic_opt_params):
    member = Member(optimization_params=basic_opt_params, ga_params=GeneticAlgorithmParams(), values=np.array([[1.0], [2.0], [3.0]]))
    assert isinstance(member.values, np.ndarray)  # Ensure that values is an ndarray
