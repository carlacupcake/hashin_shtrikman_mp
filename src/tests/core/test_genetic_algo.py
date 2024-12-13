import pytest
from hashin_shtrikman_mp.core.genetic_algo import GAParams
from pydantic import ValidationError


# Test for default values
def test_default_values():
    # Creating an instance with no arguments (will use the default values)
    ga_params = GAParams()

    # Check if default values are correctly set
    assert ga_params.num_parents == 10
    assert ga_params.num_kids == 10
    assert ga_params.num_generations == 100
    assert ga_params.num_members == 200
    assert ga_params.mixing_param == 0.5
    assert ga_params.tolerance == 1.0
    assert ga_params.weight_eff_prop == 1.0
    assert ga_params.weight_conc_factor == 1.0

# Test for custom values
def test_custom_values():
    # Customizing the parameters
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

    # Check if the custom values are correctly set
    assert ga_params.num_parents == 20
    assert ga_params.num_kids == 15
    assert ga_params.num_generations == 50
    assert ga_params.num_members == 100
    assert ga_params.mixing_param == 0.7
    assert ga_params.tolerance == 0.8
    assert ga_params.weight_eff_prop == 1.5
    assert ga_params.weight_conc_factor == 2.0

# Test for validation errors (invalid values)
def test_invalid_values():
    # Test invalid value for num_parents (should be a positive integer)
    with pytest.raises(ValidationError):
        GAParams(num_parents=-1)

    # Test invalid value for num_kids (should be a positive integer)
    with pytest.raises(ValidationError):
        GAParams(num_kids=-1)

    # Test invalid value for mixing_param (should be a float between 0 and 1)
    with pytest.raises(ValidationError):
        GAParams(mixing_param=1.5)

    # Test invalid value for tolerance (should be a positive float)
    with pytest.raises(ValidationError):
        GAParams(tolerance=-0.1)

    # Test invalid value for weight_eff_prop (should be a positive float)
    with pytest.raises(ValidationError):
        GAParams(weight_eff_prop=-1.0)

    # Test invalid value for weight_conc_factor (should be a positive float)
    with pytest.raises(ValidationError):
        GAParams(weight_conc_factor=-1.0)

# Test for boundary values
def test_boundary_values():
    # Test mixing_param on boundary (should be a float between 0 and 1)
    ga_params_min = GAParams(mixing_param=0.0)
    ga_params_max = GAParams(mixing_param=1.0)

    assert ga_params_min.mixing_param == 0.0
    assert ga_params_max.mixing_param == 1.0

    # Test tolerance at the boundary
    ga_params_tolerance = GAParams(tolerance=0.0)
    assert ga_params_tolerance.tolerance == 0.0

# Test for missing required fields (should raise a ValidationError)
def test_missing_required_fields():
    with pytest.raises(ValidationError):
        # Missing some required fields should raise an error
        GAParams(num_parents=None)
