"""test_match_finder.py."""
import pytest

from unittest import mock
from math import comb

from hashin_shtrikman_mp.core.match_finder import MatchFinder
from hashin_shtrikman_mp.core.genetic_algorithm import GeneticAlgorithmResult, Population


@pytest.fixture
def mock_consolidated_dict():
    return {
        "material_id": ["mp-1", "mp-2", "mp-3"],
        "elec_cond_300k_low_doping": [1.0, 2.5, 4.0],
        "therm_cond_300k_low_doping": [1.0, 2.5, 4.0],
        "bulk_modulus": [1.0, 2.5, 4.0],
        "shear_modulus": [1.0, 2.5, 4.0],
        "universal_anisotropy": [1.0, 2.5, 4.0]
    }


@pytest.fixture
def mock_best_designs_dict():
    return {
        "mat1": {
            "carrier-transport": {"elec_cond_300k_low_doping": [1.0], "therm_cond_300k_low_doping": [100]},
            "elastic": {"bulk_modulus": [50], "shear_modulus": [25], "universal_anisotropy": [0.9]}
        },
        "mat2": {
            "carrier-transport": {"elec_cond_300k_low_doping": [1.5], "therm_cond_300k_low_doping": [120]},
            "elastic": {"bulk_modulus": [60], "shear_modulus": [30], "universal_anisotropy": [1.0]}
        }
    }


@pytest.fixture
def mock_ga_result():

    # Mock the GA result
    mock_ga_result = mock.MagicMock(spec=GeneticAlgorithmResult)

    # Mock the required attributes for GeneticAlgorithmResult
    mock_ga_result.ga_params = {"some_param": "some_value"}
    mock_ga_result.algo_parameters = {"some_param": "some_value"}

    # Prepare the mock optimization params
    mock_optimization_params = mock.MagicMock()
    mock_optimization_params.num_materials = 3

    mock_optimization_params.property_categories = ["Category1", "Category2"]
    mock_optimization_params.property_docs = {
        "Category1": ["prop1", "prop2", "prop3"],
        "Category2": ["prop4", "prop5"]
    }

    # Set the mocked optimization parameters
    mock_ga_result.optimization_params = mock_optimization_params

    # Mock the optimized_population.get_unique_designs method
    mock_population = mock.MagicMock(spec=Population)
    mock_population.get_unique_designs.return_value = (
        [[1.1, 2.1, 3.1,
        1.2, 2.2, 3.2,
        1.3, 2.3, 3.3,
        1.4, 2.4, 3.4,
        1.5, 2.5, 3.5, 
        0.3, 0.3, 0.4]], # unique_members (mocked material properties)
        [10.0]           # unique_costs (mocked)
    )
    mock_ga_result.optimized_population = mock_population
    mock_ga_result.final_population = mock_population
    
    return mock_ga_result


@pytest.fixture
def mock_match_finder(mock_ga_result, mock_best_designs_dict, mock_consolidated_dict):
    mock_match_finder = MatchFinder(mock_ga_result)
    return mock_match_finder


# Verify the structure and contents of the best_designs_dict
def test_get_dict_of_best_designs(mock_match_finder):

    best_designs_dict = mock_match_finder.get_dict_of_best_designs()

    # Check if the dictionary contains the expected structure
    assert "mat1" in best_designs_dict
    assert "mat2" in best_designs_dict
    assert "mat3" in best_designs_dict

    # Check that each material has the right categories and properties
    for mat in best_designs_dict.values():
        assert "Category1" in mat
        assert "Category2" in mat
        assert "prop1" in mat["Category1"]
        assert "prop2" in mat["Category1"]
        assert "prop3" in mat["Category1"]
        assert "prop4" in mat["Category2"]
        assert "prop5" in mat["Category2"]

    # Check that the values are lists (because the properties are appended)
    assert isinstance(best_designs_dict["mat1"]["Category1"]["prop1"], list)
    assert isinstance(best_designs_dict["mat2"]["Category2"]["prop4"], list)

    # Check if the number of entries in the list is as expected
    assert len(best_designs_dict["mat1"]["Category1"]["prop1"]) == 1
    assert len(best_designs_dict["mat1"]["Category2"]["prop4"]) == 1


# Test edge case with empty unique_costs or unique_members
def test_empty_unique_members_or_costs(mock_match_finder):
    mock_match_finder.optimized_population.get_unique_designs.return_value = ([], [])
    best_designs_dict = mock_match_finder.get_dict_of_best_designs()

    # Ensure that the result is an empty dictionary if no unique designs exist
    for mat in best_designs_dict.values():
        for category in mat.values():
            for prop in category.values():
                assert len(prop) == 0


@mock.patch("builtins.open", new_callable=mock.mock_open)
@mock.patch.object(MatchFinder, "generate_consolidated_dict")
@mock.patch.object(MatchFinder, "get_dict_of_best_designs")
def test_get_material_matches(mock_get_dict_of_best_designs, mock_generate_consolidated_dict, mock_open, mock_match_finder, mock_consolidated_dict, mock_best_designs_dict):

    # Mocking the return values
    mock_generate_consolidated_dict.return_value = mock_consolidated_dict
    mock_get_dict_of_best_designs.return_value = mock_best_designs_dict

    # Call the method under test
    result = mock_match_finder.get_material_matches(overall_bounds_dict={}, consolidated_dict=mock_consolidated_dict, threshold=1.0)

    # Check that the result is as expected
    expected_result = {
        "mat1": [{"mp-1": {"elec_cond": 1.0, "therm_cond": 1.0, "bulk_modulus": 1.0, "shear_modulus": 1.0, "universal_anisotropy": 1.0}}],
        "mat2": [{"mp-1": {"elec_cond": 1.0, "therm_cond": 1.0, "bulk_modulus": 1.0, "shear_modulus": 1.0, "universal_anisotropy": 1.0}}]
    }
    assert result == expected_result


@mock.patch("builtins.open", new_callable=mock.mock_open)
@mock.patch.object(MatchFinder, "generate_consolidated_dict")
@mock.patch.object(MatchFinder, "get_dict_of_best_designs")
def test_no_matches(mock_get_dict_of_best_designs, mock_generate_consolidated_dict, mock_open, mock_match_finder, mock_consolidated_dict, mock_best_designs_dict):

    # Test when no matches are found (threshold too low)
    mock_generate_consolidated_dict.return_value = mock_consolidated_dict
    mock_get_dict_of_best_designs.return_value = mock_best_designs_dict

    # Call the method under test with a threshold that ensures no match
    result = mock_match_finder.get_material_matches(overall_bounds_dict={}, consolidated_dict=mock_consolidated_dict, threshold=0.0)

    # Assert that no matches are found
    assert result == {}


@mock.patch("builtins.open", new_callable=mock.mock_open)
@mock.patch.object(MatchFinder, "generate_consolidated_dict")
@mock.patch.object(MatchFinder, "get_dict_of_best_designs")
def test_threshold_effect(mock_get_dict_of_best_designs, mock_generate_consolidated_dict, mock_open, mock_match_finder, mock_consolidated_dict, mock_best_designs_dict):

    # Test with different threshold values

    # Mocking the return values
    mock_generate_consolidated_dict.return_value = mock_consolidated_dict
    mock_get_dict_of_best_designs.return_value = mock_best_designs_dict

    # Test for a threshold where some matches should be found
    result_some = mock_match_finder.get_material_matches(overall_bounds_dict={}, consolidated_dict=mock_consolidated_dict, threshold=1.0)
    assert result_some != {}

    # Test for a higher threshold where no matches should be found
    result_none = mock_match_finder.get_material_matches(overall_bounds_dict={}, consolidated_dict=mock_consolidated_dict, threshold=0.0)
    assert result_none == {}


@pytest.mark.parametrize("num_materials, num_fractions, expected_combos", [
    (3, 30, comb(3 + 30 - 1, 30)),  # Standard case: 3 materials, 30 fractions
    (3, 10, comb(3 + 10 - 1, 10)),  # Custom case: 3 materials, 10 fractions
    (1, 30, 1),                     # Edge case: 1 material (only one possible combination)
])
def test_get_all_possible_vol_frac_combos(num_materials, num_fractions, expected_combos):
    
    # Mock GeneticAlgorithmResult and MatchFinder
    mock_ga_result = mock.MagicMock()
    match_finder = MatchFinder(mock_ga_result)
    match_finder.optimization_params.num_materials = num_materials
    result = match_finder.get_all_possible_vol_frac_combos(num_fractions)

    # Check the number of combinations generated
    assert len(result) == expected_combos

    # Verify that each combination sums to approximately 1.0
    for combo in result:
        assert pytest.approx(sum(combo), rel=1e-6) == 1.0
