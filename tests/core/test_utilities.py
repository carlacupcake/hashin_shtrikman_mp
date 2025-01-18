"""test_utilities.py."""
import pytest
import types
from unittest.mock import patch, mock_open
import yaml

# Mock data for HEADERS_PATH
MOCK_HEADERS = """
Per Material:
  Category1:
    prop1: Property 1
    prop2: Property 2
    prop3: Property 3
  Category2:
    prop1: Property 4
    prop2: Property 5
Common:
  - Common Property 1
"""

from hashin_shtrikman_mp.core.utilities import *

@pytest.fixture(scope="module")
def mock_headers():
    """
    Fixture for mocking the content of the headers YAML file.
    """
    return MOCK_HEADERS

@pytest.fixture
def mock_open_and_yaml(mock_headers):
    """
    Fixture for mocking the `open` and `yaml.safe_load` functions.
    """
    with patch("builtins.open", mock_open(read_data=mock_headers)) as mock_file:
        with patch("yaml.safe_load", side_effect=yaml.safe_load) as mock_yaml:
            yield mock_file, mock_yaml

# Mocking the 'loadfn' function and 'PROPERTY_CATEGORIES' variable
@pytest.fixture
def mock_loadfn():
    with patch('hashin_shtrikman_mp.core.utilities.loadfn') as mock_loadfn:
        yield mock_loadfn

@pytest.fixture
def mock_property_categories():
    with patch('hashin_shtrikman_mp.core.utilities.PROPERTY_CATEGORIES', 'mocked_path.yaml'):
        yield

# Mock for load_property_docs to return predefined data
@pytest.fixture
def mock_load_property_docs():
    with patch('hashin_shtrikman_mp.core.utilities.load_property_docs') as mock_load_property_docs:
        # Predefined mock data for property_docs
        mock_load_property_docs.return_value = {
            "Category1": ["prop1", "prop2", "prop3"],
            "Category2": ["prop4", "prop5"]
        }
        yield mock_load_property_docs

def test_headers_with_mpids(mock_open_and_yaml):
    mock_file, mock_yaml = mock_open_and_yaml

    num_materials = 3
    property_categories = ["Category1", "Category2"]
    include_mpids = True

    expected_headers = [
        "Material 1 MP-ID", "Material 2 MP-ID", "Material 3 MP-ID",
        "Phase 1 Property 1", "Phase 2 Property 1", "Phase 3 Property 1",
        "Phase 1 Property 2", "Phase 2 Property 2", "Phase 3 Property 2",
        "Phase 1 Property 3", "Phase 2 Property 3", "Phase 3 Property 3",
        "Phase 1 Property 4", "Phase 2 Property 4", "Phase 3 Property 4",
        "Phase 1 Property 5", "Phase 2 Property 5", "Phase 3 Property 5",
        "Phase 1 Volume Fraction", "Phase 2 Volume Fraction", "Phase 3 Volume Fraction",
        "Common Property 1"
    ]

    headers = get_headers(num_materials, property_categories, include_mpids)
    assert headers == expected_headers

def test_headers_without_mpids(mock_open_and_yaml):
    mock_file, mock_yaml = mock_open_and_yaml

    num_materials = 3
    property_categories = ["Category1", "Category2"]
    include_mpids = False

    expected_headers = [
        "Phase 1 Property 1", "Phase 2 Property 1", "Phase 3 Property 1",
        "Phase 1 Property 2", "Phase 2 Property 2", "Phase 3 Property 2",
        "Phase 1 Property 3", "Phase 2 Property 3", "Phase 3 Property 3",
        "Phase 1 Property 4", "Phase 2 Property 4", "Phase 3 Property 4",
        "Phase 1 Property 5", "Phase 2 Property 5", "Phase 3 Property 5",
        "Phase 1 Volume Fraction", "Phase 2 Volume Fraction", "Phase 3 Volume Fraction",
        "Common Property 1"
    ]

    headers = get_headers(num_materials, property_categories, include_mpids)
    assert headers == expected_headers

def test_empty_property_categories(mock_open_and_yaml):
    mock_file, mock_yaml = mock_open_and_yaml

    num_materials = 3
    property_categories = []
    include_mpids = False

    expected_headers = [
        "Phase 1 Volume Fraction", "Phase 2 Volume Fraction", "Phase 3 Volume Fraction",
        "Common Property 1"
    ]

    headers = get_headers(num_materials, property_categories, include_mpids)
    assert headers == expected_headers

def test_invalid_property_category(mock_open_and_yaml):
    mock_file, mock_yaml = mock_open_and_yaml

    num_materials = 3
    property_categories = ["InvalidCategory"]
    include_mpids = False

    expected_headers = [
        "Phase 1 Volume Fraction", "Phase 2 Volume Fraction", "Phase 3 Volume Fraction",
        "Common Property 1"
    ]

    headers = get_headers(num_materials, property_categories, include_mpids)
    assert headers == expected_headers

def test_load_property_docs(mock_loadfn, mock_property_categories):
    # Define the mock return value for `loadfn`
    mock_loadfn.return_value = {"prop1": "description", "prop2": "description"}

    # Call the function under test
    result = load_property_docs()

    # Assert that loadfn was called with the mocked PROPERTY_CATEGORIES path
    mock_loadfn.assert_called_once_with('mocked_path.yaml')

    # Assert that the result matches the expected mocked data
    expected_result = {"prop1": "description", "prop2": "description"}
    assert result == expected_result

# Test when user input matches some property categories
def test_load_property_categories_match(mock_load_property_docs):
    user_input = {
        "entity1": ["prop1", "prop4"],
        "entity2": ["prop3"]
    }

    # Expected output based on the mock data
    expected_categories = ["Category1", "Category2"]
    expected_docs = {
        "Category1": ["prop1", "prop2", "prop3"],
        "Category2": ["prop4", "prop5"]
    }

    property_categories, property_docs = load_property_categories(user_input)

    # Assert the categories and docs are correct
    assert property_categories == expected_categories
    assert property_docs == expected_docs


# Test when user input does not match any property categories
def test_load_property_categories_no_match(mock_load_property_docs):
    user_input = {
        "entity1": ["prop6", "prop7"]
    }

    # Expected output when no match is found
    expected_categories = []
    expected_docs = {
        "Category1": ["prop1", "prop2", "prop3"],
        "Category2": ["prop4", "prop5"]
    }

    property_categories, property_docs = load_property_categories(user_input)

    # Assert no categories are found
    assert property_categories == expected_categories
    assert property_docs == expected_docs

# Test when user input is None
def test_load_property_categories_none(mock_load_property_docs):
    user_input = None  # This is the default behavior

    # Expected output
    expected_categories = []
    expected_docs = {
        "Category1": ["prop1", "prop2", "prop3"],
        "Category2": ["prop4", "prop5"]
    }

    property_categories, property_docs = load_property_categories(user_input)

    # Assert the categories and docs are correct
    assert property_categories == expected_categories
    assert property_docs == expected_docs


# Test when user input contains duplicate properties
def test_load_property_categories_duplicates(mock_load_property_docs):
    user_input = {
        "entity1": ["prop1", "prop1", "prop2"]
    }

    # Expected output based on the mock data (duplicates should be removed)
    expected_categories = ["Category1"]
    expected_docs = {
        "Category1": ["prop1", "prop2", "prop3"],
        "Category2": ["prop4", "prop5"]
    }

    property_categories, property_docs = load_property_categories(user_input)

    # Assert the categories and docs are correct
    assert property_categories == expected_categories
    assert property_docs == expected_docs

# Test basic case with simple string formulas
def test_compile_formulas_basic():
    formulas_dict = {
        "formula_1": "A1 + An",
        "formula_2": "phase_1 + phase_n_vol_frac"
    }

    compiled_formulas = compile_formulas(formulas_dict)

    # Check if the formulas are compiled as expected
    assert isinstance(compiled_formulas["formula_1"], types.CodeType)
    assert isinstance(compiled_formulas["formula_2"], types.CodeType)

    # Evaluate the compiled formula with example values
    A1 = 1
    An = 2
    phase_1 = 3
    phase_n_vol_frac = 4
    result_1 = eval(compiled_formulas["formula_1"])
    result_2 = eval(compiled_formulas["formula_2"])

    # Check the results match the expected outcomes
    assert result_1 == A1 + An
    assert result_2 == phase_1 + phase_n_vol_frac


# Test case with nested dictionaries
def test_compile_formulas_nested():
    formulas_dict = {
        "outer_formula": {
            "inner_formula_1": "A1 + An",
            "inner_formula_2": "phase_1 + phase_n_vol_frac"
        }
    }

    compiled_formulas = compile_formulas(formulas_dict)

    # Check if the nested formulas are compiled correctly
    assert isinstance(compiled_formulas["outer_formula"], dict)
    assert isinstance(compiled_formulas["outer_formula"]["inner_formula_1"], types.CodeType)
    assert isinstance(compiled_formulas["outer_formula"]["inner_formula_2"], types.CodeType)

    # Evaluate the nested compiled formulas
    A1 = 1
    An = 2
    phase_1 = 3
    phase_n_vol_frac = 4
    result_1 = eval(compiled_formulas["outer_formula"]["inner_formula_1"])
    result_2 = eval(compiled_formulas["outer_formula"]["inner_formula_2"])

    # Check the results match the expected outcomes
    assert result_1 == A1 + An
    assert result_2 == phase_1 + phase_n_vol_frac


# Test case where formulas_dict is None
def test_compile_formulas_none():
    compiled_formulas = compile_formulas(None)
    # Check that an empty dictionary is returned
    assert compiled_formulas == {}


# Test case where formulas are empty or have invalid syntax
def test_compile_formulas_invalid():
    formulas_dict = {
        "invalid_formula_1": "A1 + ",
        "invalid_formula_2": "phase_1 + phase_n_vol_frac + ",
        "empty_formula": ""
    }

    # Test invalid formulas; they should raise SyntaxError when compiled
    with pytest.raises(SyntaxError):
        compile_formulas(formulas_dict)
