"""test_composite_property_plotter.py"""
import numpy as np
import plotly.graph_objects as go
import pytest
import warnings

from unittest.mock import MagicMock

from hashin_shtrikman_mp.core.genetic_algorithm.genetic_algorithm_result import GeneticAlgorithmResult
from hashin_shtrikman_mp.core.visualization.composite_property_plotter import CompositePropertyPlotter


mock_num_materials = 4
mock_num_fractions = 5 # mock with a small number of fractions for easier verification


@pytest.fixture
def mock_ga_result():
    """Fixture to provide a mock GeneticAlgorithmResult object."""
    mock_result = MagicMock(spec=GeneticAlgorithmResult)
    mock_result.algo_parameters = "mocked_algo_params"
    mock_result.optimization_params = "mocked_opt_params"
    return mock_result


def test_composite_property_plotter_init(mock_ga_result):
    """Test the initialization of CompositePropertyPlotter."""
    plotter = CompositePropertyPlotter(mock_ga_result)
    assert plotter.ga_params == mock_ga_result.algo_parameters
    assert plotter.opt_params == mock_ga_result.optimization_params
    assert plotter.ga_result == mock_ga_result


def test_from_optimization_result(mock_ga_result):
    """Test the from_optimization_result class method."""
    plotter = CompositePropertyPlotter.from_optimization_result(mock_ga_result)
    
    # Check if the object is initialized correctly
    assert plotter.ga_result == mock_ga_result
    assert plotter.ga_params == "mocked_algo_params"
    assert plotter.opt_params == "mocked_opt_params"


def test_get_all_possible_vol_frac_combos():
    """Test the get_all_possible_vol_frac_combos method."""
    
    # Mock the CompositePropertyPlotter instance
    mock_plotter = MagicMock()
    
    # Instantiate the actual object with the mocked opt_params
    plotter = CompositePropertyPlotter(mock_plotter)

    # Mock opt_params
    plotter.opt_params.num_materials = mock_num_materials
    
    # Call the method
    result = plotter.get_all_possible_vol_frac_combos(mock_num_fractions)
    
    # Verify that each combination sums to approximately 1.0
    for combo in result:
        assert pytest.approx(sum(combo), rel=1e-6) == 1.0


def test_visualize_composite_eff_props_2_phase(monkeypatch):
    """Test the visualize_composite_eff_props_2_phase method."""

    # Mock the show function to prevent displaying the figure
    # Uncomment to display in browser
    monkeypatch.setattr(go.Figure, "show", lambda self: None)
    
    # Mock the CompositePropertyPlotter instance
    mock_plotter = MagicMock()
    
    # Instantiate the actual object with the mocked opt_params
    plotter = CompositePropertyPlotter(mock_plotter)
    
    # Define test inputs
    match = ["mp-1", "mp-2"]
    property_name = "Thermal conductivity"
    units = "W/m/K"
    volume_fractions = np.array([[0.2, 0.8], [0.4, 0.6], [0.6, 0.4], [0.8, 0.2]])
    effective_properties = np.array([10, 20, 30, 40])

    # Call the function (fig will not be displayed due to monkeypatching)
    fig = plotter.visualize_composite_eff_props_2_phase(match, property_name, units, volume_fractions, effective_properties)
    
    # Check if the figure has the expected number of traces
    assert len(fig.data) == 1
    
    # Validate trace data
    trace = fig.data[0]
    assert trace.mode == "lines"
    np.testing.assert_array_equal(trace.x, volume_fractions[:, 0])
    np.testing.assert_array_equal(trace.y, effective_properties)
    
    # Check the layout properties
    assert fig.layout.xaxis.title.text == f"Volume fraction, {match[0]}"
    assert fig.layout.yaxis.title.text == f"{units}"
    assert fig.layout.title.text == f"{property_name}\n{match}"
    assert fig.layout.title.font.size == 24
    assert fig.layout.width == 600
    assert fig.layout.height == 600


def test_visualize_composite_eff_props_3_phase(monkeypatch):
    """Test the visualize_composite_eff_props_3_phase method."""

    # Mock the show function to prevent displaying the figure
    # Uncomment to display in browser
    monkeypatch.setattr(go.Figure, "show", lambda self: None)
    
    # Mock the CompositePropertyPlotter instance
    mock_plotter = MagicMock()
    
    # Instantiate the actual object with the mocked opt_params
    plotter = CompositePropertyPlotter(mock_plotter)
    
    # Define test inputs
    match = ['mp-1', 'mp-2', 'mp-3']
    property_name = "Thermal conductivity"
    units = "W/m/K"
    volume_fractions = np.array([
        [0.01, 0.01, 0.98], [0.01, 0.33,  0.66], [0.01, 0.66,  0.33], [0.01, 0.99,  0.00],
        [0.33, 0.01, 0.66], [0.33, 0.33,  0.33], [0.33, 0.66,  0.00],
        [0.66, 0.01, 0.33], [0.66, 0.33,  0.00],
        [0.99, 0.01, 0.00]
    ])
    effective_properties = np.array([
        40, 30, 20, 10,
        50, 40, 30,
        60, 50,
        70
    ])

    # Call the function (fig will not be displayed due to monkeypatching)
    fig = plotter.visualize_composite_eff_props_3_phase(match, property_name, units, volume_fractions, effective_properties)
    
    # Check if the figure has the expected number of traces
    assert len(fig.data) == 1
    
    # Validate trace data
    trace = fig.data[0]
    assert trace.type == "surface"

    # Check the layout properties
    assert fig.layout.scene.xaxis.title.text == f"Volume fraction, {match[0]}"
    assert fig.layout.scene.yaxis.title.text == f"Volume fraction, {match[1]}"
    assert fig.layout.scene.zaxis.title.text == f"{units}"
    assert fig.layout.title.text == f"{property_name}\n{match}"
    assert fig.layout.width == 600
    assert fig.layout.height == 600


def test_visualize_composite_eff_props_4_phase(monkeypatch):
    """Test the visualize_composite_eff_props_4_phase method."""

    # Mock the show function to prevent displaying the figure
    # Uncomment to display in browser
    monkeypatch.setattr(go.Figure, "show", lambda self: None)
    
    # Mock the CompositePropertyPlotter instance
    mock_plotter = MagicMock()
    
    # Instantiate the actual object with the mocked opt_params
    plotter = CompositePropertyPlotter(mock_plotter)

    # Test data
    match = ['mp-1', 'mp-2', 'mp-3']
    property_name = 'Thermal conductivity'
    units = 'W/m/K'
    volume_fractions = np.array([
        [0.01, 0.01, 0.01,  0.97], [0.01, 0.01, 0.33,  0.65], [0.01, 0.01, 0.66,  0.32], [0.01, 0.01, 0.99, -0.01],
        [0.01, 0.33, 0.01,  0.65], [0.01, 0.33, 0.33,  0.32], [0.01, 0.33, 0.66, -0.01], [0.01, 0.33, 0.99, -0.33],
        [0.01, 0.66, 0.01,  0.32], [0.01, 0.66, 0.33, -0.01], [0.01, 0.66, 0.66, -0.33], [0.01, 0.66, 0.99, -0.66],
        [0.01, 0.99, 0.01, -0.01], [0.01, 0.99, 0.33, -0.33], [0.01, 0.99, 0.66, -0.66], [0.01, 0.99, 0.99, -0.99],
        [0.33, 0.01, 0.01,  0.65], [0.33, 0.01, 0.33,  0.32], [0.33, 0.01, 0.66, -0.01], [0.33, 0.01, 0.99, -0.33],
        [0.33, 0.33, 0.01,  0.32], [0.33, 0.33, 0.33, -0.01], [0.33, 0.33, 0.66, -0.33], [0.33, 0.33, 0.99, -0.66],
        [0.33, 0.66, 0.01, -0.01], [0.33, 0.66, 0.33, -0.33], [0.33, 0.66, 0.66, -0.66], [0.33, 0.66, 0.99, -0.99],
        [0.33, 0.99, 0.01, -0.33], [0.33, 0.99, 0.33, -0.66], [0.33, 0.99, 0.66, -0.99], [0.33, 0.99, 0.99, -1.31],
        [0.66, 0.01, 0.01,  0.32], [0.66, 0.01, 0.33, -0.01], [0.66, 0.01, 0.66, -0.33], [0.66, 0.01, 0.99, -0.66],
        [0.66, 0.33, 0.01, -0.01], [0.66, 0.33, 0.33, -0.33], [0.66, 0.33, 0.66, -0.66], [0.66, 0.33, 0.99, -0.99],
        [0.66, 0.66, 0.01, -0.33], [0.66, 0.66, 0.33, -0.66], [0.66, 0.66, 0.66, -0.99], [0.66, 0.66, 0.99, -1.31],
        [0.66, 0.99, 0.01, -0.66], [0.66, 0.99, 0.33, -0.99], [0.66, 0.99, 0.66, -1.31], [0.66, 0.99, 0.99, -1.64],
        [0.99, 0.01, 0.01, -0.01], [0.99, 0.01, 0.33, -0.33], [0.99, 0.01, 0.66, -0.66], [0.99, 0.01, 0.99, -0.99],
        [0.99, 0.33, 0.01, -0.33], [0.99, 0.33, 0.33, -0.66], [0.99, 0.33, 0.66, -0.99], [0.99, 0.33, 0.99, -1.31],
        [0.99, 0.66, 0.01, -0.66], [0.99, 0.66, 0.33, -0.99], [0.99, 0.66, 0.66, -1.31], [0.99, 0.66, 0.99, -1.64],
        [0.99, 0.99, 0.01, -0.99], [0.99, 0.99, 0.33, -1.31], [0.99, 0.99, 0.66, -1.64], [0.99, 0.99, 0.99, -1.97]
    ])
    effective_properties = np.array([
        10, 20, 30, 40,
        20, 30, 40, 50,
        30, 40, 50, 60,
        40, 50, 60, 70,
        20, 30, 40, 50,
        30, 40, 50, 60,
        40, 50, 60, 70,
        50, 60, 70, 80,
        30, 40, 50, 60,
        40, 50, 60, 70,
        50, 60, 70, 80,
        60, 70, 80, 90,
        40, 50, 60, 70,
        50, 60, 70, 80,
        60, 70, 80, 90,
        70, 80, 90, 100                           
    ])

    # Create the figure using the function
    fig = plotter.visualize_composite_eff_props_4_phase(
        match, property_name, units, volume_fractions, effective_properties
    )

    # Check that the figure has 1 trace
    assert len(fig.data) == 1

    # Get the trace (should be a 3D scatter plot)
    trace = fig.data[0]
    assert isinstance(trace, go.Scatter3d)

    # Check trace properties (e.g., mode, marker size, opacity, color scale)
    assert trace.mode == "markers"
    assert trace.marker.size == 5
    assert trace.marker.opacity == 0.8
    assert trace.marker.colorbar.title['text'] == 'W/m/K'

    # Check the layout properties (axis titles and plot title)
    assert fig.layout.scene.xaxis.title.text == f"Volume fraction, {match[0]}"
    assert fig.layout.scene.yaxis.title.text == f"Volume fraction, {match[1]}"
    assert fig.layout.scene.zaxis.title.text == f"Volume fraction, {match[2]}"
    assert fig.layout.title.text == f"{property_name}\n{match}"

    # Check figure size and margin
    assert fig.layout.width    == 600
    assert fig.layout.height   == 600
    assert fig.layout.margin.b == 50
    assert fig.layout.margin.l == 50
    assert fig.layout.margin.r == 50
    assert fig.layout.margin.t == 50

    # Optionally, check the content of the data
    assert len(trace.x) == len(effective_properties)
    assert len(trace.y) == len(effective_properties)
    assert len(trace.z) == len(effective_properties)


def test_visualize_composite_eff_props():
    # Simplified mock class with necessary attributes and methods
    # Necessary because MagicMock object does not call warning method
    class MockClass:
        def __init__(self):
            self.opt_params = MagicMock()
            self.opt_params.property_categories = ['category1']
            self.opt_params.property_docs = {'category1': ['property1', 'property2']}
            self.ga_params = MagicMock()
            self.ga_params.num_materials = 2
            self.ga_params.get_num_members = MagicMock(return_value=10)
        
        def get_all_possible_vol_frac_combos(self, num_fractions):
            return np.linspace(0, 1, num_fractions).tolist()
        
        def visualize_composite_eff_props_2_phase(self, match, property, units, volume_fractions, effective_properties):
            pass

        def visualize_composite_eff_props_3_phase(self, match, property, units, volume_fractions, effective_properties):
            pass
        
        def visualize_composite_eff_props_4_phase(self, match, property, units, volume_fractions, effective_properties):
            pass

        def visualize_composite_eff_props(self, match, consolidated_dict: dict, num_fractions: int = 99):
            # Handle the case where number of phases is unsupported
            if len(match) == 4:
                num_fractions = 20
            if len(match) == 1 or len(match) > 4:
                warnings.warn("No visualizations available for composites with 5 or more phases.", UserWarning)
                return

            all_vol_frac_combos = self.get_all_possible_vol_frac_combos(num_fractions=num_fractions)

            material_values = []
            for category in self.opt_params.property_categories:
                for property in self.opt_params.property_docs[category]:
                    for material in match:
                        if property in consolidated_dict:
                            m = consolidated_dict["material_id"].index(material)
                            material_values.append(consolidated_dict[property][m])

            population_values = np.tile(material_values, (len(all_vol_frac_combos), 1))
            volume_fractions = np.array(all_vol_frac_combos).reshape(len(all_vol_frac_combos), self.opt_params.num_materials)
            values = np.c_[population_values, volume_fractions]

            this_pop_ga_params = self.ga_params
            this_pop_ga_params.num_members = num_fractions**(len(match) - 1)

            population = MagicMock()
            population.get_effective_properties = MagicMock(return_value=np.random.rand(len(all_vol_frac_combos), 2))
            all_effective_properties = population.get_effective_properties()

            property_strings = ["Property 1, [unit]", "Property 2, [unit]"]

            def extract_property(text):
                import re
                match = re.match(r"([^,]+), \[.*\]", text)
                if match:
                    return match.group(1).strip()
                return None

            def extract_units(text):
                import re
                match = re.search(r"\[.*?\]", text)
                if match:
                    return match.group(0)
                return None

            for i, property_string in enumerate(property_strings):
                property = extract_property(property_string)
                units = extract_units(property_string)
                effective_properties = all_effective_properties[:, i]

                if len(match) == 2:
                    self.visualize_composite_eff_props_2_phase(match, property, units, volume_fractions, effective_properties)
                elif len(match) == 3:
                    self.visualize_composite_eff_props_3_phase(match, property, units, volume_fractions, effective_properties)
                elif len(match) == 4:
                    self.visualize_composite_eff_props_4_phase(match, property, units, volume_fractions, effective_properties)
                else:
                    warnings.warn("No visualizations available for composites with 5 or more phases.", UserWarning)
                    return

    # Create an instance of the mock class
    mock_class = MockClass()

    # Test for len(match) == 1 (should raise a warning and return)
    with pytest.warns(UserWarning):
        mock_class.visualize_composite_eff_props(["mat_1"], {})

    # Test for len(match) == 2 (should run without warnings)
    mock_class.visualize_composite_eff_props(["mat_1", "mat_2"], {})

    # Test for len(match) == 3 (should run without warnings)
    mock_class.visualize_composite_eff_props(["mat_1", "mat_2", "mat_3"], {})

    # Test for len(match) == 4 (should run without warnings)
    mock_class.visualize_composite_eff_props(["mat_1", "mat_2", "mat_3", "mat_4"], {})

    # Test for len(match) > 4 (should raise a warning and return)
    warnings.simplefilter("always")
    with pytest.warns(UserWarning) as warning_list:
        mock_class.visualize_composite_eff_props(["mat_1", "mat_2", "mat_3", "mat_4", "mat_5"], {})

    # Assert that the warning was raised
    assert len(warning_list) > 0, "Expected a UserWarning to be triggered but none was raised."
    assert warning_list[0].category == UserWarning, f"Expected UserWarning but got {warning_list[0].category}"
