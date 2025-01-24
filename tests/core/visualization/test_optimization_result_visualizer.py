"""test_optimization_result_visualizer.py"""
import pytest
import numpy as np

from unittest.mock import MagicMock

from hashin_shtrikman_mp.core.visualization.optimization_result_visualizer import OptimizationResultVisualizer
from hashin_shtrikman_mp.core.genetic_algorithm.genetic_algorithm_result import GeneticAlgorithmResult
from hashin_shtrikman_mp.core.genetic_algorithm.genetic_algorithm_parameters import GeneticAlgorithmParams
from hashin_shtrikman_mp.core.genetic_algorithm.optimization_params import OptimizationParams
from hashin_shtrikman_mp.core.genetic_algorithm.population import Population


mock_num_materials   = 2
mock_num_properties  = 4
mock_num_generations = 5
mock_num_members     = 3
mock_rows = mock_num_members
mock_values = np.array([[1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 0.2, 0.8],
                        [1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 0.5, 0.5],
                        [1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 0.8, 0.2]])
mock_costs = np.array([10, 20, 30]) # for one generation
mock_avg_parent_costs = [100, 99, 98, 97, 96]
mock_lowest_costs     = [90, 80, 70, 60, 50]


@pytest.fixture
def mock_genetic_algorithm_result():

    # Mock the algo_parameters and its attributes
    mock_algo_parameters = GeneticAlgorithmParams()
    mock_algo_parameters.num_generations = mock_num_generations
    mock_algo_parameters.num_members = mock_num_members

    # Mock the opt_parameters if necessary
    mock_opt_parameters = MagicMock(OptimizationParams)
    mock_opt_parameters.num_materials  = mock_num_materials
    mock_opt_parameters.num_properties = mock_num_properties
    mock_opt_parameters.property_categories = ["carrier-transport", "dielectric"]
    mock_opt_parameters.property_docs = {
        "carrier-transport": [
            "elec_cond_300k_low_doping",
            "therm_cond_300k_low_doping"
        ],
        "dielectric": [
            "e_ij_max"
        ]
    }

    # Mock desired_props
    mock_opt_parameters.desired_props = {
        "carrier-transport": [1.60, 1.65],
        "dielectric": [1.7]
    }

    # Mock the final population
    mock_final_population = MagicMock(Population)
    mock_final_population.values = mock_values
    mock_final_population.optimization_params = mock_opt_parameters
    mock_final_population.get_unique_designs.return_value = [mock_values, mock_costs]

    # Mock the ga_result
    ga_result = MagicMock(GeneticAlgorithmResult)
    ga_result.algo_parameters  = mock_algo_parameters
    ga_result.opt_parameters   = mock_opt_parameters
    ga_result.final_population = mock_final_population
    ga_result.lowest_costs     = mock_lowest_costs
    ga_result.avg_parent_costs = mock_avg_parent_costs

    # Mock optimization_params directly on ga_result (in case it's used elsewhere)
    ga_result.optimization_params = mock_opt_parameters

    return ga_result


def test_get_table_of_best_designs(mock_genetic_algorithm_result):
    visualizer = OptimizationResultVisualizer(mock_genetic_algorithm_result)
    table = visualizer.get_table_of_best_designs(rows=mock_rows)
    assert table.shape == (mock_rows, mock_num_properties * mock_num_materials + 1)  # +1 for cost columns


def test_print_table_of_best_designs(mock_genetic_algorithm_result):
    visualizer = OptimizationResultVisualizer(mock_genetic_algorithm_result)
    fig = visualizer.print_table_of_best_designs()
    assert fig is not None  # Check if a figure is returned


def test_plot_optimization_results(mock_genetic_algorithm_result):
    visualizer = OptimizationResultVisualizer(mock_genetic_algorithm_result)
    fig = visualizer.plot_optimization_results()
    assert fig is not None  # Check if a figure is returned


def test_plot_cost_func_contribs(mock_genetic_algorithm_result):
    visualizer = OptimizationResultVisualizer(mock_genetic_algorithm_result)
    fig = visualizer.plot_cost_func_contribs()
    assert fig is not None  # Check if a figure is returned


def test_get_headers_called(mock_genetic_algorithm_result):
    visualizer = OptimizationResultVisualizer(mock_genetic_algorithm_result)
    visualizer.print_table_of_best_designs()
    # Ensure that get_headers was called with the correct parameters
    assert visualizer.ga_result.optimization_params.num_materials == mock_num_materials
    assert visualizer.ga_result.optimization_params.property_categories == ["carrier-transport", "dielectric"]
