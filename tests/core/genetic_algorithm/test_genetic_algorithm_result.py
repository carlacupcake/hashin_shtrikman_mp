"""test_genetic_algorithm_result.py"""
import numpy as np

from hashin_shtrikman_mp.core.genetic_algorithm.genetic_algorithm_parameters import GeneticAlgorithmParams
from hashin_shtrikman_mp.core.genetic_algorithm.genetic_algorithm_result import GeneticAlgorithmResult
from hashin_shtrikman_mp.core.genetic_algorithm.population import Population
from hashin_shtrikman_mp.core.genetic_algorithm.optimization_params import OptimizationParams

def test_genetic_algorithm_result_initialization():
    algo_params = GeneticAlgorithmParams()
    opt_params = OptimizationParams()
    final_pop = Population(opt_params, algo_params)
    lowest_costs = np.array([1.0, 0.8, 0.5])
    avg_parent_costs = np.array([1.2, 1.0, 0.7])

    result = GeneticAlgorithmResult(
        algo_parameters=algo_params,
        opt_parameters=opt_params,
        final_population=final_pop,
        lowest_costs=lowest_costs,
        avg_parent_costs=avg_parent_costs
    )

    assert result.algo_parameters == algo_params
    assert result.optimization_params == opt_params
    assert result.final_population == final_pop
    assert np.array_equal(result.lowest_costs, lowest_costs)
    assert np.array_equal(result.avg_parent_costs, avg_parent_costs)
