"""genetic_algorithm_result.py."""
import numpy as np

from .genetic_algorithm_parameters import GeneticAlgorithmParams
from .optimization_params import OptimizationParams
from .population import Population

class GeneticAlgorithmResult():
    """
        Represents the result of a genetic algorithm run.

        Parameters:
            algo_parameters (GeneticAlgorithmParams)
            final_population (Population)
            lowest_costs (ndarray)
            - Lowest cost values across generations.
            avg_parent_costs (ndarray)
            - Average cost of the top-performing parents across generations.
            optimization_params (OptimizationParams)
        """

    def __init__(self,
                 algo_parameters:  GeneticAlgorithmParams,
                 opt_parameters:   OptimizationParams,
                 final_population: Population,
                 lowest_costs:     np.ndarray,
                 avg_parent_costs: np.ndarray):

        self.algo_parameters     = algo_parameters
        self.final_population    = final_population
        self.lowest_costs        = lowest_costs
        self.avg_parent_costs    = avg_parent_costs
        self.optimization_params = opt_parameters
