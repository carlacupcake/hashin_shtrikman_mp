"""genetic_algorithm.py."""
import numpy as np

from ..user_input import UserInput
from .genetic_algorithm_parameters import GeneticAlgorithmParams
from .genetic_algorithm_result import GeneticAlgorithmResult
from .member import Member
from .optimization_params import OptimizationParams
from .population import Population

class GeneticAlgorithm:

    def run(self,
            user_inputs:    UserInput,
            ga_algo_params: GeneticAlgorithmParams = None,
            gen_counter:    bool = False):
        """
        Executes the Genetic Algorithm (GA) optimization process.

        Initializes a population, evaluates costs, and iteratively
        evolves the population through breeding and selection to minimize
        the cost function over multiple generations. The best and average costs
        for each generation are tracked, and the final population is returned
        alongside optimization results.

        Args:
            user_inputs (UserInput)
            ga_algo_params (GeneticAlgorithmParams, optional)
            gen_counter (bool, optional)

        Returns
        -------
            GeneticAlgorithmResult
        """

        optimization_parameters = OptimizationParams.from_user_input(user_inputs)

        if ga_algo_params is None:
            ga_algo_params = GeneticAlgorithmParams()

        # Unpack necessary attributes from self
        num_parents     = ga_algo_params.num_parents
        num_kids        = ga_algo_params.num_kids
        num_generations = ga_algo_params.num_generations
        num_members     = ga_algo_params.num_members

        lower_bounds = optimization_parameters.lower_bounds
        upper_bounds = optimization_parameters.upper_bounds

        # Initialize arrays to store the cost and original indices of each generation
        all_costs = np.ones((num_generations, num_members))

        # Initialize arrays to store best performer and parent avg
        lowest_costs = np.zeros(num_generations)     # best cost
        avg_parent_costs = np.zeros(num_generations) # avg cost of parents

        # Generation counter
        g = 0

        # Initialize array to store costs for current generation
        costs = np.zeros(num_members)

        # Randomly populate first generation
        population = Population(optimization_params=optimization_parameters,
                                ga_params=ga_algo_params)
        population.set_random_values(
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            start_member=0,
            indices_elastic_moduli=optimization_parameters.indices_elastic_moduli
        )

        # Calculate the costs of the first generation
        population.set_costs()

        # Sort the costs of the first generation
        [sorted_costs, sorted_indices] = population.sort_costs()
        all_costs[g, :] = sorted_costs.reshape(1, num_members)

        # Store the cost of the best performer and average cost of the parents
        lowest_costs[g] = np.min(sorted_costs)
        avg_parent_costs[g] = np.mean(sorted_costs[0:num_parents])

        # Update population based on sorted indices
        population.set_order_by_costs(sorted_indices)

        # Perform all later generations
        while g < num_generations:

            if gen_counter:
                print(f"Generation {g} of {num_generations}")

            # Retain the parents from the previous generation
            costs[0:num_parents] = sorted_costs[0:num_parents]

            # Select top parents from population to be breeders
            for p in range(0, num_parents, 2):
                phi1, phi2 = np.random.rand(2)
                kid1 = phi1 * population.values[p, :] + (1-phi1) * population.values[p+1, :]
                kid2 = phi2 * population.values[p, :] + (1-phi2) * population.values[p+1, :]

                # Append offspring to population, overwriting old population members
                population.values[num_parents+p,   :] = kid1
                population.values[num_parents+p+1, :] = kid2

                # Cast offspring to members and evaluate costs
                kid1 = Member(values=kid1,
                              optimization_params=optimization_parameters,
                              ga_params=ga_algo_params)
                kid2 = Member(values=kid2,
                              optimization_params=optimization_parameters,
                              ga_params=ga_algo_params)
                costs[num_parents+p]   = kid1.get_cost()
                costs[num_parents+p+1] = kid2.get_cost()

            # Randomly generate new members to fill the rest of the population
            parents_plus_kids = num_parents + num_kids
            population.set_random_values(
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                start_member=parents_plus_kids,
                indices_elastic_moduli=optimization_parameters.indices_elastic_moduli
            )

            # Calculate the costs of the gth generation
            population.set_costs()

            # Sort the costs for the gth generation
            [sorted_costs, sorted_indices] = population.sort_costs()
            all_costs[g, :] = sorted_costs.reshape(1, num_members)

            # Store the cost of the best performer and average cost of the parents
            lowest_costs[g] = np.min(sorted_costs)
            avg_parent_costs[g] = np.mean(sorted_costs[0:num_parents])

            # Update population based on sorted indices
            population.set_order_by_costs(sorted_indices)

            # Update the generation counter
            g = g + 1

        return GeneticAlgorithmResult(
            algo_parameters=ga_algo_params,
            final_population=population,
            lowest_costs=lowest_costs,
            avg_parent_costs=avg_parent_costs,
            opt_parameters=optimization_parameters
        )
