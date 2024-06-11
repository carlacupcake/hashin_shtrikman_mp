# coptimization.pyx

# distutils: language=c++
# Define to silence the deprecation warning
cdef extern from *:
    """
    #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
    """

import numpy as np
cimport numpy as cnp
cimport cython
from libc.stdlib cimport rand, srand, RAND_MAX

# Custom imports
from member import Member  
from population import Population
from genetic_algo import GAParams

def cset_HS_optim_params(list property_categories,
                         dict property_docs,
                         dict lower_bounds,
                         dict upper_bounds,
                         dict desired_props,
                         int num_materials,
                         int num_properties,
                         object ga_params,
                         dict calc_guide):

        """
        MAIN OPTIMIZATION FUNCTION - CYTHONIZED
        """
        # Not compatible with Pydantic architecture
        # As such, separated from optimization.py

        cdef int num_parents = ga_params.num_parents
        cdef int num_kids = ga_params.num_kids
        cdef int num_generations = ga_params.num_generations
        cdef int num_members = ga_params.num_members
        cdef int g = 0

        # Initialize arrays
        cdef cnp.ndarray[cnp.double_t, ndim=2] all_costs = np.ones((num_generations, num_members))
        cdef cnp.ndarray[cnp.double_t, ndim=1] lowest_costs = np.zeros(num_generations)
        cdef cnp.ndarray[cnp.double_t, ndim=1] avg_parent_costs = np.zeros(num_generations)
        cdef cnp.ndarray[cnp.double_t, ndim=1] costs = np.zeros(num_members)

        # Randomly populate first generation
        population = Population(num_materials=num_materials, 
                                num_properties=num_properties, 
                                property_categories=property_categories, 
                                property_docs=property_docs,
                                desired_props=desired_props, 
                                ga_params=ga_params,
                                calc_guide=calc_guide)
        population.set_random_values(lower_bounds=lower_bounds, 
                                    upper_bounds=upper_bounds, 
                                    num_members=ga_params.num_members)

        # Calculate the costs of the first generation
        population.cset_costs()   

        # Sort the costs of the first generation
        sorted_costs, sorted_indices = population.sort_costs()  
        all_costs[0, :] = sorted_costs.reshape(1, num_members) 

        # Store the cost of the best performer and average cost of the parents 
        lowest_costs[0] = np.min(sorted_costs)
        avg_parent_costs[0] = np.mean(sorted_costs[0:num_parents])

        # Before the loop, seed the random number generator
        srand(0)

        # Declare phi1 and phi2 outside the loop
        cdef double phi1, phi2

        # Perform all later generations    
        for g in range(1, num_generations):
            print(f"Generation {g} of {num_generations}")
            costs[:num_parents] = sorted_costs[:num_parents]  # retain the parents from the previous generation

            # Select top parents from population to be breeders
            for p in range(0, num_parents, 2):
                with nogil:
                    phi1 = rand() / (RAND_MAX + 1.0)
                    phi2 = rand() / (RAND_MAX + 1.0)
                kid1 = phi1 * population.values[p, :] + (1 - phi1) * population.values[p+1, :]
                kid2 = phi2 * population.values[p, :] + (1 - phi2) * population.values[p+1, :]
                
                #with nogil:
                # Append offspring to population, overwriting old population members 
                population.values[num_parents+p, 0:] = kid1  # Direct array assignment without slice
                population.values[num_parents+p+1, 0:] = kid2

                #with gil:
                # Cast offspring to members and evaluate costs
                kid1 = Member(num_materials=num_materials,
                            num_properties=num_properties,
                            values=kid1,
                            property_categories=property_categories,
                            property_docs=property_docs,
                            desired_props=desired_props,
                            ga_params=ga_params,
                            calc_guide=calc_guide)
                kid2 = Member(num_materials=num_materials,
                            num_properties=num_properties,
                            values=kid2,
                            property_categories=property_categories,
                            property_docs=property_docs,
                            desired_props=desired_props,
                            ga_params=ga_params,
                            calc_guide=calc_guide)
                costs[num_parents+p] = kid1.get_cost()
                costs[num_parents+p+1] = kid2.get_cost()

            # Randomly generate new members to fill the rest of the population
            members_minus_parents_minus_kids = num_members - num_parents - num_kids
            population.set_random_values(lower_bounds=lower_bounds,
                                         upper_bounds=upper_bounds,
                                         num_members=members_minus_parents_minus_kids)

            # Calculate the costs of the gth generation
            population.cset_costs()

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

        return population, lowest_costs, avg_parent_costs
