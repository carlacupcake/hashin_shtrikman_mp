# population_cython.pyx
# distutils: language=c++

# Define to silence the deprecation warning
cdef extern from *:
    """
    #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
    """

# Import necessary modules
import numpy as np
cimport numpy as cnp
cimport cython
from member import Member  
from genetic_algo import GAParams

def set_costs_cython(cnp.ndarray[cnp.double_t, ndim=2] population_values,
                     int num_members,
                     int num_materials,
                     int num_properties,
                     list property_categories,
                     dict property_docs,
                     dict desired_props,
                     object ga_params,
                     dict calc_guide):
    cdef cnp.ndarray[cnp.double_t, ndim=1] costs = np.zeros(num_members)
    cdef int i

    # Iterate over population members
    for i in range(num_members):
        this_member = Member(num_materials=num_materials,
                             num_properties=num_properties,
                             values=population_values[i],
                             property_categories=property_categories,
                             property_docs=property_docs,
                             desired_props=desired_props,
                             ga_params=ga_params,
                             calc_guide=calc_guide)
        # Calculate cost for this member
        costs[i] = this_member.get_cost()

    return costs
