"""population.py."""
import numpy as np

from .genetic_algorithm_parameters import GeneticAlgorithmParams
from .member import Member
from .optimization_params import OptimizationParams


class Population:
    """
    Class to hold the population of members.

    The class also implements methods to generate the initial population,
    set the costs of the members, and sort the members based on their costs.
    """

    def __init__(self,
                 optimization_params: OptimizationParams,
                 ga_params: GeneticAlgorithmParams,
                 values: np.ndarray = None,
                 costs: np.ndarray = None) -> None:
        """
        Parameters
        ----------
        optimization_params : OptimizationParams
            _description_
        ga_params : GeneticAlgorithmParams
            _description_
        values : _type_, optional
            Matrix of values representing the population's properties., by default None
        costs : _type_, optional
            Array of costs associated with each member of the population., by default None
        """
        self.opt_params = optimization_params
        self.ga_params = ga_params

        num_members = ga_params.num_members  # Assuming GAParams model has a num_members field directly accessible
        num_properties = self.opt_params.num_properties
        num_materials = self.opt_params.num_materials

        self.values = values
        if self.values is None:
            self.values = np.zeros((num_members, num_properties * num_materials))

        self.costs = costs
        if self.costs is None:
            self.costs = np.zeros((num_members, num_properties * num_materials))

    def get_unique_designs(self) -> "Population":
        """
        Retrieves the unique designs from the population based on their costs.

        This function calculates the costs of all members in the population,
        rounds the costs to three decimal places, and then identifies the unique designs
        by selecting the members with unique rounded costs.
        It returns the unique members and their corresponding costs.

        Args:
            None.

        Returns
        -------
            list: A list containing two elements:
                - A numpy array of unique population members corresponding to unique costs.
                - A numpy array of unique costs for these members.
        """
        self.set_costs()
        final_costs = self.costs
        rounded_costs = np.round(final_costs, decimals=3)

        # Obtain unique members and costs
        [unique_costs, unique_indices] = np.unique(rounded_costs, return_index=True)
        unique_members = self.values[unique_indices]
        return [unique_members, unique_costs]

    def get_effective_properties(self) -> np.ndarray:
        """
        Calculates the effective properties for each member in the population.

        This function iterates over all population members, creating a `Member` instance for each,
        and retrieves the effective properties using the `get_effective_properties` method of the `Member` class.
        The effective properties exclude the volume fraction, and are returned as an array where each row represents
        the effective properties of one member.

        Args:
            None.

        Returns
        -------
            np.ndarray: A 2D numpy array where each row contains the effective properties for
                        a single member of the population. The number of columns corresponds to
                        the number of material properties, excluding the volume fraction.
        """
        population_values = self.values
        num_members = self.ga_params.num_members
        num_properties = self.opt_params.num_properties - 1 # do not include volume fraction

        all_effective_properties = np.zeros((num_members, num_properties))
        for i in range(num_members):
            this_member = Member(ga_params=self.ga_params,
                                 optimization_params=self.opt_params,
                                 values=population_values[i, :])
            eff_props = this_member.get_effective_properties()
            all_effective_properties[i, :] = eff_props

        return all_effective_properties

    def set_random_values(self,
                          lower_bounds: np.ndarray = None,
                          upper_bounds: np.ndarray = None,
                          start_member: int = 0,
                          indices_elastic_moduli: list = None) -> "Population":
        """
        Sets random values for the properties of each member in the population.

        The function first initializes and processes the lower and upper bounds for the material properties.
        It then fills in the population values randomly within these bounds. Additionally, if bulk modulus
        and shear modulus are specified, the function ensures that these values are ordered correctly in each
        member of the population. The volume fractions are also set to sum to 1, with each material’s volume
        fraction being randomly selected within the bounds.

        Args:
            lower_bounds (np.ndarray, optional)
            upper_bounds (np.ndarray, optional)
            start_member (int, optional): The index of the first member to update in the population. Defaults to 0.
            indices_elastic_moduli (list, optional): A list containing the indices for bulk and shear moduli,
                                                    used for ensuring proper ordering. Defaults to None.

        Returns
        -------
            Population: The instance of the population with updated property values for each member.
        """
        # Initialize bounds lists
        if indices_elastic_moduli is None:
            indices_elastic_moduli = [None, None]
        if upper_bounds is None:
            upper_bounds = {}
        if lower_bounds is None:
            lower_bounds = {}
        lower_bounds_list = []
        upper_bounds_list = []

        # Unpack bounds from dictionaries, include bounds for all materials
        for material in lower_bounds:
            if material != "volume-fractions":
                for category, properties in lower_bounds[material].items():
                    if category in self.opt_params.property_categories:
                        for prop in properties:
                            lower_bounds_list.append(prop)

        for material in upper_bounds:
            if material != "volume-fractions":
                for category, properties in upper_bounds[material].items():
                    if category in self.opt_params.property_categories:
                        for prop in properties:
                            upper_bounds_list.append(prop)

        # Cast lists to numpy arrays
        lower_bounds_array = np.array(lower_bounds_list)
        upper_bounds_array = np.array(upper_bounds_list)

        # Fill in the population, not including the volume fractions
        num_materials = len(lower_bounds.keys()) - 1 # subtract the entry for the mixture properties
        population_size = len(self.values)
        for i in range(start_member, population_size):
            self.values[i, :-num_materials] = np.random.uniform(lower_bounds_array, upper_bounds_array)

        # Adjust for bulk and shear moduli, if they are present
        # (cannot have bulk_i < bulk_j and shear_i > shear_j simultaneously)
        # The last num_materials entries are volume fractions, not material properties
        stop = self.opt_params.num_materials * (self.opt_params.num_properties - 1)
        # Subtract 1 so as not to include volume fraction
        step = self.opt_params.num_properties - 1

        # Extract bulk moduli and shear moduli from population
        [bulk_idx, shear_idx] = indices_elastic_moduli

        # if bulk)idx & shear_idx are None, then just skip the following
        if bulk_idx is None and shear_idx is None:
            return self

        # Order the materials in each member according to bulk modulus
        for i in range(start_member, population_size):
            member = self.values[i, :]
            sorted_bulk_indices = np.argsort(member[bulk_idx:stop:step])
            unsorted_member = np.zeros((self.opt_params.num_materials, (self.opt_params.num_properties - 1)))
            for m in range(self.opt_params.num_materials):
                start = m * (self.opt_params.num_properties - 1)
                end = start + (self.opt_params.num_properties - 1)
                material = member[start:end]
                unsorted_member[m, :] = material
            sorted_member = unsorted_member[sorted_bulk_indices]
            self.values[i, :-self.opt_params.num_materials] = sorted_member.flatten()

        # Use sorted bulk moduli values to potentially replace lower bound on shear modulus
        # so that random values allow for  bulk_i < bulk_j and shear_i < shear_j simultaneously
        shear_indices = list(range(shear_idx, stop, step))
        for i in range(start_member, population_size):
            member = self.values[i, :]
            shear_mods = member[shear_idx:stop:step]
            for m in range(1, self.opt_params.num_materials):
                idx = shear_indices[m]
                self.values[i, idx] = np.random.uniform(max(shear_mods[m-1],
                                                            lower_bounds_array[idx]),
                                                        max(shear_mods[m-1],
                                                            upper_bounds_array[idx]))

        # Include volume fractions
        for i in range(start_member, population_size):
            sum_vf = 0
            for v in range(num_materials - 1):
                lb = lower_bounds["volume-fractions"][v]
                ub = min(1 - sum_vf, upper_bounds["volume-fractions"][v])
                this_vf = np.random.uniform(lb, ub)
                self.values[i, -num_materials + v] = this_vf
                sum_vf += this_vf

            # Final volume fraction not random b/c must sum to 1
            self.values[i, -1] = 1 - sum_vf

        return self

    def set_costs(self) -> "Population":
        """
        Calculates the costs for each member in the population.

        This method iterates through each member in the population, calculates their individual cost using
        the `get_cost` method from the `Member` class, and stores the resulting costs in the `self.costs` array.

        Args:
            None

        Returns
        -------
            Population: The instance itself with the `self.costs` array
                        populated with the calculated costs for each member.
        """
        population_values = self.values
        num_members = self.ga_params.num_members
        costs = np.zeros(num_members)
        for i in range(num_members):
            this_member = Member(ga_params=self.ga_params,
                                 optimization_params=self.opt_params,
                                 values=population_values[i, :])
            costs[i] = this_member.get_cost()

        self.costs = costs
        return self

    def set_order_by_costs(self, sorted_indices: np.ndarray = None) ->  "Population":
        """
        Reorders the population based on the sorted indices of costs.

        Args:
            sorted_indices (numpy.ndarray)

        Returns
        -------
            Population: The instance itself with the `self.values`
            array reordered according to `sorted_indices`.

        """
        temporary = np.zeros((self.ga_params.num_members,
                              self.opt_params.num_properties * self.opt_params.num_materials))
        for i in range(len(sorted_indices)):
            temporary[i,:] = self.values[int(sorted_indices[i]),:]
        self.values = temporary
        return self

    def sort_costs(self) -> list:
        """
        Sorts the costs and returns the sorted values along with their corresponding indices.

        Args:
            None

        Returns
        -------
            list: A list containing two arrays:
                - The first element is a 1D or 2D array of sorted costs.
                - The second element is a 1D or 2D array of indices that
                  would sort the original `self.costs`.
        """
        sorted_costs = np.sort(self.costs, axis=0)
        sorted_indices = np.argsort(self.costs, axis=0)
        return [sorted_costs, sorted_indices]