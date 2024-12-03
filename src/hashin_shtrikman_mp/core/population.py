from typing import Any, Union

import numpy as np
from pydantic import BaseModel, Field, model_validator

# Custom imports
from .genetic_algo import GAParams
from .member import Member


class Population(BaseModel):
    """
    Class to hold the population of members. The class also implements
    methods to generate the initial population, set the costs of the
    members, and sort the members based on their costs.
    """

    num_materials: int = Field(
        default=0,
        description="Number of materials in the ultimate composite."
    )
    num_properties: int = Field(
        default=0,
        description="Number of properties that each member of the population has."
    )
    property_categories: list[str] = Field(
        default=[],
        description="List of property categories considered for optimization."
    )
    property_docs: dict[str, dict[str, Any]] = Field(
        default={},
        description="A hard coded yaml file containing property categories and their individual properties."
    )
    desired_props: dict[str, list[float]] = Field(
        default={},
        description="Dictionary mapping individual properties to their desired "
                    "properties."
    )
    values: Union[np.ndarray, None] = Field(
        default=None,
        description="Matrix of values representing the population's properties."
    )
    costs: Union[np.ndarray, None] = Field(
        default=None,
        description="Array of costs associated with each member of the population."
    )
    ga_params: GAParams = Field(
        default_factory=GAParams,
        description="Parameter initilization class for the genetic algorithm."
    )
    calc_guide: Union[dict[str, Any], Any] = Field(
        default_factory=lambda: None,
        description="Calculation guide for property evaluation with compiled expressions."
    )

    # To use np.ndarray or other arbitrary types in your Pydantic models
    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode="before")
    def set_default_values_and_costs(cls, values):
        ga_params = GAParams(**values.get("ga_params", {})) if isinstance(values.get("ga_params"), dict) else values.get("ga_params", GAParams())

        num_members = ga_params.num_members  # Assuming GAParams model has a num_members field directly accessible
        num_properties = values.get("num_properties", 0)
        num_materials = values.get("num_materials", 0)

        # Set default for values if not provided or is np.empty
        if values.get("values") is None or (isinstance(values.get("values"), np.ndarray) and values.get("values").size == 0):
            values["values"] = np.zeros((num_members, num_properties * num_materials))

        # Set default for costs in a similar manner
        if values.get("costs") is None or (isinstance(values.get("costs"), np.ndarray) and values.get("costs").size == 0):
            values["costs"] = np.zeros((num_members, num_properties * num_materials))

        return values

    #------ Getter Methods ------#
    def get_unique_designs(population, costs):

        # Costs are often equal to >10 decimal points
        # Truncate to obtain a richer set of suggestions
        new_costs = np.round(costs, decimals=3)

        # Obtain unique members and costs
        [unique_costs, unique_indices] = np.unique(new_costs, return_index=True)
        unique_members = population[unique_indices]

        return [unique_members, unique_costs]

    def get_effective_properties(self):
        population_values = self.values
        num_members = self.ga_params.num_members
        num_properties = self.num_properties - 1 # do not include volume fraction

        all_effective_properties = np.zeros((num_members, num_properties))
        for i in range(num_members):
            this_member = Member(num_materials=self.num_materials,
                                 num_properties=self.num_properties,
                                 values=population_values[i, :],
                                 property_categories=self.property_categories,
                                 property_docs=self.property_docs,
                                 desired_props=self.desired_props,
                                 ga_params=self.ga_params,
                                 calc_guide=self.calc_guide)
            eff_props = this_member.get_effective_properties()
            all_effective_properties[i, :] = eff_props

        return all_effective_properties

    #------ Setter Methods ------#
    def set_random_values(self, lower_bounds = None, upper_bounds = None, start_member = 0, indices_elastic_moduli = None):

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
                    if category in self.property_categories:
                        for property in properties:
                            lower_bounds_list.append(property)

        for material in upper_bounds:
            if material != "volume-fractions":
                for category, properties in upper_bounds[material].items():
                    if category in self.property_categories:
                        for property in properties:
                            upper_bounds_list.append(property)

        # Cast lists to numpy arrays
        lower_bounds_array = np.array(lower_bounds_list)
        upper_bounds_array = np.array(upper_bounds_list)

        # Fill in the population, not including the volume fractions
        num_materials = len(lower_bounds.keys()) - 1 # subtract the entry for the mixture properties
        population_size = len(self.values)
        for i in range(start_member, population_size):
            self.values[i, :-num_materials] = np.random.uniform(lower_bounds_array, upper_bounds_array)

        # Adjust for bulk and shear moduli, if they are present (cannot have bulk_i < bulk_j and shear_i > shear_j simultaneously)
        stop = self.num_materials * (self.num_properties - 1) # the last num_materials entries are volume fractions, not material properties
        step = self.num_properties - 1                        # subtract 1 so as not to include volume fraction

        # Extract bulk moduli and shear moduli from population
        [bulk_idx, shear_idx] = indices_elastic_moduli

        # if bulk)idx & shear_idx are None, then just skip the following
        if bulk_idx is None and shear_idx is None:
            return self

        else:
            # Order the materials in each member according to bulk modulus
            for i in range(start_member, population_size):
                member = self.values[i, :]
                sorted_bulk_indices = np.argsort(member[bulk_idx:stop:step])
                unsorted_member = np.zeros((self.num_materials, (self.num_properties - 1)))
                for m in range(self.num_materials):
                    start = m * (self.num_properties - 1)
                    end = start + (self.num_properties - 1)
                    material = member[start:end]
                    unsorted_member[m, :] = material
                sorted_member = unsorted_member[sorted_bulk_indices]
                self.values[i, :-self.num_materials] = sorted_member.flatten()

            # Use sorted bulk moduli values to potentially replace lower bound on shear modulus
            # so that random values allow for  bulk_i < bulk_j and shear_i < shear_j simultaneously
            shear_indices = list(range(shear_idx, stop, step))
            for i in range(start_member, population_size):
                member = self.values[i, :]
                shear_mods = member[shear_idx:stop:step]
                for m in range(1, self.num_materials):
                    idx = shear_indices[m]
                    self.values[i, idx] = np.random.uniform(max(shear_mods[m-1], lower_bounds_array[idx]), max(shear_mods[m-1], upper_bounds_array[idx]))

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

    def set_costs(self):
        population_values = self.values
        num_members = self.ga_params.num_members
        costs = np.zeros(num_members)
        for i in range(num_members):
            this_member = Member(num_materials=self.num_materials,
                                 num_properties=self.num_properties,
                                 values=population_values[i, :],
                                 property_categories=self.property_categories,
                                 property_docs=self.property_docs,
                                 desired_props=self.desired_props,
                                 ga_params=self.ga_params,
                                 calc_guide=self.calc_guide)
            costs[i] = this_member.get_cost()

        self.costs = costs
        return self

    def set_order_by_costs(self, sorted_indices):
        temporary = np.zeros((self.ga_params.num_members, self.num_properties * self.num_materials))
        for i in range(len(sorted_indices)):
            temporary[i,:] = self.values[int(sorted_indices[i]),:]
        self.values = temporary
        return self

    #------ Other Class Methods ------#

    def sort_costs(self):
        sorted_costs = np.sort(self.costs, axis=0)
        sorted_indices = np.argsort(self.costs, axis=0)
        return [sorted_costs, sorted_indices]

