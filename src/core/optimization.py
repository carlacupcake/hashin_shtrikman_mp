# From MPRester
import re
import json
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, root_validator
import yaml
from monty.serialization import loadfn

# Custom Classes
from core.genetic_algo import GAParams
from core.member import Member
from core.population import Population

# Other
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from mp_api.client import MPRester
from mpcontribs.client import Client
from tabulate import tabulate
from log.custom_logger import logger
import copy
from importlib import resources
from pathlib import Path

# HashinShtrikman class defaults
DEFAULT_FIELDS: dict    = {"material_id": [], 
                           "is_stable": [], 
                           "band_gap": [], 
                           "is_metal": [],
                           "formula_pretty": [],}
MODULE_DIR = Path(__file__).resolve().parent

class HashinShtrikman(BaseModel):
    """
    Hashin-Shtrikman optimization class. 

    Class to integrate Hashin-Shtrikman (HS) bounds with a genetic algorithm, 
    leveraging the Materials Project (MP) database.
    """

    api_key: Optional[str] = Field(default=None, 
                                   description="API key for accessing Materials "
                                   "Project database.")
    mp_contribs_project: Optional[str] = Field(default=None, 
                                               description="MPContribs project name "
                                               "for querying project-specific data.")
    user_input: Dict = Field(default_factory=dict, 
                             description="User input specifications for the "
                             "optimization process.")
    fields: Dict[str, List[Any]] = Field(default_factory=lambda: DEFAULT_FIELDS.copy(), 
                                         description="Fields to query from the "
                                         "Materials Project database.")
    num_properties: int = Field(default=0, 
                                description="Number of properties being optimized.")
    ga_params: GAParams = Field(default_factory=GAParams, 
                                description="Parameter initilization class for the "
                                "genetic algorithm.")
    final_population: Population = Field(default_factory=Population, 
                                         description="Final population object after "
                                         "optimization.")
    cost_history: np.ndarray = Field(default_factory=lambda: np.empty(0),
                                     description="Historical cost values of "
                                     "populations across generations.")
    lowest_costs: np.ndarray = Field(default_factory=lambda: np.empty(0), 
                                     description="Lowest cost values across "
                                     "generations.")
    avg_parent_costs: np.ndarray = Field(default_factory=lambda: np.empty(0), 
                                         description="Average cost of the "
                                         "top-performing parents across generations.")
    calc_guide: Dict[str, Any] = Field(default_factory=lambda: 
                                       loadfn("cost_calculation_formulas.yaml"), 
                                       description="Calculation guide for property "
                                       "evaluation. This is a hard coded yaml file.")
    property_categories: List[str] = Field(default_factory=list, 
                                           description="List of property categories "
                                           "considered for optimization.")
    property_docs: Dict[str, Dict[str, Any]] = Field(default_factory=dict, 
                                                     description="A hard coded yaml "
                                                     "file containing property "
                                                     "categories and their individual "
                                                     "properties.")
    desired_props: Dict[str, List[float]] = Field(default_factory=dict, 
                                                  description="Dictionary mapping "
                                                  "individual properties to their "
                                                  "desired properties.")
    lower_bounds: Dict[str, Any] = Field(default_factory=dict, 
                                         description="Lower bounds for properties of "
                                         "materials considered in the optimization.")
    upper_bounds: Dict[str, Any] = Field(default_factory=dict, 
                                         description="Upper bounds for properties of "
                                         "materials considered in the optimization.")
    
    # To use np.ndarray or other arbitrary types in your Pydantic models
    class Config:
        arbitrary_types_allowed = True

    @root_validator(pre=True)
    def load_and_set_properties(cls, values):
        # Load property categories and docs
        property_categories, property_docs = cls.load_property_categories(f"{MODULE_DIR}/../io/inputs/mp_property_docs.yaml", user_input=values.get("user_input", {}))
        values["property_categories"] = property_categories
        values["property_docs"] = property_docs
        
        # Load calculation guide, if necessary
        calc_guide = loadfn(values.get("calc_guide", f"{MODULE_DIR}/../io/inputs/cost_calculation_formulas.yaml"))
        values["calc_guide"] = calc_guide
        
        # Since user_input is required to set desired props and bounds, ensure it's processed last
        user_input = values.get("user_input", {})
        if user_input:
            desired_props = cls.set_desired_props_from_user_input(user_input, property_categories=property_categories, property_docs=property_docs)
            lower_bounds = cls.set_bounds_from_user_input(user_input, 'lower_bound', property_docs=property_docs)
            upper_bounds = cls.set_bounds_from_user_input(user_input, 'upper_bound', property_docs=property_docs)
            num_properties = cls.set_num_properties_from_desired_props(desired_props, lower_bounds)

            # Update values accordingly
            values.update({
                "desired_props": desired_props,
                "lower_bounds": lower_bounds,
                "upper_bounds": upper_bounds,
                "num_properties": num_properties
            })
        
        return values


    #------ Load property docs from MP ------# 
    @staticmethod
    def load_property_categories(filename=f"{MODULE_DIR}/../io/inputs/mp_property_docs.yaml", user_input: Dict = {}):
            print(f"Loading property categories from {filename}.")
            import os
            print(f"Loading property categories from {os.getcwd()}.")
            """Load property categories from a JSON file."""
            property_categories = []
            try:
                property_docs = loadfn(filename)
                
                # Flatten the user input to get a list of all properties defined by the user
                user_defined_properties = []

                for material_props in user_input.values():
                    for props in material_props.keys():
                        user_defined_properties.append(props)
                        #only keep the unique entries of the list
                        user_defined_properties = list(set(user_defined_properties))

                # Iterate through property categories to check which are present in the user input
                for category, properties in property_docs.items():
                    if any(prop in user_defined_properties for prop in properties):
                        property_categories.append(category)

            except FileNotFoundError:
                print(f"File {filename} not found.")
            except json.JSONDecodeError:
                print(f"Error decoding JSON from file {filename}.")
            
            print(f"property_categories = {property_categories}")
            return property_categories, property_docs
        
    #------ Getter Methods ------#

    def get_lower_bounds(self):
        return self.lower_bounds
    
    def get_upper_bounds(self):
        return self.upper_bounds
    
    def get_retain_parents(self):
        return self.retain_parents
    
    def get_allow_mutations(self):
        return self.allow_mutations

    def get_property_docs(self):
        return self.property_docs
    
    def get_desired_props(self):
        return self.desired_props
    
    def get_has_props(self):
        return self.has_props
    
    def get_fields(self):
        return self.fields
    
    def get_num_properties(self):
        return self.num_properties
    
    def get_ga_params(self):
        return self.ga_params
    
    def get_final_population(self):
        return self.final_population
    
    def get_cost_history(self):
        return self.cost_history
    
    def get_lowest_costs(self):
        return self.lowest_costs           

    def get_avg_parent_costs(self):
        return self.avg_parent_costs
    
    def get_headers(self, include_mpids=False, file_name = f"{MODULE_DIR}/../io/inputs/display_table_headers.yaml"):
        
        with open(file_name, 'r') as stream:
            try:
                data = yaml.safe_load(stream)
                headers = []

                if include_mpids:
                    headers.extend(["Material 1 MP-ID", "Material 2 MP-ID"])

                # Temporarily store headers by category to ensure phase grouping
                temp_headers = {}

                # Exclude 'Common' initially and handle it at the end
                categories_to_exclude = ["Common"]

                for material_key, categories in data.items():
                    if material_key in categories_to_exclude:
                        continue

                    for category, properties in categories.items():
                        if category in self.property_categories:
                            if category not in temp_headers:
                                temp_headers[category] = []

                            for prop_key, prop_value in properties.items():
                                temp_headers[category].append(prop_value)

                # Add the consolidated headers from temp_headers to the final list, ensuring proper order
                for category in temp_headers:
                    headers.extend(temp_headers[category])

                # Handle 'Common' properties if present
                if "Common" in data:
                    for common_key in data["Common"].keys():
                        headers.append(common_key)

            except yaml.YAMLError as exc:
                print(exc)
        
        return headers
        
    def get_unique_designs(self):

        # Costs are often equal to >10 decimal points, truncate to obtain a richer set of suggestions
        self.final_population.set_costs()
        final_costs = self.final_population.get_costs()
        rounded_costs = np.round(final_costs, decimals=3)
    
        # Obtain unique members and costs
        [unique_costs, unique_indices] = np.unique(rounded_costs, return_index=True)
        unique_members = self.final_population.values[unique_indices]

        return [unique_members, unique_costs] 

    def get_table_of_best_designs(self):

        [unique_members, unique_costs] = self.get_unique_designs()
        table_data = np.hstack((unique_members[0:20, :], unique_costs[0:20].reshape(-1, 1))) 

        return table_data

    def get_dict_of_best_designs(self):

        # Initialize dictionaries for each material based on selected property categories
        best_designs_dict = {"mat1": {}, 
                             "mat2": {}}
        
        # Initialize the structure for each category
        for category in self.property_categories:
            for mat in best_designs_dict.keys():
                best_designs_dict[mat][category] = {}
                for prop in self.property_docs[category]:
                    best_designs_dict[mat][category][prop] = []

        [unique_members, unique_costs] = self.get_unique_designs()

        # Populate the dictionary with unique design values
        for i, _ in enumerate(unique_costs):
            idx = 0
            for category in self.property_categories:
                for mat in best_designs_dict.keys():
                    for prop in self.property_docs[category]:
                        best_designs_dict[mat][category][prop].append(unique_members[i][idx])
                        idx += 1

        return best_designs_dict
    
    def get_material_matches(self, consolidated_dict: dict = {}): 

        best_designs_dict = self.get_dict_of_best_designs()
        print(f"best_designs_dict = {best_designs_dict}")
        
        # TODO get from latest final_dict file: change this to a method that reads from the latest MP database
        if consolidated_dict == {}:
            with open("test_final_dict") as f:
                consolidated_dict = json.load(f)

        # Initialize sets for matching indices
        mat1_matches = set(range(len(consolidated_dict["material_id"])))
        mat2_matches = set(range(len(consolidated_dict["material_id"])))

        # Helper function to get matching indices based on property extrema
        def get_matching_indices(prop_name, bounds_dict, mat_key):
            lower_bound = min(bounds_dict[mat_key][prop_name])
            logger.info(f"lower_bound_{prop_name} = {lower_bound}")
            upper_bound = max(bounds_dict[mat_key][prop_name])
            logger.info(f"upper_bound_{prop_name} = {upper_bound}")
            return {i for i, value in enumerate(consolidated_dict[prop_name]) if lower_bound <= value <= upper_bound}

        # Iterate over categories and properties
        for category in self.property_categories:
            if category in self.property_docs:
                for prop in self.property_docs[category]:
                    if prop in consolidated_dict:  # Ensure property exists in consolidated_dict
                        mat1_matches &= get_matching_indices(prop, best_designs_dict["mat1"], category)
                        logger.info(f"mat1_match_index_{prop} = {get_matching_indices(prop, best_designs_dict['mat1'], category)}")
                        mat2_matches &= get_matching_indices(prop, best_designs_dict["mat2"], category)
                        logger.info(f"mat2_match_index_{prop} = {get_matching_indices(prop, best_designs_dict['mat2'], category)}")

        # Extract mp-ids based on matching indices
        mat_1_ids = [consolidated_dict["material_id"][i] for i in mat1_matches]
        mat_2_ids = [consolidated_dict["material_id"][i] for i in mat2_matches]

        return mat_1_ids, mat_2_ids     


    def get_material_match_costs(self, 
                                 mat_1_ids, 
                                 mat_2_ids, 
                                 consolidated_dict: dict = {}):

        for m1 in mat_1_ids:
            for m2 in mat_2_ids:
                m1_idx = consolidated_dict["material_id"].index(m1)
                m2_idx = consolidated_dict["material_id"].index(m2)
                material_values: List = []
                # Iterate through each property category of interest
                for category in self.property_categories:
                    if category in self.property_docs:
                        # Append material 1 properties
                        self.append_property_values(self.property_docs[category], m1_idx, material_values, consolidated_dict)
                        # Append material 2 properties
                        self.append_property_values(self.property_docs[category], m2_idx, material_values, consolidated_dict)

                # Create population of same properties for all members based on material match pair
                values = np.reshape(material_values*self.ga_params.get_num_members(), (self.ga_params.get_num_members(), len(material_values))) 
                population = np.reshape(values, (self.ga_params.get_num_members(), len(material_values)))

                # Only the vary the mixing parameter and volume fraction across the population
                # create uniform mixing params from 0 to 1 with a spacing of 0.02 but with a shape of self.ga_params.get_num_members() & 1
                mixing_param = np.linspace(0.01, 0.99, self.ga_params.get_num_members()).reshape(self.ga_params.get_num_members(), 1)
                phase1_vol_frac = np.linspace(0.01, 0.99, self.ga_params.get_num_members()).reshape(self.ga_params.get_num_members(), 1)

                # Include the random mixing parameters and volume fractions in the population
                values = np.c_[population, mixing_param, phase1_vol_frac]    

                # Instantiate the population and find the best performers
                population_obj = Population(num_properties=self.num_properties, 
                                        values=values, 
                                        property_categories=self.property_categories, 
                                        desired_props=self.desired_props, 
                                        ga_params=self.ga_params, 
                                        calc_guide=self.calc_guide, 
                                        property_docs=self.property_docs)
                population_obj.set_costs()
                sorted_costs, sorted_indices = population_obj.sort_costs()
                population_obj.set_order_by_costs(sorted_indices)
                sorted_costs = np.reshape(sorted_costs, (len(sorted_costs), 1))

                # Assemble a table for printing
                mat1_id = np.reshape([m1]*self.ga_params.get_num_members(), 
                                     (self.ga_params.get_num_members(),1))
                mat2_id = np.reshape([m2]*self.ga_params.get_num_members(), 
                                     (self.ga_params.get_num_members(),1))
                table_data = np.c_[mat1_id, mat2_id, population_obj.values, sorted_costs] 
                
                print("\nMATERIALS PROJECT PAIRS AND HASHIN-SHTRIKMAN RECOMMENDED VOLUME FRACTION")
                print(tabulate(table_data[0:5, :], headers=self.get_headers())) # hardcoded to be 5 rows, could change
                
                # with open("table_data.csv", "w") as f:
                #     f.write(",".join(self.get_headers()) + "\n")
                #     # np.savetxt(f, table_data, delimiter=",")
                #     np.savetxt(f, table_data, delimiter=",", fmt="%s")
    
    def append_property_values(self, properties, m_idx, material_values, consolidated_dict):
        for prop in properties:
            if prop in consolidated_dict:
                material_values.append(consolidated_dict[prop][m_idx])


    #------ Setter Methods ------#
    
    @staticmethod
    def set_bounds_from_user_input(user_input: Dict, bound_key: str, property_docs: Dict[str, List[str]]):
        if bound_key not in ['upper_bound', 'lower_bound']:
            raise ValueError("bound_key must be either 'upper_bound' or 'lower_bound'.")
        
        bounds: Dict[str, Dict[str, List[float]]] = {}
        for material, properties in user_input.items():
            if material == 'mixture':  # Skip 'mixture' as it's not a material
                continue

            bounds[material] = {}
            for category, prop_list in property_docs.items():
                category_bounds = []

                for prop in prop_list:
                    if prop in properties and bound_key in properties[prop]:
                        # Append the specified bound if the property is found
                        category_bounds.append(properties[prop][bound_key])

                if category_bounds:
                    bounds[material][category] = category_bounds

        return bounds
    
    def set_lower_bounds(self, lower_bounds):
        self.lower_bounds = lower_bounds
        return self
    
    def set_upper_bounds(self, upper_bounds):
        self.upper_bounds = upper_bounds
        return self
    
    def set_retain_parents(self, retain_parents):
        self.retain_parents = retain_parents
        return self
    
    def set_allow_mutations(self, allow_mutations):
        self.allow_mutations = allow_mutations
        return self

    def set_property_docs(self, property_docs):
        self.property_docs = property_docs
        return self
    
    def set_desired_props(self, desired_props):
        self.desired_props = desired_props
        return self
    
    @staticmethod
    def set_desired_props_from_user_input(user_input: Dict, property_categories: List[str], property_docs: Dict):

        # Initialize the dictionary to hold the desired properties
        desired_props: Dict[str, List[float]] = {category: [] for category in property_categories}

        # Extracting the desired properties from the 'mixture' part of final_dict
        mixture_props = user_input.get('mixture', {})
        print(f"mixture_props = {mixture_props}")

        # Iterate through each property category and its associated properties
        for category, properties in property_docs.items():
            for prop in properties:
                # Check if the property is in the mixture; if so, append its desired value
                if prop in mixture_props:
                    desired_props[category].append(mixture_props[prop]['desired_prop'])

        return desired_props
    
    def set_has_props(self, has_props):
        self.has_props = has_props
        return self
    
    def set_fields(self, fields):
        self.fields = fields
        return self
    
    def set_num_properties(self, num_properties):
        self.num_properties = num_properties
        return num_properties
    
    @staticmethod
    def set_num_properties_from_desired_props(desired_props, lower_bounds):
        num_properties = 0

        # Iterate through property categories to count the total number of properties
        for _, properties in desired_props.items():
            num_properties += len(properties)  # Add the number of properties in each category

        # Multiply by the number of materials in the composite
        num_materials = len(lower_bounds)  # Assuming self.lower_bounds is correctly structured
        num_properties = num_properties * num_materials

        # Add variables for mixing parameter and volume fraction
        num_properties += 2

        return num_properties
    
    def set_ga_params(self, ga_params):
        self.ga_params = ga_params
        return self
    
    def set_final_population(self, final_pop):
        self.final_population = final_pop
        return self
    
    def set_cost_history(self, cost_history):
        self.cost_history = cost_history
        return self
    
    def set_lowest_costs(self, lowest_costs):
        self.lowest_costs = lowest_costs
        return self           

    def set_avg_parent_costs(self, avg_parent_costs):
        self.avg_parent_costs = avg_parent_costs
        return self
    
    def set_HS_optim_params(self):
        
        # MAIN OPTIMIZATION FUNCTION

        # Unpack necessary attributes from self
        num_parents = self.ga_params.get_num_parents()
        num_kids = self.ga_params.get_num_kids()
        num_generations = self.ga_params.get_num_generations()
        num_members = self.ga_params.get_num_members()
        
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
        population = Population(num_properties=self.num_properties, 
                                property_categories=self.property_categories, 
                                desired_props=self.desired_props, 
                                ga_params=self.ga_params, 
                                property_docs=self.property_docs, 
                                calc_guide=self.calc_guide)
        population.set_initial_random(self.lower_bounds, self.upper_bounds)

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

            print(f"Generation {g} of {num_generations}")
            costs[0:num_parents] = sorted_costs[0:num_parents] # retain the parents from the previous generation
            
            # Select top parents from population to be breeders
            for p in range(0, num_parents, 2):
                phi1, phi2 = np.random.rand(2)
                kid1 = phi1 * population.values[p, :] + (1-phi1) * population.values[p+1, :]
                kid2 = phi2 * population.values[p, :] + (1-phi2) * population.values[p+1, :]
                
                # Append offspring to population, overwriting old population members 
                population.values[num_parents+p,   :] = kid1
                population.values[num_parents+p+1, :] = kid2
                # Cast offspring to members and evaluate costs
                kid1 = Member(num_properties=self.num_properties, 
                              values=kid1, 
                              property_categories=self.property_categories, 
                              desired_props=self.desired_props, 
                              ga_params=self.ga_params, 
                              calc_guide=self.calc_guide, 
                              property_docs=self.property_docs)
                kid2 = Member(num_properties=self.num_properties, 
                              values=kid2, 
                              property_categories=self.property_categories, 
                              desired_props=self.desired_props, 
                              ga_params=self.ga_params, 
                              calc_guide=self.calc_guide, 
                              property_docs=self.property_docs)
                costs[num_parents+p]   = kid1.get_cost()
                costs[num_parents+p+1] = kid2.get_cost()
                        
            # Randomly generate new members to fill the rest of the population
            members_minus_parents_minus_kids = num_members - num_parents - num_kids
            population.set_new_random(members_minus_parents_minus_kids, self.lower_bounds, self.upper_bounds)

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

        # Update self attributes following optimization
        self.final_population = population
        self.cost_history = all_costs
        self.lowest_costs = lowest_costs
        self.avg_parent_costs = avg_parent_costs     
        
        return self         

    #------ Other Methods ------#

    def print_table_of_best_designs(self):

        table_data = self.get_table_of_best_designs()
        print("\nHASHIN-SHTRIKMAN + GENETIC ALGORITHM RECOMMENDED MATERIAL PROPERTIES")
        print(tabulate(table_data, headers=self.get_headers()))
    
    def plot_optimization_results(self):
        fig, ax = plt.subplots(figsize=(10,6))
        ax.plot(range(self.ga_params.get_num_generations()), self.avg_parent_costs, label="Avg. of top 10 performers")
        ax.plot(range(self.ga_params.get_num_generations()), self.lowest_costs, label="Best costs")
        plt.xlabel("Generation", fontsize= 20)
        plt.ylabel("Cost", fontsize=20)
        plt.title("Genetic Algorithm Results", fontsize = 24)
        plt.legend(fontsize = 14)
        plt.show()  
    
    def generate_consolidated_dict(self, total_docs = None):

        # MAIN FUNCTION USED TO GENERATE MATRIAL PROPERTY DICTIONARY DEPENDING ON USER REQUEST

        if "carrier-transport" in self.property_categories:
            client = Client(apikey=self.api_key, project=self.mp_contribs_project)
        else:
            client = Client(apikey=self.api_key)
        
        new_fields = copy.deepcopy(self.fields)

        # Iterate over the user input to dynamically update self.fields based on requested property categories
        # Iterate over property categories and update new_fields based on mp_property_docs
        for category in self.property_categories:
            if category in self.property_docs:
                if category == "carrier-transport":
                    new_fields["mp-ids-contrib"] = []

                for prop in self.property_docs[category]:
                    # Initialize empty list for each property under the category
                    new_fields[prop] = []

        self.set_fields(new_fields)

        logger.info(f"self.fields: {self.fields}")

        with MPRester(self.api_key) as mpr:
            
            docs = mpr.materials.summary.search(fields=self.fields)
            from mpi4py import MPI

            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()

            # Calculate the size of each chunk
            if total_docs is None:
                total_docs = len(docs)
                chunk_size = len(docs) // size
            elif isinstance(total_docs, int):
                chunk_size = total_docs // size

            # Calculate the start and end indices for this process"s chunk
            start = rank * chunk_size
            end = start + chunk_size if rank != size - 1 else total_docs  # The last process gets the remainder

            # Each process gets a different chunk
            chunk = docs[start:end]

            # for i, doc in enumerate(docs):
            for i, doc in enumerate(chunk[0:total_docs]):

                logger.info(f"Process {rank}: {i} of {len(chunk[0:total_docs])}")

                required_fields = [doc.material_id, doc.is_stable, doc.is_metal]

                for category in self.property_categories:
                    if category in self.property_docs:
                        for prop, value in self.property_docs[category].items():
                            if category == "carrier-transport":
                                try:
                                    mp_id = doc.material_id                           
                                    query = {"identifier": mp_id}
                                    my_dict = client.download_contributions(query=query, include=["tables"])[0]
                                    required_fields.append(my_dict["identifier"])
                                except IndexError:
                                    continue
                            elif category == "elastic":
                                prop_value = getattr(doc, prop, None)
                                # For "elastic", you expect 'value' to be directly usable
                                if value is not None:  # 'voigt' or similar
                                    if prop_value is not None:
                                        prop_value = prop_value.get(value, None)
                                required_fields.append(prop_value)
                            else:
                                # For other categories, just append the property name for now
                                prop_value = getattr(doc, prop, None)
                                required_fields.append(prop_value)
                
                # logger.info(f"required_fields = {required_fields}")

                if all(field is not None for field in required_fields):

                    for c in DEFAULT_FIELDS.keys():
                        self.fields[c].append(getattr(doc, c, None))
                    
                    for category in self.property_categories:
                        if category in self.property_docs:
                            if category == "carrier-transport":
                                self.fields["mp-ids-contrib"].append(my_dict["identifier"])
                                
                            for prop, value in self.property_docs[category].items():
                                
                                # Carrier transport
                                if category == "carrier-transport":
                                    self.mp_contribs_prop(prop, my_dict)
                                
                                # Elastic
                                elif category == "elastic":
                                    prop_value = getattr(doc, prop, None)
                                    if value is not None:  # 'voigt' or similar
                                        if prop_value is not None:
                                            prop_value = prop_value.get(value, None)
                                    self.fields[prop].append(prop_value)
                                
                                # All other property categories
                                else:
                                    for prop in self.property_docs[category]:
                                        # Dynamically get the property value from the doc
                                        prop_value = getattr(doc, prop, None)  # Returns None if prop doesn't exist
                                        # Initialize empty list for each property under the category
                                        self.fields[prop].append(prop_value)
        
            # comm.gather the self.fields data after the for loop
            gathered_fields = comm.gather(self.fields, root=0)

            # On process 0, consolidate self.fields data into a single dictionary -- consolidated_dict
            if rank == 0:
                consolidated_dict: Dict[str, List[Any]] = {}
                if gathered_fields is not None:
                    for fields in gathered_fields:
                        for key, value in fields.items():
                            if key in consolidated_dict:
                                consolidated_dict[key].extend(value)
                            else:
                                consolidated_dict[key] = value

                # Save the consolidated results to a JSON file
                now = datetime.now()
                my_file_name = f"{MODULE_DIR}/../io/outputs/consolidated_dict_" + now.strftime("%m_%d_%Y_%H_%M_%S")
                with open(my_file_name, "w") as my_file:
                    json.dump(consolidated_dict, my_file)

        return consolidated_dict
    
    def superscript_to_int(self, superscript_str):
        superscript_to_normal = {
            "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
            "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9"
        }
        normal_str = "".join(superscript_to_normal.get(char, char) for char in superscript_str)
        return int(normal_str)
    
    def mp_contribs_prop(self, prop, my_dict):
        if prop == "therm_cond_300k_low_doping":
            table_column = 7
        elif prop == "elec_cond_300k_low_doping":
            table_column = 5

        prop_str = my_dict["tables"][table_column].iloc[2, 0]
        if not isinstance(prop_str, str):
            prop_str = str(prop_str)
        prop_str = prop_str.replace(",", "")

        if "×10" in prop_str:
            # Extract the numeric part before the "±" symbol and the exponent
            prop_str, prop_exponent_str = re.search(r"\((.*?) ±.*?\)×10(.*)", prop_str).groups()
            # Convert the exponent part to a format that Python can understand
            prop_exponent = self.superscript_to_int(prop_exponent_str.strip())
            # Combine the numeric part and the exponent part, and convert the result to a float
            prop_value = float(f"{prop_str}e{prop_exponent}") * 1e-14  # multply by relaxation time, 10 fs
            logger.info(f"{prop}_if_statement = {prop_value}")
        else:
            prop_value = float(prop_str) * 1e-14  # multply by relaxation time, 10 fs
            logger.info(f"{prop}_else_statement = {prop_value}")
        
        self.fields[prop].append(prop_value)