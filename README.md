# Integration of Hashin-Shtrikman bounds using Genetic algorithm w/ Materials Project database

***Aim*** /n
The goal of this project is to determine the material properties and volume fractions of two materials in a binary mixture to meet a set of desired material properties. This involves integrating Hashin-Shtrikman (HS) bounds with a genetic algorithm, leveraging the Materials Project (MP) database.

***Workflow***
**User Input Initialization**: Instantiate a user input object with the desired material properties, including upper and lower search bounds for properties of both materials.

**HS Object Initialization**: Create a Hashin-Shtrikman object with Materials Project API credentials, user input, and the required material properties.

**Optimization Parameters**: Set the optimization parameters for the HS object, such as the total number of parents, children, members in a generation, and the number of generations.

**Population Initialization**: Instantiate a population object with the optimization parameters defined in the previous step.

**Random Property Assignment**: Randomly assign values to the properties of each population member using np.random, ensuring each material property is within the user-defined bounds.

**Cost Calculation**: Calculate the cost for each member of the population based on how well they meet the desired material properties.

**Selection of Top Performers**: Sort the costs in ascending order and retain the top "n" parents from the population of generation #1.

**Breeding and Offspring Generation**: Select the top "n" parents from the previous generation as breeders to create new offspring for generation #2.

**Cost Evaluation for New Generation**: Calculate the costs for all members of generation #2 and repeat the process of retaining the top "n" parents.

**Iteration Over Generations**: Repeat the breeding, offspring generation, and cost evaluation process for "g" number of generations.

**Cost vs. Generation Plotting**: Store and plot the minimum cost of a member and the average cost of all members of a population for each generation.

**Final Results Display**: After the final generation, display a table of material properties for the top "n" parents sorted in ascending order of their costs.

**Material Properties Dictionary Creation**: Generate a dictionary of mp_ids and their corresponding material properties of interest using the MP-API.

**Pairwise Population Creation**: For each pair of materials, create a population by setting the volume fraction randomly with np.random while maintaining all other properties.

**Top Pair Selection**: Sort the population in ascending order of costs and select the top-performing "j" parents.

**Display of Top-Performing Pairs**: Repeat the process for all possible pairs of materials and display the top-performing pairs along with their volume fractions.

***Implementation Notes***
Ensure you have valid credentials for the Materials Project API which you can find by registering yourself with Materials Project -- https://next-gen.materialsproject.org/
Optimization parameters (number of parents, children, etc.) should be chosen based on the complexity of the desired material properties and computational resources.
The genetic algorithm's efficiency and effectiveness can vary greatly based on the optimization parameters and the definition of the cost function.
Visualization of cost versus generation can provide insights into the convergence behavior of the genetic algorithm.


***Miscallaneous features***
**mpi4py support is added to append final_dict**

To take benefit of mpi parallelization, one can run the script as follows:

Install mpi4py via pip:
pip install mpi4py

In case installation via pip fails, you can use brew + pip instead:
brew install mpi4py
pip install mpi4py

Run the script as:
mpiexec -n 4 python orp_test.py
