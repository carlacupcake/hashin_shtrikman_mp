# Class: `GeneticAlgorithmParams`

Class to hold the parameters used for the genetic algorithm. <br>
Built on the `Pydantic BaseModel`.

## Attributes

* `property_categories`, type: `list[str]` <br>
    List of property categories considered for optimization.

* `num_parents`, type: `PositiveInt` <br>
    Number of parent members to retain in each generation."

* `num_kids`, type: `PositiveInt` <br>
    Number of children to produce from the parent members.

* `num_generations`, type: `PositiveInt` <br>
    Total number of generations to simulate in the genetic algorithm.

* `num_members`, type: `PositiveInt` <br>
    Total number of members in each generation of the population.

* `mixing_param`, type: `Annotated[float, Field(strict=True, ge=0, le=1)]` <br>
    For linear scaling between effective min and max. It is recommended to use 0.5 in the absence of experimental data.

* `tolerance`, type: `Annotated[float, Field(strict=True, ge=0)]` <br>
    This parameter sets the threshold for considering the deviation of concentration factors from their ideal values. It is used to adjust the sensitivity of the cost function to variations in material property concentrations, with a lower tolerance indicating stricter requirements for concentration matching. In the cost function, tolerance helps to determine the weight applied to concentration factors, influencing the penalty for deviations in material properties from their desired values.

* `weight_eff_prop`, type: `Annotated[float, Field(strict=True, ge=0)]` <br>
    This weight factor scales the importance of the effective property matching component of the cost function. It determines how strongly the difference between the calculated effective properties of the composite material and the desired properties influences the total cost. A higher value places more emphasis on accurately matching these effective properties, aiming to optimize the material composition towards achieving specific property targets.

* `weight_conc_factor`, type: `Annotated[float, Field(strict=True, ge=0)]`
    This weight factor controls the significance of the concentration factor matching in the cost function. It scales the penalty applied for deviations of concentration factors from their ideal or tolerated levels, thus affecting the optimization process's focus on material distribution within the composite. A higher value means that achieving the desired concentration balance between the composite's constituents is more critical to minimizing the overall cost.
