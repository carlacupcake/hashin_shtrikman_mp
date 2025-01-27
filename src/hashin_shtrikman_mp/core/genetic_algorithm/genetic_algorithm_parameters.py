"""genetic_algo.py."""
from typing import Annotated
from pydantic import BaseModel, Field, PositiveInt


class GeneticAlgorithmParams(BaseModel):
    """
    Class to hold the parameters used for the genetic algorithm.
    """

    num_parents: PositiveInt = Field(
        default=10,
        description="Number of parent members to retain in each generation."
    )
    num_kids: PositiveInt = Field(
        default=10,
        description="Number of children to produce from the parent members."
    )
    num_generations: PositiveInt = Field(
        default=100,
        description="Total number of generations to simulate in the genetic algorithm."
    )
    num_members: PositiveInt = Field(
        default=200,
        description="Total number of members in each generation of the population."
    )
    mixing_param: Annotated[float, Field(strict=True, ge=0, le=1)] = Field(
        default = 0.5,
        description ="For linear scaling between effective min and max."
                     "It is recommended to use 0.5 in the absence of experimental data"
    )
    tolerance: Annotated[float, Field(strict=True, ge=0)] = Field(
        default=1.0,
        description="This parameter sets the threshold for considering the deviation of "
                    "concentration factors from their ideal values. It is used to adjust "
                    "the sensitivity of the cost function to variations in material "
                    "property concentrations, with a lower tolerance indicating "
                    "stricter requirements for concentration matching. In the cost "
                    "function, tolerance helps to determine the weight applied to "
                    "concentration factors, influencing the penalty for deviations in "
                    "material properties from their desired values."
    )
    weight_eff_prop: Annotated[float, Field(strict=True, ge=0)] = Field(
        default=1.0,
        description="This weight factor scales the importance of the effective property "
                    "matching component of the cost function. It determines how "
                    "strongly the difference between the calculated effective "
                    "properties of the composite material and the desired properties "
                    "influences the total cost. A higher value places more emphasis "
                    "on accurately matching these effective properties, aiming to "
                    "optimize the material composition towards achieving specific "
                    "property targets."
    )
    weight_conc_factor: Annotated[float, Field(strict=True, ge=0)] = Field(
        default=1.0,
        description="This weight factor controls the significance of the "
                    "concentration factor matching in the cost function. It scales "
                    "the penalty applied for deviations of concentration factors "
                    "from their ideal or tolerated levels, thus affecting the "
                    "optimization process's focus on material distribution within "
                    "the composite. A higher value means that achieving the "
                    "desired concentration balance between the composite's "
                    "constituents is more critical to minimizing the overall cost."
    )
