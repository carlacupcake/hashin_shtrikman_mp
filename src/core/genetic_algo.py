from pydantic import BaseModel, Field

class GAParams(BaseModel):
    """
    Class to hold the parameters used for the genetic algorithm.
    """
    num_parents: int = Field(
        default=10,
        description="Number of parent members to retain in each generation."
    )
    num_kids: int = Field(
        default=10,
        description="Number of children to produce from the parent members."
    )
    num_generations: int = Field(
        default=500,
        description="Total number of generations to simulate in the genetic algorithm."
    )
    num_members: int = Field(
        default=200,
        description="Total number of members in each generation of the population."
    )
    mixing_param: float = Field(
        default = 0.5,
        description = "TODO"
    )
    tolerance: float = Field(
        default=0.5,
        description="This parameter sets the threshold for considering the deviation of "
                    "concentration factors from their ideal values. It is used to adjust "
                    "the sensitivity of the cost function to variations in material "
                    "property concentrations, with a lower tolerance indicating "
                    "stricter requirements for concentration matching. In the cost "
                    "function, tolerance helps to determine the weight applied to "
                    "concentration factors, influencing the penalty for deviations in "
                    "material properties from their desired values."
    )
    weight_eff_prop: float = Field(
        default=10.0,
        description="This weight factor scales the importance of the effective property "
                    "matching component of the cost function. It determines how "
                    "strongly the difference between the calculated effective "
                    "properties of the composite material and the desired properties "
                    "influences the total cost. A higher value places more emphasis "
                    "on accurately matching these effective properties, aiming to "
                    "optimize the material composition towards achieving specific "
                    "property targets."
    )
    weight_conc_factor: float = Field(
        default=0.5,
        description="This weight factor controls the significance of the "
                    "concentration factor matching in the cost function. It scales "
                    "the penalty applied for deviations of concentration factors "
                    "from their ideal or tolerated levels, thus affecting the "
                    "optimization process's focus on material distribution within "
                    "the composite. A higher value means that achieving the "
                    "desired concentration balance between the composite's "
                    "constituents is more critical to minimizing the overall cost."
    )


    

