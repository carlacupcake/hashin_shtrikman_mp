#ifndef GENETIC_ALGO_H
#define GENETIC_ALGO_H

/*
Struct to hold the parameters used for the genetic algorithm.
*/
typedef struct {
    int num_parents;     // Number of parent members to retain in each generation. 
    int num_kids;        // Number of children to produce from the parent members.
    int num_generations; // Total number of generations to simulate in the genetic algorithm.
    int num_members;     // Total number of members in each generation of the population.
    double mixing_param; // TODO
    double tolerance;  /*
        This parameter sets the threshold for considering the deviation of
        concentration factors from their ideal values. It is used to adjust 
        the sensitivity of the cost function to variations in material 
        property concentrations, with a lower tolerance indicating 
        stricter requirements for concentration matching. In the cost 
        function, tolerance helps to determine the weight applied to 
        concentration factors, influencing the penalty for deviations in 
        material properties from their desired values.
        */
    double weight_eff_prop; /*
        This weight factor scales the importance of the effective property 
        matching component of the cost function. It determines how 
        strongly the difference between the calculated effective 
        properties of the composite material and the desired properties 
        influences the total cost. A higher value places more emphasis 
        on accurately matching these effective properties, aiming to 
        optimize the material composition towards achieving specific 
        property targets.
        */
    double weight_conc_factor; /*
        This weight factor controls the significance of the 
        concentration factor matching in the cost function. It scales 
        the penalty applied for deviations of concentration factors 
        from their ideal or tolerated levels, thus affecting the 
        optimization process's focus on material distribution within 
        the composite. A higher value means that achieving the 
        desired concentration balance between the composite's 
        constituents is more critical to minimizing the overall cost.
        */
} GAParams;

#endif // GENETIC_ALGO_H


    


