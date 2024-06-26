#ifndef MEMBER_H
#define MEMBER_H

/*
Struct to represent a member of the population in genetic algorithm optimization.
Stores the properties and configuration for genetic algorithm operations.
*/
typedef struct {
    int num_materials;           // Number of materials in the ultimate composite.
    int num_properties;          // Number of properties that each member of the population has.
    double* values;              // Values array representing the member's properties.          
    char** property_categories;  // List of property categories considered for optimization.
    int num_property_categories; // Number of property categories, needed for iteration. Not included in python Member class
    HashTable* property_docs;    // A hard coded yaml file containing property categories
    double* flat_des_props;      // Flattened list of lists of desired properties, converted from python dictionary
    int* lengths_des_props;      // Lengths of the sublists in the original list of lists of desired properties
    GAParams* ga_params;         // Parameter initilization class for the genetic algorithm.
    HashTable* calc_guide;       // Calculation guide for property evaluation from a hard coded yaml file.
} Member;

// Function prototypes

double get_cost(Member *self);

double* get_general_eff_props(Member* self, int idx, int* num_eff_props);

double* get_general_cfs(Member* self, int idx, int* num_cfs);

double* get_elastic_eff_props(Member* self, int idx, int* num_props);

double* get_elastic_cfs(Member* self, int idx, int* num_cfs); 


#endif // MEMBER_H