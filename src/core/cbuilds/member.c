#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <yaml.h>
#include <float.h>
#include "genetic_algo.h"
#include "hash_table.h"
#include "member.h"
#include "tinyexpr.h"
#include "yaml_parser.h"
#include <Python.h>

#define MAX_PROPERTIES 100

// Pure C version of get_cost for member
double get_cost(Member *self) {
    double tolerance = self->ga_params->tolerance;
    double weight_conc_factor = self->ga_params->weight_conc_factor;
    double weight_eff_prop = self->ga_params->weight_eff_prop;

    double effective_properties[MAX_PROPERTIES];  // Adjust size as needed
    double concentration_factors[MAX_PROPERTIES]; // Adjust size as needed
    double cost_func_weights[MAX_PROPERTIES];     // Adjust size as needed

    int idx = 0;

    for (int i = 0; i < self->num_property_categories; i++) {
        char *category = self->property_categories[i];
        
        if (strcmp(category, "elastic") == 0) {
            double *moduli_eff_props;
            double *moduli_cfs;

            // Assuming get_elastic_eff_props_and_cfs returns arrays of properties and factors
            int num_props = 2;
            int num_cfs = 4;
            moduli_eff_props = get_elastic_eff_props(self, idx, &num_props);
            moduli_cfs = get_elastic_cfs(self, idx, &num_cfs);

            for (int j = 0; j < num_props; j++) {
                effective_properties[idx] = moduli_eff_props[j];
                concentration_factors[idx] = moduli_cfs[j];
                idx++;
            }

            double *eff_univ_aniso;
            double *cfs_univ_aniso;
            int num_univ_aniso_props = 0; // Placeholder for the number of properties returned

            // Assuming get_general_eff_prop_and_cfs returns arrays of properties and factors
            eff_univ_aniso = get_general_eff_props(self, idx, &num_univ_aniso_props);
            cfs_univ_aniso = get_general_cfs(self, idx, &num_univ_aniso_props);

            effective_properties[idx] = *eff_univ_aniso;
            concentration_factors[idx] = *cfs_univ_aniso;
            idx++;

        } else {
            int num_docs = 0; // Placeholder for the number of documents in category
            HashTable *docs = (HashTable *)lookup(self->property_docs, category); // Assuming property_docs is a HashTable

            for (int p = idx; p < idx + num_docs; p++) {
                double *new_eff_props;
                double *new_cfs;
                int num_new_props = 0; // Placeholder for the number of properties returned

                new_eff_props = get_general_eff_props(self, p, &num_new_props);
                new_cfs = get_general_cfs(self, p, &num_new_props); 

                for (int j = 0; j < num_new_props; j++) {
                    effective_properties[idx] = new_eff_props[j];
                    concentration_factors[idx] = new_cfs[j];
                    idx++;
                }
            }

            idx += num_docs;
        }
    }

    // Determine weights based on concentration factor magnitudes
    for (int i = 0; i < idx; i++) {
        if ((concentration_factors[i] - tolerance) / tolerance > 0) {
            cost_func_weights[i] = weight_conc_factor;
        } else {
            cost_func_weights[i] = 0.0;
        }
    }

    // Calculate W
    double domains = self->num_property_categories;
    double W = 1.0 / domains;

    // Compute cost function in C
    double cost = 0.0;
    for (int i = 0; i < idx; i++) {
        cost += weight_eff_prop * W * fabs((self->desired_props[i] - effective_properties[i]) / effective_properties[i]);
        cost += cost_func_weights[i] * fabs((concentration_factors[i] - tolerance) / tolerance);
    }

    return cost;
}

double* get_general_eff_props(Member* self, int idx, int* num_eff_props) {
    // Initialize effective properties array
    double* effective_properties = (double*)calloc(1, sizeof(double)); // Only one effective property is returned
    if (!effective_properties) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    // Prepare indices for looping over properties
    int stop = self->num_properties - self->num_materials;
    int step = self->num_properties - 1;

    // Initialize variables for phase properties
    double phase1 = DBL_MAX, phase2 = -DBL_MAX;
    int phase1_idx = 0, phase2_idx = 0;

    // Loop to find min and max properties
    for (int i = idx; i < stop; i += step) {
        double prop = self->values[i];
        if (prop < phase1) {
            phase1 = prop;
            phase1_idx = i;
        }
        if (prop > phase2) {
            phase2 = prop;
            phase2_idx = i;
        }
    }

    // Extract volume fractions
    double phase1_vol_frac = self->values[self->num_properties - self->num_materials + phase1_idx];
    double phase2_vol_frac = self->values[self->num_properties - self->num_materials + phase2_idx];

    double effective_prop_min, effective_prop_max;

    // Compute effective property bounds with Hashin-Shtrikman
    if (phase1 == phase2) {
        effective_prop_min = phase1;
        effective_prop_max = phase2;
    } else {
        effective_prop_min = evaluate_formula(self->calc_guide, "effective_props.eff_min", phase1, phase2, phase1_vol_frac, phase2_vol_frac);
        effective_prop_max = evaluate_formula(self->calc_guide, "effective_props.eff_max", phase1, phase2, phase1_vol_frac, phase2_vol_frac);
    }

    // Compute concentration factors for electrical load sharing
    double mixing_param = self->ga_params->mixing_param;
    double effective_prop = mixing_param * effective_prop_max + (1 - mixing_param) * effective_prop_min;

    // Store the effective property
    effective_properties[0] = effective_prop;
    *num_eff_props = 1; // Only one effective property is returned

    return effective_properties;
}

double* get_general_cfs(Member* self, int idx, int* num_cfs) {
    // Initialize concentration factors array
    double* concentration_factors = (double*)calloc(2, sizeof(double)); // We need to store 2 concentration factors
    if (!concentration_factors) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    // Prepare indices for looping over properties
    int stop = self->num_properties - self->num_materials;
    int step = self->num_properties - 1;

    // Initialize variables for phase properties
    double phase1 = DBL_MAX, phase2 = -DBL_MAX;
    int phase1_idx = 0, phase2_idx = 0;

    // Loop to find min and max properties
    for (int i = idx; i < stop; i += step) {
        double prop = self->values[i];
        if (prop < phase1) {
            phase1 = prop;
            phase1_idx = i;
        }
        if (prop > phase2) {
            phase2 = prop;
            phase2_idx = i;
        }
    }

    // Extract volume fractions
    double phase1_vol_frac = self->values[self->num_properties - self->num_materials + phase1_idx];
    double phase2_vol_frac = self->values[self->num_properties - self->num_materials + phase2_idx];

    double effective_prop = 0.0; // Placeholder for the effective property
    double cf_response1_cf_load1, cf_response2_cf_load2;

    if (phase1_vol_frac == 0) {
        cf_response1_cf_load1 = pow(1 / phase2_vol_frac, 2);
        cf_response2_cf_load2 = pow(1 / phase2_vol_frac, 2);
    } else if (phase2_vol_frac == 0) {
        cf_response1_cf_load1 = pow(1 / phase1_vol_frac, 2);
        cf_response2_cf_load2 = pow(1 / phase1_vol_frac, 2);
    } else if (phase1 == phase2) {
        cf_response1_cf_load1 = pow(1 / phase1_vol_frac, 2);
        cf_response2_cf_load2 = pow(1 / phase2_vol_frac, 2);
    } else {
        cf_response1_cf_load1 = evaluate_formula(self->calc_guide, "concentration_factors.cf_1", phase1, phase2, phase1_vol_frac, effective_prop);
        cf_response2_cf_load2 = evaluate_formula(self->calc_guide, "concentration_factors.cf_2", phase1, phase2, phase2_vol_frac, effective_prop);
    }

    // Write over default calculation for concentration factor
    concentration_factors[0] = cf_response1_cf_load1;
    concentration_factors[1] = cf_response2_cf_load2;

    *num_cfs = 2; // We have two concentration factors

    return concentration_factors;
}

double* get_elastic_eff_props(Member* self, int idx, int* num_props) {
    
    // Initialize effective property array
    double* effective_properties = (double*)calloc(2, sizeof(double)); // Assuming we only need to store 2 properties
    if (!effective_properties) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    // Prepare indices for looping over properties
    int stop = -self->num_materials;
    int step = self->num_properties - 1;

    // Extract bulk moduli and shear moduli from member
    double phase1_bulk = DBL_MAX, phase2_bulk = -DBL_MAX;
    double phase1_shear = DBL_MAX, phase2_shear = -DBL_MAX;
    int phase1_bulk_idx = 0, phase2_bulk_idx = 0;
    int phase1_shear_idx = 0, phase2_shear_idx = 0;

    // Loop to find min and max bulk moduli
    for (int i = idx; i < self->num_properties + stop; i += step) {
        double bulk_mod = self->values[i];
        if (bulk_mod < phase1_bulk) {
            phase1_bulk = bulk_mod;
            phase1_bulk_idx = i;
        }
        if (bulk_mod > phase2_bulk) {
            phase2_bulk = bulk_mod;
            phase2_bulk_idx = i;
        }
    }

    // Loop to find min and max shear moduli
    for (int i = idx + 1; i < self->num_properties + stop; i += step) {
        double shear_mod = self->values[i];
        if (shear_mod < phase1_shear) {
            phase1_shear = shear_mod;
            phase1_shear_idx = i;
        }
        if (shear_mod > phase2_shear) {
            phase2_shear = shear_mod;
            phase2_shear_idx = i;
        }
    }

    // Check index consistency
    if ((phase1_bulk_idx != phase1_shear_idx) || (phase2_bulk_idx != phase2_shear_idx)) {
        fprintf(stderr, "Cannot perform optimization when phase indices are inconsistent.\n");
        exit(EXIT_FAILURE);
    }

    // Extract volume fractions
    double phase1_vol_frac = self->values[self->num_properties - self->num_materials + phase1_bulk_idx];
    double phase2_vol_frac = self->values[self->num_properties - self->num_materials + phase2_bulk_idx];

    // Get Hashin-Shtrikman effective properties for bulk and shear moduli
    double effective_bulk_mod_min, effective_bulk_mod_max;
    double effective_shear_mod_min, effective_shear_mod_max;

    if (phase1_bulk == phase2_bulk) {
        effective_bulk_mod_min = phase1_bulk;
        effective_bulk_mod_max = phase2_bulk;
    } else {
        effective_bulk_mod_min = evaluate_formula(self->calc_guide, "effective_props.bulk_mod_min", phase1_bulk, phase2_bulk, phase1_shear, phase2_shear, phase1_vol_frac, phase2_vol_frac);
        effective_bulk_mod_max = evaluate_formula(self->calc_guide, "effective_props.bulk_mod_max", phase1_bulk, phase2_bulk, phase1_shear, phase2_shear, phase1_vol_frac, phase2_vol_frac);
    }

    if (phase1_shear == phase2_shear) {
        effective_shear_mod_min = phase1_shear;
        effective_shear_mod_max = phase2_shear;
    } else {
        effective_shear_mod_min = evaluate_formula(self->calc_guide, "effective_props.shear_mod_min", phase1_bulk, phase2_bulk, phase1_shear, phase2_shear, phase1_vol_frac, phase2_vol_frac);
        effective_shear_mod_max = evaluate_formula(self->calc_guide, "effective_props.shear_mod_max", phase1_bulk, phase2_bulk, phase1_shear, phase2_shear, phase1_vol_frac, phase2_vol_frac);
    }

    // Compute concentration factors for mechanical load sharing
    double mixing_param = self->ga_params->mixing_param;
    double bulk_mod_eff = evaluate_formula(self->calc_guide, "effective_props.eff_prop", effective_bulk_mod_min, effective_bulk_mod_max, mixing_param);
    double shear_mod_eff = evaluate_formula(self->calc_guide, "effective_props.eff_prop", effective_shear_mod_min, effective_shear_mod_max, mixing_param);

    effective_properties[0] = bulk_mod_eff;
    effective_properties[1] = shear_mod_eff;

    *num_props = 2; // We have two effective properties

    return effective_properties;
}

double* get_elastic_cfs(Member* self, int idx, int* num_cfs) {
    // Initialize concentration factors array
    double* concentration_factors = (double*)calloc(4, sizeof(double)); // We need to store 4 concentration factors
    if (!concentration_factors) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    // Prepare indices for looping over properties
    int stop = self->num_properties - self->num_materials;
    int step = self->num_properties - 1;

    // Extract bulk moduli and shear moduli from member
    double phase1_bulk = DBL_MAX, phase2_bulk = -DBL_MAX;
    double phase1_shear = DBL_MAX, phase2_shear = -DBL_MAX;
    int phase1_bulk_idx = 0, phase2_bulk_idx = 0;
    int phase1_shear_idx = 0, phase2_shear_idx = 0;

    // Loop to find min and max bulk moduli
    for (int i = idx; i < stop; i += step) {
        double bulk_mod = self->values[i];
        if (bulk_mod < phase1_bulk) {
            phase1_bulk = bulk_mod;
            phase1_bulk_idx = i;
        }
        if (bulk_mod > phase2_bulk) {
            phase2_bulk = bulk_mod;
            phase2_bulk_idx = i;
        }
    }

    // Loop to find min and max shear moduli
    for (int i = idx + 1; i < stop; i += step) {
        double shear_mod = self->values[i];
        if (shear_mod < phase1_shear) {
            phase1_shear = shear_mod;
            phase1_shear_idx = i;
        }
        if (shear_mod > phase2_shear) {
            phase2_shear = shear_mod;
            phase2_shear_idx = i;
        }
    }

    // Check index consistency
    if ((phase1_bulk_idx != phase1_shear_idx) || (phase2_bulk_idx != phase2_shear_idx)) {
        fprintf(stderr, "Cannot perform optimization when phase indices are inconsistent.\n");
        exit(EXIT_FAILURE);
    }

    // Extract volume fractions
    double phase1_vol_frac = self->values[self->num_properties - self->num_materials + phase1_bulk_idx];
    double phase2_vol_frac = self->values[self->num_properties - self->num_materials + phase2_bulk_idx];

    double bulk_mod_eff = 0.0;  // effective bulk modulus
    double shear_mod_eff = 0.0; // effective shear modulus

    double cf_phase1_bulk, cf_phase2_bulk, cf_phase1_shear, cf_phase2_shear;

    if (phase1_vol_frac == 0) {
        cf_phase2_bulk  = 1 / phase2_vol_frac;
        cf_phase1_bulk  = 1 / phase2_vol_frac;
        cf_phase2_shear = 1 / phase2_vol_frac;
        cf_phase1_shear = 1 / phase2_vol_frac;
    } else if (phase2_vol_frac == 0) {
        cf_phase2_bulk  = 1 / phase1_vol_frac;
        cf_phase1_bulk  = 1 / phase1_vol_frac;
        cf_phase2_shear = 1 / phase1_vol_frac;
        cf_phase1_shear = 1 / phase1_vol_frac;
    } else if (phase1_bulk == phase2_bulk) {
        cf_phase2_bulk = 1 / phase2_vol_frac;
        cf_phase1_bulk = 1 / phase1_vol_frac;
    } else {
        cf_phase2_bulk = evaluate_formula(self->calc_guide, "concentration_factors.cf_2_elastic", phase2_vol_frac, phase1_bulk, phase2_bulk, bulk_mod_eff, 0);
        cf_phase1_bulk = evaluate_formula(self->calc_guide, "concentration_factors.cf_1_elastic", phase1_vol_frac, phase2_vol_frac, 0, 0, cf_phase2_bulk);
    }

    if (phase1_vol_frac == 0) {
        cf_phase2_shear = 1 / phase2_vol_frac;
        cf_phase1_shear = 1 / phase2_vol_frac;
    } else if (phase2_vol_frac == 0) {
        cf_phase2_shear = 1 / phase1_vol_frac;
        cf_phase1_shear = 1 / phase1_vol_frac;
    } else if (phase1_shear == phase2_shear) {
        cf_phase2_shear = 1 / phase2_vol_frac;
        cf_phase1_shear = 1 / phase1_vol_frac;
    } else {
        cf_phase2_shear = evaluate_formula(self->calc_guide, "concentration_factors.cf_2_elastic", phase2_vol_frac, phase1_shear, phase2_shear, shear_mod_eff, 0);
        cf_phase1_shear = evaluate_formula(self->calc_guide, "concentration_factors.cf_1_elastic", phase1_vol_frac, phase2_vol_frac, 0, 0, cf_phase2_shear);
    }

    // Write over default calculation for concentration factor
    concentration_factors[0] = cf_phase1_bulk;
    concentration_factors[1] = cf_phase1_shear;
    concentration_factors[2] = cf_phase2_bulk;
    concentration_factors[3] = cf_phase2_shear;

    *num_cfs = 4; // We have four concentration factors

    return concentration_factors;
}

// FOR PYTHON INTEGRATION
static PyObject *py_get_cost(PyObject *self, PyObject *args) {

    int num_materials;           
    int num_properties;          
    double* values;              
    char** property_categories;  
    int num_property_categories; 
    HashTable* property_docs;    
    double* desired_props;       
    GAParams* ga_params;         
    HashTable* calc_guide;       

    if (!PyArg_ParseTuple(args, "di", &num_materials, 
                                      &num_properties, 
                                      &values, 
                                      &property_categories, 
                                      &num_property_categories,
                                      &property_docs,
                                      &desired_props,
                                      &ga_params,
                                      &calc_guide)) {
        return NULL;
    }

    Member member = {num_materials, 
                     num_properties, 
                     values, 
                     property_categories, 
                     num_property_categories,
                     property_docs,
                     desired_props,
                     ga_params,
                     calc_guide};
    double cost = get_cost(&member);

    return PyFloat_FromDouble(cost);
}

static PyMethodDef methods[] = {
    {"get_cost", py_get_cost, METH_VARARGS, "Calculate cost using C function."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "mymodule",
    NULL,
    -1,
    methods
};

PyMODINIT_FUNC PyInit_mymodule(void) {
    return PyModule_Create(&module);
}