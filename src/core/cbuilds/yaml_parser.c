#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>
#include <limits.h>
#include <yaml.h>
#include <float.h>
#include "chash_table.h"
#include "tinyexpr.h"

// YAML parsing function
void parse_yaml_to_hash_table(const char *filename, CHashTable *table) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Could not open file: %s\n", filename);
        return;
    }

    yaml_parser_t parser;
    yaml_event_t event;

    if (!yaml_parser_initialize(&parser)) {
        fputs("Failed to initialize parser!\n", stderr);
        fclose(file);
        return;
    }

    yaml_parser_set_input_file(&parser, file);

    char *current_category = NULL;
    char *current_key = NULL;
    char *current_value = NULL;

    while (1) {
        if (!yaml_parser_parse(&parser, &event)) {
            fprintf(stderr, "Parser error %d\n", parser.error);
            goto cleanup;
        }

        if (event.type == YAML_MAPPING_START_EVENT) {
            current_category = NULL;
        } else if (event.type == YAML_SCALAR_EVENT) {
            char *value = (char *)event.data.scalar.value;
            if (!current_category) {
                current_category = strdup(value);
            } else if (!current_key) {
                current_key = strdup(value);
            } else {
                current_value = strdup(value);

                // Concatenate category and key
                char full_key[256];
                snprintf(full_key, sizeof(full_key), "%s.%s", current_category, current_key);

                // Insert into hash table
                insert(table, full_key, current_value);

                free(current_key);
                current_key = NULL;
                current_value = NULL;
            }
        }

        if (event.type == YAML_MAPPING_END_EVENT) {
            free(current_category);
            current_category = NULL;
        }

        if (event.type == YAML_STREAM_END_EVENT) {
            break;
        }

        yaml_event_delete(&event);
    }

cleanup:
    yaml_event_delete(&event);
    yaml_parser_delete(&parser);
    fclose(file);
}

// YAML evaluate function
double evaluate_formula(CHashTable *table, const char *key, ... ) {
    char *formula = (char *)lookup(table, key);
    if (!formula) {
        fprintf(stderr, "Formula for key %s not found.\n", key);
        return 0.0;
    }

    // Copy the formula to a mutable buffer
    char formula_buffer[1024];
    strncpy(formula_buffer, formula, sizeof(formula_buffer));   

    // Replace placeholders with actual values
    va_list args;
    va_start(args, key);
    
    const char *placeholder;
    double value;
    while ((placeholder = va_arg(args, const char *)) != NULL) {
        value = va_arg(args, double);
        
        // Convert value to string
        char value_str[64];
        snprintf(value_str, sizeof(value_str), "%g", value);
        
        // Replace all occurrences of placeholder in formula_buffer
        char *pos;
        while ((pos = strstr(formula_buffer, placeholder)) != NULL) {
            size_t len_before = pos - formula_buffer;
            size_t len_placeholder = strlen(placeholder);
            size_t len_value = strlen(value_str);
            size_t len_after = strlen(pos + len_placeholder);

            memmove(pos + len_value, pos + len_placeholder, len_after + 1);
            memcpy(pos, value_str, len_value);
        }
    }
    va_end(args);

    // Evaluate the final formula
    te_variable vars[] = { };
    te_expr *expr = te_compile(formula_buffer, vars, 0, 0);
    if (!expr) {
        fprintf(stderr, "Failed to compile expression: %s\n", formula_buffer);
        return 0.0;
    }

    double result = te_eval(expr);
    te_free(expr);

    return result;
}