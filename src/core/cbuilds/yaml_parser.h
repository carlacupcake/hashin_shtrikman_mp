#ifndef YAML_PARSER_H
#define YAML_PARSER_H

#include "chash_table.h"

// YAML parsing function
void parse_yaml_to_hash_table(const char *filename, CHashTable *table);

// YAML evaluate function
double evaluate_formula(CHashTable *table, const char *key, ... );

#endif // YAML_PARSER_H