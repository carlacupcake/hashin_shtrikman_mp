#ifndef YAML_PARSER_H
#define YAML_PARSER_H

#include "hash_table.h"

// YAML parsing function
void parse_yaml_to_hash_table(const char *filename, HashTable *table);

// YAML evaluate function
double evaluate_formula(HashTable *table, const char *key, ... );

#endif // YAML_PARSER_H