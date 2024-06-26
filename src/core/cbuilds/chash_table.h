#ifndef CHASH_TABLE_H
#define CHASH_TABLE_H

#include <stddef.h>  // for size_t

// Define the CHashEntry Struct:
// Each entry in the hash table will store a key and a value. 
// The key can be a string, and the value can be any type (using a void * pointer to allow for different value types).
typedef struct CHashEntry {
    const char *key;
    char *value;
    struct CHashEntry *next; // For handling collisions (linked list)
} CHashEntry;

// Define the CHashTable Struct:
// The hash table will contain an array of pointers to HashEntry structs.
typedef struct CHashTable {
    size_t size;
    CHashEntry **buckets; // Array of pointers to HashEntry
} CHashTable;

// Function prototypes

// Initialize a hash table
CHashTable *create_table(int size);

// Insert a key-value pair into the hash table
void insert(CHashTable *table, const char *key, void *value);

// Search for a value by key in the hash table
void *lookup(CHashTable *table, const char *key);

// Delete a key-value pair from the hash table
void delete_table(CHashTable *table, const char *key);

// Free the memory allocated for the hash table
void free_table(CHashTable *table);

#endif // CHASH_TABLE_H
