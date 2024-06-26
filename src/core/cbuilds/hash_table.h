#ifndef HASH_TABLE_H
#define HASH_TABLE_H

#include <stddef.h>  // for size_t

// Define the HashEntry Struct:
// Each entry in the hash table will store a key and a value. 
// The key can be a string, and the value can be any type (using a void * pointer to allow for different value types).
typedef struct HashEntry {
    const char *key;
    char *value;
    struct HashEntry *next; // For handling collisions (linked list)
} HashEntry;

// Define the Hash Table Struct:
// The hash table will contain an array of pointers to HashEntry structs.
typedef struct HashTable {
    size_t size;
    HashEntry **buckets; // Array of pointers to HashEntry
} HashTable;

// Function prototypes

// Initialize a hash table
HashTable *create_table(int size);

// Insert a key-value pair into the hash table
void insert(HashTable *table, const char *key, void *value);

// Search for a value by key in the hash table
void *lookup(HashTable *table, const char *key);

// Delete a key-value pair from the hash table
void delete_table(HashTable *table, const char *key);

// Free the memory allocated for the hash table
void free_table(HashTable *table);

#endif // HASH_TABLE_H
