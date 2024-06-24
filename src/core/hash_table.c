#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include "hash_table.h"

// Hash Function
unsigned int hash(const char* key, int size) {
    unsigned long int hashval = 0;
    int i = 0;
    while (hashval < ULONG_MAX && i < strlen(key)) {
        hashval = hashval << 8;
        hashval += key[i];
        i++;
    }
    return hashval % size;
}

// Initialize the hash table
HashTable *create_table(int size) {
    HashTable *table = malloc(sizeof(HashTable));
    table->buckets = malloc(sizeof(HashEntry *) * size);
    for (int i = 0; i < size; i++) {
        table->buckets[i] = NULL;
    }
    table->size = size;
    return table;
}

// Define insertion function for adding key-value pairs to the hash table
void insert(HashTable *table, const char *key, void *value) {
    unsigned int bucket = hash(key, table->size);
    HashEntry *new_HashEntry = malloc(sizeof(HashEntry));
    new_HashEntry->key = strdup(key);
    new_HashEntry->value = value;
    new_HashEntry->next = table->buckets[bucket];
    table->buckets[bucket] = new_HashEntry;
}

// Define lookup function for retrieving a value from a key
void *lookup(HashTable *table, const char *key) {
    unsigned int bucket = hash(key, table->size);
    HashEntry *HashEntry = table->buckets[bucket];
    while (HashEntry != NULL) {
        if (strcmp(HashEntry->key, key) == 0) {
            return HashEntry->value;
        }
        HashEntry = HashEntry->next;
    }
    return NULL;
}

// Define a deletion function for removing key-value pairs from the hash table
void delete_table(HashTable *table, const char *key) {
    unsigned int bucket = hash(key, table->size);
    HashEntry *entry = table->buckets[bucket];
    HashEntry *prev = NULL;
    
    while (entry != NULL && strcmp(entry->key, key) != 0) {
        prev = entry;
        entry = entry->next;
    }
    
    if (entry == NULL) return; // Key not found
    
    if (prev == NULL) {
        table->buckets[bucket] = entry->next;
    } else {
        prev->next = entry->next;
    }
    
    free((char *)entry->key);
    free(entry->value);
    free(entry);
}

// Free memory when done using the hash table
void free_table(HashTable *table) {
    for (int i = 0; i < table->size; i++) {
        HashEntry *entry = table->buckets[i];
        while (entry != NULL) {
            HashEntry *temp = entry;  // Define temp as a pointer to the current entry
            entry = entry->next;      // Move to the next entry in the bucket
            free((char *)temp->key);  // Free the memory allocated for the key
            free(temp->value);        // Free the memory allocated for the value
            free(temp);               // Free the memory allocated for the entry itself
        }
    }
    free(table->buckets);  // Free the memory allocated for the buckets array
    free(table);           // Free the memory allocated for the hash table
}
