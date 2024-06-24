// MAIN
int main() {
    // Create the hash table
    HashTable *calc_guide = create_table(100);

    // Parse the YAML file and populate the hash table
    parse_yaml_to_hash_table("cost_calculation_formulas.yaml", calc_guide);

    // Assuming Member struct definition and initialization
    Member member;
    member.calc_guide = calc_guide;

    // Define your variables
    double phase1_bulk = 1.0;
    double phase2_bulk = 2.0;
    double phase1_shear = 3.0;
    double phase2_shear = 4.0;
    double phase1_vol_frac = 0.5;
    double phase2_vol_frac = 0.5;

    // Evaluate a formula from the hash table
    double result = evaluate_formula(
        member.calc_guide, "effective_props.bulk_mod_min",
        "{phase1_bulk}", phase1_bulk,
        "{phase2_bulk}", phase2_bulk,
        "{phase1_shear}", phase1_shear,
        "{phase2_shear}", phase2_shear,
        "{phase1_vol_frac}", phase1_vol_frac,
        "{phase2_vol_frac}", phase2_vol_frac,
        NULL  // Terminate the variable list
    );

    printf("Result: %f\n", result);

    // Free resources (not shown here, but remember to free allocated memory)
    return 0;
}