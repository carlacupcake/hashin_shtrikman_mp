def compile_formulas(formulas_dict):
    compiled_formulas = {}
    for key, formula in formulas_dict.items():
        if isinstance(formula, str):
            # List all variables used in cost_calculation_formulas.yaml as Python variables
            compiled_formula = formula.format(
                A1="A1",
                An="An",
                A1_term_i="A1_term_i",
                An_term_i="An_term_i",
                alpha_1="alpha_1",
                alpha_n="alpha_n",
                bulk_alpha_1="bulk_alpha_1",
                bulk_alpha_n="bulk_alpha_n",
                shear_alpha_1="shear_alpha_1",
                shear_alpha_n="shear_alpha_n",
                phase_1="phase_1",
                phase_i="phase_i",
                phase_n="phase_n",
                phase_1_vol_frac="phase_1_vol_frac",
                phase_i_vol_frac="phase_i_vol_frac",
                phase_n_vol_frac="phase_n_vol_frac",
                phase_1_bulk="phase_1_bulk",
                phase_1_shear="phase_1_shear",
                phase_i_elastic="phase_i_elastic",
                phase_n_bulk="phase_n_bulk",
                phase_n_shear="phase_n_shear",
                eff_max="eff_max",
                eff_min="eff_min",
                eff_prop="eff_prop",
                eff_elastic="eff_elastic",
                mixing_param="mixing_param",
                cf_load_i="cf_load_i",
                cf_response_i="cf_response_i",
                cf_elastic_i="cf_eleastic_i",
                vf_weighted_sum_cfs="vf_weighted_sum_cfs"
            )
            # Compile the formula
            compiled_formulas[key] = compile(compiled_formula, "<string>", "eval")
        elif isinstance(formula, dict):
            # Recursively compile nested dictionaries
            compiled_formulas[key] = compile_formulas(formula)

    return compiled_formulas
