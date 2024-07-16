def compile_formulas(formulas_dict):
    compiled_formulas = {}
    for key, formula in formulas_dict.items():
        if isinstance(formula, str):
            # List all variables used in cost_calculation_formulas.yaml as Python variables
            compiled_formula = formula.format(
                phase1='phase1', 
                phase2='phase2', 
                phase1_vol_frac='phase1_vol_frac', 
                phase2_vol_frac='phase2_vol_frac',
                mixing_param='mixing_param', 
                phase1_bulk='phase1_bulk', 
                phase2_bulk='phase2_bulk',
                phase1_shear='phase1_shear', 
                phase2_shear='phase2_shear', 
                effective_property='effective_property',
                eff_max='eff_max', 
                eff_min='eff_min', 
                cf_2_elastic='cf_2_elastic'
            )
            # Compile the formula
            compiled_formulas[key] = compile(compiled_formula, '<string>', 'eval')
        elif isinstance(formula, dict):
            # Recursively compile nested dictionaries
            compiled_formulas[key] = compile_formulas(formula)

    return compiled_formulas