def compile_expressions(expressions_dict):
    compiled_expressions = {}
    for key, formula in expressions_dict.items():
        if isinstance(formula, str):
            # Replace placeholders with valid Python variable names
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
            compiled_expressions[key] = compile(compiled_formula, '<string>', 'eval')
        elif isinstance(formula, dict):
            # Recursively compile nested dictionaries
            compiled_expressions[key] = compile_expressions(formula)

    return compiled_expressions