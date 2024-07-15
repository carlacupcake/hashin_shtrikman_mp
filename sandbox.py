import yaml
from string import Template
import numexpr as ne

# Function to evaluate the YAML formula with given variables
def evaluate_yaml_formula(formula, **variables):
    # Ensure all variables are converted to float if they are numbers
    for key, value in variables.items():
        if isinstance(value, (int, float)):
            variables[key] = float(value)

    # Substitute the template placeholders with actual variable values
    template = Template(formula)
    substituted_formula = template.substitute(variables)

    # Evaluate the substituted formula using numexpr
    result = ne.evaluate(substituted_formula)
    return result

# Example usage
yaml_content = """
effective_props:
  eff_min: "{phase1} + {phase2_vol_frac} / (1/({phase2} - {phase1}) + {phase1_vol_frac}/(3*{phase1}))"
  eff_max: "{phase2} + {phase1_vol_frac} / (1/({phase1} - {phase2}) + {phase2_vol_frac}/(3*{phase2}))"
  eff_prop: "{mixing_param} * {eff_max}  + (1 - {mixing_param}) * {eff_min}"
  bulk_mod_min: "{phase1_bulk} + {phase2_vol_frac} / (1/({phase2_bulk} - {phase1_bulk}) + 3*{phase1_vol_frac}/(3*{phase1_bulk} + 4*{phase1_shear}))"
  bulk_mod_max: "{phase2_bulk} + {phase1_vol_frac} / (1/({phase1_bulk} - {phase2_bulk}) + 3*{phase2_vol_frac}/(3*{phase2_bulk} + 4*{phase2_shear}))"
  shear_mod_min: "{phase1_shear} + {phase2_vol_frac} / (1/({phase2_shear} - {phase1_shear}) + 6*{phase1_vol_frac}*({phase1_bulk} + 2*{phase1_shear}) / (5*{phase1_shear}*(3*{phase1_bulk} + 4*{phase1_shear})))"
  shear_mod_max: "{phase2_shear} + {phase1_vol_frac} / (1/({phase1_shear} - {phase2_shear}) + 6*{phase2_vol_frac}*({phase2_bulk} + 2*{phase2_shear}) / (5*{phase2_shear}*(3*{phase2_bulk} + 4*{phase2_shear})))"
concentration_factors:
  cf_1: "{phase1}/{effective_property} * (1/{phase1_vol_frac} * ({phase2} - {effective_property})/({phase2} - {phase1}))**2"
  cf_2: "{phase2}/{effective_property} * (1/{phase2_vol_frac} * ({phase1} - {effective_property})/({phase1} - {phase2}))**2"
  cf_2_elastic: "1/{phase2_vol_frac} * ({effective_property} - {phase1}) / ({phase2} - {phase1})"
  cf_1_elastic: "1/{phase1_vol_frac} * (1 - {phase2_vol_frac} * {cf_2_elastic})"
"""

calc_guide = yaml.safe_load(yaml_content)

phase1 = 1.0
phase2 = 2.0
phase1_vol_frac = 0.4
phase2_vol_frac = 0.6
mixing_param = 0.5
phase1_bulk = 1.5
phase2_bulk = 2.5
phase1_shear = 0.8
phase2_shear = 1.2
effective_property = 1.7

variables = {
    'phase1': phase1,
    'phase2': phase2,
    'phase1_vol_frac': phase1_vol_frac,
    'phase2_vol_frac': phase2_vol_frac,
    'mixing_param': mixing_param,
    'phase1_bulk': phase1_bulk,
    'phase2_bulk': phase2_bulk,
    'phase1_shear': phase1_shear,
    'phase2_shear': phase2_shear,
    'effective_property': effective_property
}

# Example calculation
effective_prop_min = evaluate_yaml_formula(calc_guide['effective_props']['eff_min'], **variables)
effective_prop_max = evaluate_yaml_formula(calc_guide['effective_props']['eff_max'], **variables)

print(f"Effective Min: {effective_prop_min}")
print(f"Effective Max: {effective_prop_max}")
