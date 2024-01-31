import numpy as np
from ga_params_class import GAParams
from hs_logger import logger

class Member:

    def __init__(
            self,
            num_properties: int = 0,
            values:         np.ndarray = np.empty,
            property_docs:  list = [],
            desired_props:  dict={},
            ga_params:      GAParams = GAParams(),
            ):
            self.num_properties = num_properties
            self.values         = np.zeros(shape=(num_properties, 1)) if values is np.empty else values
            self.property_docs  = property_docs
            self.desired_props  = desired_props
            self.ga_params      = ga_params

    #------ Getter Methods ------#
    def get_num_properties(self):
        return self.num_properties
    
    def get_values(self):
        return self.values
    
    def get_property_docs(self):
        return self.property_docs
    
    def get_desired_props(self):
        return self.desired_props
    
    def get_ga_params(self):
        return self.ga_params
            
    def get_cost(self):

        # MAIN COST FUNCTION
        # Concentration factors guide (for 2 material composites only): Carrier transport (6), Dielectric (8), Elastic (6), Magnetic (4), Piezoelectic (2)

        # Extract attributes from self
        tolerance = self.ga_params.get_tolerance()
        weight_eff_prop = self.ga_params.get_weight_eff_prop()
        weight_conc_factor = self.ga_params.get_weight_conc_factor()

        # Extract mixing_param and volume fraction from self.values        
        mixing_param = self.values[-2]
        phase1_vol_frac = self.values[-1]
        phase2_vol_frac = 1 - phase1_vol_frac

        # Initialize effective property, concentration factor, and weight arrays
        # Initialize to zero so as not to contribute to cost if unchanged
        effective_properties = []
        concentration_factors = [] 
        cost_func_weights = []   
    
        idx = 0
        if "carrier-transport" in self.property_docs:

            # Extract electrical conductivities from member (property 2 should be larger)
            mat1_elec_cond, mat2_elec_cond = self.values[idx:idx+2]
            idx += 2 # could be generalized to num_materials in the future
            elec_cond1 = mat1_elec_cond if mat1_elec_cond <= mat2_elec_cond else mat2_elec_cond
            elec_cond2 = mat2_elec_cond if mat1_elec_cond <= mat2_elec_cond else mat1_elec_cond

            # Compute effective electrical conductivity mins and maxes with Hashin-Shtrikman
            if elec_cond1 == elec_cond2:
                effective_elec_cond_min = elec_cond1
                effective_elec_cond_max = elec_cond2
            else:
                effective_elec_cond_min = elec_cond1 + phase2_vol_frac / (1/(elec_cond2 - elec_cond1) + phase1_vol_frac/(3*elec_cond1))
                effective_elec_cond_max = elec_cond2 + phase1_vol_frac / (1/(elec_cond1 - elec_cond2) + phase2_vol_frac/(3*elec_cond2))

            # Compute concentration factors for electrical load sharing
            elec_cond_eff = mixing_param * effective_elec_cond_max + (1 - mixing_param) * effective_elec_cond_min
            effective_properties.append(elec_cond_eff)
            if elec_cond1 == elec_cond2:
                cf_current1_cf_elec1 = (1/phase1_vol_frac)**2 
                cf_current2_cf_elec2 = (1/phase2_vol_frac)**2 
            else:
                cf_current1_cf_elec1 = elec_cond1/elec_cond_eff * (1/phase1_vol_frac * (elec_cond2 - elec_cond_eff)/(elec_cond2 - elec_cond1))**2 
                cf_current2_cf_elec2 = elec_cond2/elec_cond_eff * (1/phase2_vol_frac * (elec_cond1 - elec_cond_eff)/(elec_cond1 - elec_cond2))**2 
            concentration_factors.append(cf_current1_cf_elec1)
            concentration_factors.append(cf_current2_cf_elec2)

            # Extract thermal conductivities from member
            mat1_therm_cond, mat2_therm_cond = self.values[idx:idx+2]
            idx += 2
            therm_cond1 = mat1_therm_cond if mat1_therm_cond <= mat2_therm_cond else mat2_therm_cond
            therm_cond2 = mat2_therm_cond if mat1_therm_cond <= mat2_therm_cond else mat1_therm_cond

            # Compute effective thermal conductivity mins and maxes with Hashin-Shtrikman
            if therm_cond1 == therm_cond2:
                effective_therm_cond_min = therm_cond1 
                effective_therm_cond_max = therm_cond2 
            else:
                effective_therm_cond_min = therm_cond1 + phase2_vol_frac / (1/(therm_cond2 - therm_cond1) + phase1_vol_frac/(3*therm_cond1))
                effective_therm_cond_max = therm_cond2 + phase1_vol_frac / (1/(therm_cond1 - therm_cond2) + phase2_vol_frac/(3*therm_cond2))            
            
            # Compute concentration factors for thermal load sharing
            # cf_temp for temperature gradient concentration factor
            # cf_heat for heat flux concentration factor
            therm_cond_eff = mixing_param * effective_therm_cond_max + (1 - mixing_param) * effective_therm_cond_min
            effective_properties.append(therm_cond_eff)
            if therm_cond1 == therm_cond2:
                cf_temp2 = 1/phase2_vol_frac
                cf_temp1 = 1/phase1_vol_frac 
            else:
                cf_temp2 = 1/phase2_vol_frac * (therm_cond_eff - therm_cond1) / (therm_cond2 - therm_cond1) 
                cf_temp1 = 1/phase1_vol_frac * (1 - phase2_vol_frac * cf_temp2) 
            cf_heat2 = therm_cond2 * cf_temp2 * 1/therm_cond_eff
            cf_heat1 = 1/phase1_vol_frac * (1 - phase2_vol_frac * cf_heat2)  
            concentration_factors.append(cf_temp1)
            concentration_factors.append(cf_temp2)
            concentration_factors.append(cf_heat1)
            concentration_factors.append(cf_heat2)

        if "dielectric" in self.property_docs:

            # Extract total dielectric constant from member 
            mat1_e_total, mat2_e_total = self.values[idx:idx+2]
            idx += 2
            e_total1 = mat1_e_total if mat1_e_total <= mat2_e_total else mat2_e_total
            e_total2 = mat2_e_total if mat1_e_total <= mat2_e_total else mat1_e_total

            # Compute effective total ielectric constant mins and maxes with Hashin-Shtrikman
            if e_total1 == e_total2:
                effective_e_total_min = e_total1 
                effective_e_total_max = e_total2 
            else:
                effective_e_total_min = e_total1 + phase2_vol_frac / (1/(e_total2 - e_total1) + phase1_vol_frac/(3*e_total1))
                effective_e_total_max = e_total2 + phase1_vol_frac / (1/(e_total1 - e_total2) + phase2_vol_frac/(3*e_total2))

            # Compute concentration factors for total dielectric load sharing
            # cf_pol for polarization concentration factor
            # cf_elec for electric field concentration factor
            e_total_eff = mixing_param * effective_e_total_max + (1 - mixing_param) * effective_e_total_min
            effective_properties.append(e_total_eff)
            if e_total1 == e_total2:
                cf_pol1_cf_elec1_tot = (1/phase1_vol_frac)**2 
                cf_pol2_cf_elec2_tot = (1/phase2_vol_frac)**2
            else:
                cf_pol1_cf_elec1_tot = e_total1/e_total_eff * (1/phase1_vol_frac * (e_total2 - e_total_eff)/(e_total2 - e_total1))**2 
                cf_pol2_cf_elec2_tot = e_total2/e_total_eff * (1/phase2_vol_frac * (e_total1 - e_total_eff)/(e_total1 - e_total2))**2 
            concentration_factors.append(cf_pol1_cf_elec1_tot)
            concentration_factors.append(cf_pol2_cf_elec2_tot)

            # Extract ionic dielectric constant from member 
            mat1_e_ionic, mat2_e_ionic = self.values[idx:idx+2]
            idx += 2
            e_ionic1 = mat1_e_ionic if mat1_e_ionic <= mat2_e_ionic else mat2_e_ionic
            e_ionic2 = mat2_e_ionic if mat1_e_ionic <= mat2_e_ionic else mat1_e_ionic

            # Compute effective ionic dielectric constant mins and maxes with Hashin-Shtrikman
            if e_ionic1 == e_ionic2:
                effective_e_ionic_min = e_ionic1 
                effective_e_ionic_max = e_ionic2 
            else:
                effective_e_ionic_min = e_ionic1 + phase2_vol_frac / (1/(e_ionic2 - e_ionic1) + phase1_vol_frac/(3*e_ionic1))
                effective_e_ionic_max = e_ionic2 + phase1_vol_frac / (1/(e_ionic1 - e_ionic2) + phase2_vol_frac/(3*e_ionic2))

            # Compute concentration factors for ionic dielectric load sharing
            e_ionic_eff = mixing_param * effective_e_ionic_max + (1 - mixing_param) * effective_e_ionic_min
            effective_properties.append(e_ionic_eff)
            if e_ionic1 == e_ionic2:
                cf_pol1_cf_elec1_ionic = (1/phase1_vol_frac)**2
                cf_pol2_cf_elec2_ionic = (1/phase2_vol_frac)**2
            else:
                cf_pol1_cf_elec1_ionic = e_ionic1/e_ionic_eff * (1/phase1_vol_frac * (e_ionic2 - e_ionic_eff)/(e_ionic2 - e_ionic1))**2
                cf_pol2_cf_elec2_ionic = e_ionic2/e_ionic_eff * (1/phase2_vol_frac * (e_ionic1 - e_ionic_eff)/(e_ionic1 - e_ionic2))**2
            concentration_factors.append(cf_pol1_cf_elec1_ionic)
            concentration_factors.append(cf_pol2_cf_elec2_ionic)

            # Extract electronic dielectric constant from member 
            mat1_e_elec, mat2_e_elec = self.values[idx:idx+2]
            idx += 2
            e_elec1 = mat1_e_elec if mat1_e_elec <= mat2_e_elec else mat2_e_elec
            e_elec2 = mat2_e_elec if mat1_e_elec <= mat2_e_elec else mat1_e_elec

            # Compute effective electronic dielectric constant mins and maxes with Hashin-Shtrikman
            if e_elec1 == e_elec2:
                effective_e_elec_min = e_elec1
                effective_e_elec_max = e_elec2
            else:
                effective_e_elec_min = e_elec1 + phase2_vol_frac / (1/(e_elec2 - e_elec1) + phase1_vol_frac/(3*e_elec1))
                effective_e_elec_max = e_elec2 + phase1_vol_frac / (1/(e_elec1 - e_elec2) + phase2_vol_frac/(3*e_elec2))

            # Compute concentration factors for electronic dielectric load sharing
            e_elec_eff = mixing_param * effective_e_elec_max + (1 - mixing_param) * effective_e_elec_min
            effective_properties.append(e_elec_eff)
            if e_elec1 == e_elec2:
                cf_pol1_cf_elec1_elec = (1/phase1_vol_frac)**2 
                cf_pol2_cf_elec2_elec = (1/phase2_vol_frac)**2
            else:
                cf_pol1_cf_elec1_elec = e_elec1/e_elec_eff * (1/phase1_vol_frac * (e_elec2 - e_elec_eff)/(e_elec2 - e_elec1))**2 
                cf_pol2_cf_elec2_elec = e_elec2/e_elec_eff * (1/phase2_vol_frac * (e_elec1 - e_elec_eff)/(e_elec1 - e_elec2))**2
            concentration_factors.append(cf_pol1_cf_elec1_elec)
            concentration_factors.append(cf_pol2_cf_elec2_elec)

            # Extract dielectric n from member 
            mat1_n, mat2_n = self.values[idx:idx+2]
            idx += 2
            n1 = mat1_n if mat1_n <= mat2_n else mat2_n
            n2 = mat2_n if mat1_n <= mat2_n else mat1_n

            # Compute effective dielectric n mins and maxes with Hashin-Shtrikman
            if n1 == n2:
                effective_n_min = n1 
                effective_n_max = n2 
            else:
                effective_n_min = n1 + phase2_vol_frac / (1/(n2 - n1) + phase1_vol_frac/(3*n1))
                effective_n_max = n2 + phase1_vol_frac / (1/(n1 - n2) + phase2_vol_frac/(3*n2))

            # Compute concentration factors for dielectric n load sharing
            n_eff = mixing_param * effective_n_max + (1 - mixing_param) * effective_n_min
            effective_properties.append(n_eff)
            if n1 == n2:
                cf_pol1_cf_elec1_n = (1/phase1_vol_frac)**2 
                cf_pol2_cf_elec2_n = (1/phase2_vol_frac)**2
            else:
                cf_pol1_cf_elec1_n = n1/n_eff * (1/phase1_vol_frac * (n2 - n_eff)/(n2 - n1))**2 
                cf_pol2_cf_elec2_n = n2/n_eff * (1/phase2_vol_frac * (n1 - n_eff)/(n1 - n2))**2
            concentration_factors.append(cf_pol1_cf_elec1_n)
            concentration_factors.append(cf_pol2_cf_elec2_n)

        if "elastic" in self.property_docs:

            # Extract bulk modulus and shear modulus from member 
            mat1_bulk_mod, mat2_bulk_mod = self.values[idx:idx+2]
            idx += 2
            bulk_mod1  = mat1_bulk_mod if mat1_bulk_mod <= mat2_bulk_mod else mat2_bulk_mod
            bulk_mod2  = mat2_bulk_mod if mat1_bulk_mod <= mat2_bulk_mod else mat1_bulk_mod

            mat1_shear_mod, mat2_shear_mod = self.values[idx:idx+2]
            idx += 2
            shear_mod1 = mat1_shear_mod if mat1_shear_mod <= mat2_shear_mod else mat2_shear_mod
            shear_mod2 = mat2_shear_mod if mat1_shear_mod <= mat2_shear_mod else mat1_shear_mod

            # Compute effective property mins and maxes with Hashin-Shtrikman
            if bulk_mod1 == bulk_mod2:
                effective_bulk_mod_min  = bulk_mod1
                effective_bulk_mod_max  = bulk_mod2
            else:
                effective_bulk_mod_min  = bulk_mod1 + phase2_vol_frac / (1/(bulk_mod2 - bulk_mod1) + 3*phase1_vol_frac/(3*bulk_mod1 + 4*shear_mod1))
                effective_bulk_mod_max  = bulk_mod2 + phase1_vol_frac / (1/(bulk_mod1 - bulk_mod2) + 3*phase2_vol_frac/(3*bulk_mod2 + 4*shear_mod2))
            if shear_mod1 == shear_mod2:
                effective_shear_mod_min = shear_mod1
                effective_shear_mod_max = shear_mod2
            else:
                effective_shear_mod_min = shear_mod1 + phase2_vol_frac / (1/(shear_mod2 - shear_mod1) + 6*phase1_vol_frac*(bulk_mod1 + 2*shear_mod1) / (5*shear_mod1*(3*bulk_mod1 + 4*shear_mod1)))
                effective_shear_mod_max = shear_mod2 + phase1_vol_frac / (1/(shear_mod1 - shear_mod2) + 6*phase2_vol_frac*(bulk_mod2 + 2*shear_mod2) / (5*shear_mod2*(3*bulk_mod2 + 4*shear_mod2)))

            # Compute concentration factors for mechanical load sharing
            # cf_bulk_mod for bulk modulus concentration factor
            # cf_shear_mod for shear modulus concentration factor
            bulk_mod_eff  = mixing_param * effective_bulk_mod_max  + (1 - mixing_param) * effective_bulk_mod_min
            shear_mod_eff = mixing_param * effective_shear_mod_max + (1 - mixing_param) * effective_shear_mod_min
            effective_properties.append(bulk_mod_eff)
            effective_properties.append(shear_mod_eff)
            if bulk_mod1 == bulk_mod2:
                cf_bulk_mod2 = 1/phase2_vol_frac
                cf_bulk_mod1 = 1/phase1_vol_frac
            else:
                cf_bulk_mod2 = 1/phase2_vol_frac * bulk_mod2/bulk_mod_eff * (bulk_mod_eff - bulk_mod1) / (bulk_mod2 - bulk_mod1)
                cf_bulk_mod1 = 1/phase1_vol_frac * (1 - phase2_vol_frac * cf_bulk_mod2)
            if shear_mod1 == shear_mod2:
                cf_shear_mod2 = 1/phase2_vol_frac
                cf_shear_mod1 = 1/phase1_vol_frac
            else:
                cf_shear_mod2 = 1/phase2_vol_frac * shear_mod2/shear_mod_eff * (shear_mod_eff - shear_mod1)/(shear_mod2 - shear_mod1)
                cf_shear_mod1 = 1/phase1_vol_frac * (1 - phase2_vol_frac * cf_shear_mod2)
            concentration_factors.append(cf_bulk_mod1)
            concentration_factors.append(cf_bulk_mod2)
            concentration_factors.append(cf_shear_mod1)
            concentration_factors.append(cf_shear_mod2)

            # Extract universal anisotropy from member 
            mat1_univ_aniso, mat2_univ_aniso = self.values[idx:idx+2]
            idx += 2
            univ_aniso1 = mat1_univ_aniso if mat1_univ_aniso <= mat2_univ_aniso else mat2_univ_aniso
            univ_aniso2 = mat2_univ_aniso if mat1_univ_aniso <= mat2_univ_aniso else mat1_univ_aniso

            # Compute effective univeral anisotropy mins and maxes with Hashin-Shtrikman
            if univ_aniso1 == univ_aniso2:
                effective_univ_aniso_min = univ_aniso1 
                effective_univ_aniso_max = univ_aniso2 
            else:
                effective_univ_aniso_min = univ_aniso1 + phase2_vol_frac / (1/(univ_aniso2 - univ_aniso1) + phase1_vol_frac/(3*univ_aniso1))
                effective_univ_aniso_max = univ_aniso2 + phase1_vol_frac / (1/(univ_aniso1 - univ_aniso2) + phase2_vol_frac/(3*univ_aniso2))

            # Compute concentration factors for anisotropy load sharing
            # cf_univ_aniso for universal anisotropy concentration factor
            univ_aniso_eff = mixing_param * effective_univ_aniso_max + (1 - mixing_param) * effective_univ_aniso_min
            effective_properties.append(univ_aniso_eff)
            if univ_aniso1 == univ_aniso2:
                cf_univ_aniso1 = (1/phase1_vol_frac)**2 
                cf_univ_aniso2 = (1/phase2_vol_frac)**2
            else:
                cf_univ_aniso1 = univ_aniso1/univ_aniso_eff * (1/phase1_vol_frac * (univ_aniso2 - univ_aniso_eff)/(univ_aniso2 - univ_aniso1))**2 
                cf_univ_aniso2 = univ_aniso2/univ_aniso_eff * (1/phase2_vol_frac * (univ_aniso1 - univ_aniso_eff)/(univ_aniso1 - univ_aniso2))**2
            concentration_factors.append(cf_univ_aniso1)
            concentration_factors.append(cf_univ_aniso2)

        if "magnetic" in self.property_docs:

            # Extract total magnetization from member 
            mat1_tot_mag, mat2_tot_mag = self.values[idx:idx+2]
            idx += 2
            tot_mag1 = mat1_tot_mag if mat1_tot_mag <= mat2_tot_mag else mat2_tot_mag
            tot_mag2 = mat2_tot_mag if mat1_tot_mag <= mat2_tot_mag else mat1_tot_mag

            # Compute effective magnetization mins and maxes with Hashin-Shtrikman
            if tot_mag1 == tot_mag2:
                effective_tot_mag_min = tot_mag1
                effective_tot_mag_max = tot_mag2
            else:
                effective_tot_mag_min = tot_mag1 + phase2_vol_frac / (1/(tot_mag2 - tot_mag1) + phase1_vol_frac/(3*tot_mag1))
                effective_tot_mag_max = tot_mag2 + phase1_vol_frac / (1/(tot_mag1 - tot_mag2) + phase2_vol_frac/(3*tot_mag2))

            # Compute concentration factors for magnetic load sharing
            # cf_temp for magnetization concentration factor
            tot_mag_eff = mixing_param * effective_tot_mag_max + (1 - mixing_param) * effective_tot_mag_min
            effective_properties.append(tot_mag_eff)
            if tot_mag1 == tot_mag2:
                cf_tot_mag1 = (1/phase1_vol_frac)**2 
                cf_tot_mag2 = (1/phase2_vol_frac)**2
            else:
                cf_tot_mag1 = tot_mag1/tot_mag_eff * (1/phase1_vol_frac * (tot_mag2 - tot_mag_eff)/(tot_mag2 - tot_mag1))**2 
                cf_tot_mag2 = tot_mag2/tot_mag_eff * (1/phase2_vol_frac * (tot_mag1 - tot_mag_eff)/(tot_mag1 - tot_mag2))**2
            concentration_factors.append(cf_tot_mag1)
            concentration_factors.append(cf_tot_mag2)

            # Extract total magnetization normalized volume from member 
            mat1_tot_mag_norm_vol, mat2_tot_mag_norm_vol = self.values[idx:idx+2]
            idx += 2
            tot_mag_norm_vol1 = mat1_tot_mag_norm_vol if mat1_tot_mag_norm_vol <= mat2_tot_mag_norm_vol else mat2_tot_mag_norm_vol
            tot_mag_norm_vol2 = mat2_tot_mag_norm_vol if mat1_tot_mag_norm_vol <= mat2_tot_mag_norm_vol else mat1_tot_mag_norm_vol

            # Compute effective property mins and maxes with Hashin-Shtrikman
            if tot_mag_norm_vol1 == tot_mag_norm_vol2:
                effective_tot_mag_norm_vol_min = tot_mag_norm_vol1 
                effective_tot_mag_norm_vol_max = tot_mag_norm_vol2 
            else:
                effective_tot_mag_norm_vol_min = tot_mag_norm_vol1 + phase2_vol_frac / (1/(tot_mag_norm_vol2 - tot_mag_norm_vol1) + phase1_vol_frac/(3*tot_mag_norm_vol1))
                effective_tot_mag_norm_vol_max = tot_mag_norm_vol2 + phase1_vol_frac / (1/(tot_mag_norm_vol1 - tot_mag_norm_vol2) + phase2_vol_frac/(3*tot_mag_norm_vol2))

            # Compute concentration factors for magnetic load sharing (normalized by volume)
            tot_mag_norm_vol_eff = mixing_param * effective_tot_mag_norm_vol_max + (1 - mixing_param) * effective_tot_mag_norm_vol_min
            effective_properties.append(tot_mag_norm_vol_eff)
            if tot_mag_norm_vol1 == tot_mag_norm_vol2:
                cf_tot_mag_norm_vol1 = (1/phase1_vol_frac)**2 
                cf_tot_mag_norm_vol2 = (1/phase2_vol_frac)**2
            else:
                cf_tot_mag_norm_vol1 = tot_mag_norm_vol1/tot_mag_norm_vol_eff * (1/phase1_vol_frac * (tot_mag_norm_vol2 - tot_mag_norm_vol_eff)/(tot_mag_norm_vol2 - tot_mag_norm_vol1))**2 
                cf_tot_mag_norm_vol2 = tot_mag_norm_vol2/tot_mag_norm_vol_eff * (1/phase2_vol_frac * (tot_mag_norm_vol1 - tot_mag_norm_vol_eff)/(tot_mag_norm_vol1 - tot_mag_norm_vol2))**2
            concentration_factors.append(cf_tot_mag_norm_vol1)
            concentration_factors.append(cf_tot_mag_norm_vol2)

        if "piezoelectric" in self.property_docs:
            
            # Extract total magnetization normalized volume from member 
            mat1_e_ij, mat2_e_ij = self.values[idx:idx+2]
            idx += 2
            e_ij1 = mat1_e_ij if mat1_e_ij <= mat2_e_ij else mat2_e_ij
            e_ij2 = mat2_e_ij if mat1_e_ij <= mat2_e_ij else mat1_e_ij

            # Compute effective property mins and maxes with Hashin-Shtrikman
            if e_ij1 == e_ij2:
                effective_e_ij_min = e_ij1 
                effective_e_ij_max = e_ij2 
            else:
                effective_e_ij_min = e_ij1 + phase2_vol_frac / (1/(e_ij2 - e_ij1) + phase1_vol_frac/(3*e_ij1))
                effective_e_ij_max = e_ij2 + phase1_vol_frac / (1/(e_ij1 - e_ij2) + phase2_vol_frac/(3*e_ij2))

            # Compute concentration factors for piezoelectric load sharing
            # cf_pol for polarization concentration factor
            # cf_elec for electric field concentration factor
            e_ij_eff = mixing_param * effective_e_ij_max + (1 - mixing_param) * effective_e_ij_min
            effective_properties.append(e_ij_eff)
            if e_ij1 == e_ij2:
                cf_pol1_cf_elec1_ij = (1/phase1_vol_frac)**2 
                cf_pol2_cf_elec2_ij = (1/phase2_vol_frac)**2
            else:
                cf_pol1_cf_elec1_ij = e_ij1/e_ij_eff * (1/phase1_vol_frac * (e_ij2 - e_ij_eff)/(e_ij2 - e_ij1))**2 
                cf_pol2_cf_elec2_ij = e_ij2/e_ij_eff * (1/phase2_vol_frac * (e_ij1 - e_ij_eff)/(e_ij1 - e_ij2))**2
            concentration_factors.append(cf_pol1_cf_elec1_ij)
            concentration_factors.append(cf_pol2_cf_elec2_ij)

        # Determine weights based on concentration factor magnitudes
        for i, factor in enumerate(concentration_factors):
            if (factor - tolerance) / tolerance > 0:
                cost_func_weights.append(weight_conc_factor)
            else:
                cost_func_weights.append(0)

        # Cast concentration factors, effective properties and weights to numpy arrays
        concentration_factors = np.array(concentration_factors)
        effective_properties = np.array(effective_properties)
        cost_func_weights = np.array(cost_func_weights)

        # Extract desired properties from dictionary
        des_props = []
        if "carrier-transport" in self.property_docs:
            des_props.extend(self.desired_props["carrier-transport"])
        if "dielectric" in self.property_docs:
            des_props.extend(self.desired_props["dielectric"])
        if "elastic" in self.property_docs:
            des_props.extend(self.desired_props["elastic"])
        if "magnetic" in self.property_docs:
            des_props.extend(self.desired_props["magnetic"]) 
        if "piezoelectric" in self.property_docs:
            des_props.extend(self.desired_props["piezoelectric"])
        des_props = np.array(des_props)

        # Assemble the cost function
        domains = len(self.property_docs)
        W = 1/domains
        cost = weight_eff_prop*W * np.sum(abs(np.divide(des_props - effective_properties, effective_properties))) + np.sum(np.multiply(cost_func_weights, abs(np.divide(concentration_factors - tolerance, tolerance))))

        return cost

    #------ Setter Methods ------#

    def set_num_properties(self, num_properties):
        self.num_properties = num_properties
        return self
    
    def set_values(self, values):
        self.values = values
        return self
    
    def set_property_docs(self, property_docs):
        self.property_docs = property_docs
        return self
    
    def set_desired_props(self, des_props):
        self.desired_props = des_props
        return self
    
    def set_ga_params(self, ga_params):
        self.ga_params = ga_params
        return self
    