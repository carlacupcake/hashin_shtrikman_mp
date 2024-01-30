import numpy as np
from ga_params_class import GAParams
from hs_logger import logger

class GeneticString:

    def __init__(
            self,
            dv: int = 0,
            values: np.ndarray = np.empty,
            property_docs: list = [],
            desired_props: dict={},
            ga_params:  GAParams = GAParams(),
            ):
            self.dv = dv
            self.values = np.array(shape=(dv,1)) if values is None else values
            self.property_docs = property_docs
            self.desired_props = desired_props
            self.ga_params = ga_params

    #------ Getter Methods ------#
    def get_dv(self):
        return self.dv
    
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

        # Concentration factors guide (for 2 material composites only):
        # Carrier transport (6): CJ1CE1, CJ2CE2, Ct1, Ct2, Cq1, Cq2
        # Dielectric (8): CP1E1_tot, CP2E2_tot, CP1E1_ionic, CP2E2_ionic, CP1E1_elec, CP2E2_elec, CP1E1_n, CP2E2_n
        # Elastic (6): Ck1, Ck2, Cmu1, Cmu2, CA1, CA2
        # Magnetic (4): CM1, CM2, CMvol1, CMvol2
        # Piezoelectic (2): Ceij1, Ceij2

        # Extract attributes from self
        TOL = self.ga_params.get_TOL()
        w1 = self.ga_params.get_w1()
        wj = self.ga_params.get_wj()

        # Extract gamma and volume fraction from self.values        
        gamma = self.values[-2]
        v1 = self.values[-1]
        v2 = 1 - v1

        # Initialize effective property, concentration factor, and weight arrays
        # Initialize to zero so as not to contribute to cost if unchanged
        effective_properties = []
        concentration_factors = [] 
        weights = []   
    
        idx = 0
        if "carrier-transport" in self.property_docs:

            # Extract electrical conductivities from genetic string (property 2 should be larger)
            Lam0_mat1, Lam0_mat2 = self.values[idx:idx+2]
            idx += 2 # could be generalized to num_materials in the future
            sig1 = Lam0_mat1 if Lam0_mat1 <= Lam0_mat2 else Lam0_mat2
            sig2 = Lam0_mat2 if Lam0_mat1 <= Lam0_mat2 else Lam0_mat1

            # Compute effective electrical conductivity mins and maxes with Hashin-Shtrikman
            if sig1 == sig2:
                effective_sig_min = sig1
                effective_sig_max = sig2
            else:
                effective_sig_min = sig1 + v2 / (1/(sig2 - sig1) + v1/(3*sig1))
                effective_sig_max = sig2 + v1 / (1/(sig1 - sig2) + v2/(3*sig2))

            # Compute concentration factors for electrical load sharing
            sig_eff = gamma * effective_sig_max + (1 - gamma) * effective_sig_min
            effective_properties.append(sig_eff)
            if sig1 == sig2:
                CJ1CE1 = (1/v1)**2 
                CJ2CE2 = (1/v2)**2 
            else:
                CJ1CE1 = sig1/sig_eff * (1/v1 * (sig2 - sig_eff)/(sig2 - sig1))**2 
                CJ2CE2 = sig2/sig_eff * (1/v2 * (sig1 - sig_eff)/(sig1 - sig2))**2 
            concentration_factors.append(CJ1CE1)
            concentration_factors.append(CJ2CE2)

            # Extract thermal conductivities from genetic string
            Lam1_mat1, Lam1_mat2 = self.values[idx:idx+2]
            idx += 2
            K1 = Lam1_mat1 if Lam1_mat1 <= Lam1_mat2 else Lam1_mat2
            K2 = Lam1_mat2 if Lam1_mat1 <= Lam1_mat2 else Lam1_mat1

            # Compute effective thermal conductivity mins and maxes with Hashin-Shtrikman
            if K1 == K2:
                effective_K_min = K1 
                effective_K_max = K2 
            else:
                effective_K_min = K1 + v2 / (1/(K2 - K1) + v1/(3*K1))
                effective_K_max = K2 + v1 / (1/(K1 - K2) + v2/(3*K2))            
            
            # Compute concentration factors for thermal load sharing
            K_eff = gamma * effective_K_max + (1 - gamma) * effective_K_min
            effective_properties.append(K_eff)
            if K1 == K2:
                Ct2 = 1/v2
                Ct1 = 1/v1 
            else:
                Ct2 = 1/v2 * (K_eff - K1) / (K2 - K1) 
                Ct1 = 1/v1 * (1 - v2 * Ct2) 
            Cq2 = K2 * Ct2 * 1/K_eff
            Cq1 = 1/v1 * (1 - v2 * Cq2)  
            concentration_factors.append(Ct1)
            concentration_factors.append(Ct2)
            concentration_factors.append(Cq1)
            concentration_factors.append(Cq2)

        if "dielectric" in self.property_docs:

            # Extract total dielectric constant from genetic string 
            Lam2_mat1, Lam2_mat2 = self.values[idx:idx+2]
            idx += 2
            epsTot1 = Lam2_mat1 if Lam2_mat1 <= Lam2_mat2 else Lam2_mat2
            epsTot2 = Lam2_mat2 if Lam2_mat1 <= Lam2_mat2 else Lam2_mat1

            # Compute effective total ielectric constant mins and maxes with Hashin-Shtrikman
            if epsTot1 == epsTot2:
                effective_epsTot_min = epsTot1 
                effective_epsTot_max = epsTot2 
            else:
                effective_epsTot_min = epsTot1 + v2 / (1/(epsTot2 - epsTot1) + v1/(3*epsTot1))
                effective_epsTot_max = epsTot2 + v1 / (1/(epsTot1 - epsTot2) + v2/(3*epsTot2))

            # Compute concentration factors for total dielectric load sharing
            epsTot_eff = gamma * effective_epsTot_max + (1 - gamma) * effective_epsTot_min
            effective_properties.append(epsTot_eff)
            if epsTot1 == epsTot2:
                CP1E1_tot = (1/v1)**2 
                CP2E2_tot = (1/v2)**2
            else:
                CP1E1_tot = epsTot1/epsTot_eff * (1/v1 * (epsTot2 - epsTot_eff)/(epsTot2 - epsTot1))**2 
                CP2E2_tot = epsTot2/epsTot_eff * (1/v2 * (epsTot1 - epsTot_eff)/(epsTot1 - epsTot2))**2 
            concentration_factors.append(CP1E1_tot)
            concentration_factors.append(CP2E2_tot)

            # Extract ionic dielectric constant from genetic string 
            Lam3_mat1, Lam3_mat2 = self.values[idx:idx+2]
            idx += 2
            epsIonic1 = Lam3_mat1 if Lam3_mat1 <= Lam3_mat2 else Lam3_mat2
            epsIonic2 = Lam3_mat2 if Lam3_mat1 <= Lam3_mat2 else Lam3_mat1

            # Compute effective ionic dielectric constant mins and maxes with Hashin-Shtrikman
            if epsIonic1 == epsIonic2:
                effective_epsIonic_min = epsIonic1 
                effective_epsIonic_max = epsIonic2 
            else:
                effective_epsIonic_min = epsIonic1 + v2 / (1/(epsIonic2 - epsIonic1) + v1/(3*epsIonic1))
                effective_epsIonic_max = epsIonic2 + v1 / (1/(epsIonic1 - epsIonic2) + v2/(3*epsIonic2))

            # Compute concentration factors for ionic dielectric load sharing
            epsIonic_eff = gamma * effective_epsIonic_max + (1 - gamma) * effective_epsIonic_min
            effective_properties.append(epsIonic_eff)
            if epsIonic1 == epsIonic2:
                CP1E1_ionic = (1/v1)**2
                CP2E2_ionic = (1/v2)**2
            else:
                CP1E1_ionic = epsIonic1/epsIonic_eff * (1/v1 * (epsIonic2 - epsIonic_eff)/(epsIonic2 - epsIonic1))**2
                CP2E2_ionic = epsIonic2/epsIonic_eff * (1/v2 * (epsIonic1 - epsIonic_eff)/(epsIonic1 - epsIonic2))**2
            concentration_factors.append(CP1E1_ionic)
            concentration_factors.append(CP2E2_ionic)

            # Extract electronic dielectric constant from genetic string 
            Lam4_mat1, Lam4_mat2 = self.values[idx:idx+2]
            idx += 2
            epsElec1 = Lam4_mat1 if Lam4_mat1 <= Lam4_mat2 else Lam4_mat2
            epsElec2 = Lam4_mat2 if Lam4_mat1 <= Lam4_mat2 else Lam4_mat1

            # Compute effective electronic dielectric constant mins and maxes with Hashin-Shtrikman
            if epsElec1 == epsElec2:
                effective_epsElec_min = epsElec1
                effective_epsElec_max = epsElec2
            else:
                effective_epsElec_min = epsElec1 + v2 / (1/(epsElec2 - epsElec1) + v1/(3*epsElec1))
                effective_epsElec_max = epsElec2 + v1 / (1/(epsElec1 - epsElec2) + v2/(3*epsElec2))

            # Compute concentration factors for electronic dielectric load sharing
            epsElec_eff = gamma * effective_epsElec_max + (1 - gamma) * effective_epsElec_min
            effective_properties.append(epsElec_eff)
            if epsElec1 == epsElec2:
                CP1E1_elec = (1/v1)**2 
                CP2E2_elec = (1/v2)**2
            else:
                CP1E1_elec = epsElec1/epsElec_eff * (1/v1 * (epsElec2 - epsElec_eff)/(epsElec2 - epsElec1))**2 
                CP2E2_elec = epsElec2/epsElec_eff * (1/v2 * (epsElec1 - epsElec_eff)/(epsElec1 - epsElec2))**2
            concentration_factors.append(CP1E1_elec)
            concentration_factors.append(CP2E2_elec)

            # Extract dielectric n from genetic string 
            Lam5_mat1, Lam5_mat2 = self.values[idx:idx+2]
            idx += 2
            epsn1 = Lam5_mat1 if Lam5_mat1 <= Lam5_mat2 else Lam5_mat2
            epsn2 = Lam5_mat2 if Lam5_mat1 <= Lam5_mat2 else Lam5_mat1

            # Compute effective dielectric n mins and maxes with Hashin-Shtrikman
            if epsn1 == epsn2:
                effective_epsn_min = epsn1 
                effective_epsn_max = epsn2 
            else:
                effective_epsn_min = epsn1 + v2 / (1/(epsn2 - epsn1) + v1/(3*epsn1))
                effective_epsn_max = epsn2 + v1 / (1/(epsn1 - epsn2) + v2/(3*epsn2))

            # Compute concentration factors for dielectric n load sharing
            epsn_eff = gamma * effective_epsn_max + (1 - gamma) * effective_epsn_min
            effective_properties.append(epsn_eff)
            if epsn1 == epsn2:
                CP1E1_n = (1/v1)**2 
                CP2E2_n = (1/v2)**2
            else:
                CP1E1_n = epsn1/epsn_eff * (1/v1 * (epsn2 - epsn_eff)/(epsn2 - epsn1))**2 
                CP2E2_n = epsn2/epsn_eff * (1/v2 * (epsn1 - epsn_eff)/(epsn1 - epsn2))**2
            concentration_factors.append(CP1E1_n)
            concentration_factors.append(CP2E2_n)

        if "elastic" in self.property_docs:

            # Extract bulk modulus and shear modulus from genetic string 
            Lam6_mat1, Lam6_mat2 = self.values[idx:idx+2]
            idx += 2
            k1  = Lam6_mat1 if Lam6_mat1 <= Lam6_mat2 else Lam6_mat2
            k2  = Lam6_mat2 if Lam6_mat1 <= Lam6_mat2 else Lam6_mat1

            Lam7_mat1, Lam7_mat2 = self.values[idx:idx+2]
            idx += 2
            mu1 = Lam7_mat1 if Lam7_mat1 <= Lam7_mat2 else Lam7_mat2
            mu2 = Lam7_mat2 if Lam7_mat1 <= Lam7_mat2 else Lam7_mat1

            # Compute effective property mins and maxes with Hashin-Shtrikman
            if k1 == k2:
                effective_k_min  = k1
                effective_k_max  = k2
            else:
                effective_k_min  = k1 + v2 / (1/(k2 -k1) + 3*v1/(3*k1 + 4*mu1))
                effective_k_max  = k2 + v1 / (1/(k1 -k2) + 3*v2/(3*k2 + 4*mu2))
            if mu1 == mu2:
                effective_mu_min = mu1
                effective_mu_max = mu2
            else:
                effective_mu_min = mu1 + v2 / (1/(mu2 - mu1) + 6*v1*(k1 + 2*mu1) / (5*mu1*(3*k1 + 4*mu1)))
                effective_mu_max = mu2 + v1 / (1/(mu1 - mu2) + 6*v2*(k2 + 2*mu2) / (5*mu2*(3*k2 + 4*mu2)))

            # Compute concentration factors for mechanical load sharing
            k_eff  = gamma * effective_k_max  + (1 - gamma) * effective_k_min
            mu_eff = gamma * effective_mu_max + (1 - gamma) * effective_mu_min
            effective_properties.append(k_eff)
            effective_properties.append(mu_eff)
            if k1 == k2:
                Ck2 = 1/v2
                Ck1 = 1/v1
            else:
                Ck2 = 1/v2 * k2/k_eff * (k_eff - k1) / (k2 - k1)
                Ck1 = 1/v1 * (1 - v2 * Ck2)
            if mu1 == mu2:
                Cmu2 = 1/v2
                Cmu1 = 1/v1
            else:
                Cmu2 = 1/v2 * mu2/mu_eff * (mu_eff - mu1)/(mu2 - mu1)
                Cmu1 = 1/v1 * (1 - v2 * Cmu2)
            concentration_factors.append(Ck1)
            concentration_factors.append(Ck2)
            concentration_factors.append(Cmu1)
            concentration_factors.append(Cmu2)

            # Extract universal anisotropy from genetic string 
            Lam8_mat1, Lam8_mat2 = self.values[idx:idx+2]
            idx += 2
            A1 = Lam8_mat1 if Lam8_mat1 <= Lam8_mat2 else Lam8_mat2
            A2 = Lam8_mat2 if Lam8_mat1 <= Lam8_mat2 else Lam8_mat1

            # Compute effective univeral anisotropy mins and maxes with Hashin-Shtrikman
            if A1 == A2:
                effective_A_min = A1 
                effective_A_max = A2 
            else:
                effective_A_min = A1 + v2 / (1/(A2 - A1) + v1/(3*A1))
                effective_A_max = A2 + v1 / (1/(A1 - A2) + v2/(3*A2))

            # Compute concentration factors for anisotropy load sharing
            A_eff = gamma * effective_A_max + (1 - gamma) * effective_A_min
            effective_properties.append(A_eff)
            if A1 == A2:
                CA1 = (1/v1)**2 
                CA2 = (1/v2)**2
            else:
                CA1 = A1/A_eff * (1/v1 * (A2 - A_eff)/(A2 - A1))**2 
                CA2 = A2/A_eff * (1/v2 * (A1 - A_eff)/(A1 - A2))**2
            concentration_factors.append(CA1)
            concentration_factors.append(CA2)

        if "magnetic" in self.property_docs:

            # Extract total magnetization from genetic string 
            Lam9_mat1, Lam9_mat2 = self.values[idx:idx+2]
            idx += 2
            M1 = Lam9_mat1 if Lam9_mat1 <= Lam9_mat2 else Lam9_mat2
            M2 = Lam9_mat2 if Lam9_mat1 <= Lam9_mat2 else Lam9_mat1

            # Compute effective magnetization mins and maxes with Hashin-Shtrikman
            if M1 == M2:
                effective_M_min = M1
                effective_M_max = M2
            else:
                effective_M_min = M1 + v2 / (1/(M2 - M1) + v1/(3*M1))
                effective_M_max = M2 + v1 / (1/(M1 - M2) + v2/(3*M2))

            # Compute concentration factors for magnetic load sharing
            M_eff = gamma * effective_M_max + (1 - gamma) * effective_M_min
            effective_properties.append(M_eff)
            if M1 == M2:
                CM1 = (1/v1)**2 
                CM2 = (1/v2)**2
            else:
                CM1 = M1/M_eff * (1/v1 * (M2 - M_eff)/(M2 - M1))**2 
                CM2 = M2/M_eff * (1/v2 * (M1 - M_eff)/(M1 - M2))**2
            concentration_factors.append(CM1)
            concentration_factors.append(CM2)

            # Extract total magnetization normalized volume from genetic string 
            Lam10_mat1, Lam10_mat2 = self.values[idx:idx+2]
            idx += 2
            Mvol1 = Lam10_mat1 if Lam10_mat1 <= Lam10_mat2 else Lam10_mat2
            Mvol2 = Lam10_mat2 if Lam10_mat1 <= Lam10_mat2 else Lam10_mat1

            # Compute effective property mins and maxes with Hashin-Shtrikman
            if Mvol1 == Mvol2:
                effective_Mvol_min = Mvol1 
                effective_Mvol_max = Mvol2 
            else:
                effective_Mvol_min = Mvol1 + v2 / (1/(Mvol2 - Mvol1) + v1/(3*Mvol1))
                effective_Mvol_max = Mvol2 + v1 / (1/(Mvol1 - Mvol2) + v2/(3*Mvol2))

            # Compute concentration factors for magnetic load sharing (normalized by volume)
            Mvol_eff = gamma * effective_Mvol_max + (1 - gamma) * effective_Mvol_min
            effective_properties.append(Mvol_eff)
            if Mvol1 == Mvol2:
                CMvol1 = (1/v1)**2 
                CMvol2 = (1/v2)**2
            else:
                CMvol1 = Mvol1/Mvol_eff * (1/v1 * (Mvol2 - Mvol_eff)/(Mvol2 - Mvol1))**2 
                CMvol2 = Mvol2/Mvol_eff * (1/v2 * (Mvol1 - Mvol_eff)/(Mvol1 - Mvol2))**2
            concentration_factors.append(CMvol1)
            concentration_factors.append(CMvol2)

        if "piezoelectric" in self.property_docs:
            
            # Extract total magnetization normalized volume from genetic string 
            Lam11_mat1, Lam11_mat2 = self.values[idx:idx+2]
            idx += 2
            epsij1 = Lam11_mat1 if Lam11_mat1 <= Lam11_mat2 else Lam11_mat2
            epsij2 = Lam11_mat2 if Lam11_mat1 <= Lam11_mat2 else Lam11_mat1

            # Compute effective property mins and maxes with Hashin-Shtrikman
            if epsij1 == epsij2:
                effective_epsij_min = epsij1 
                effective_epsij_max = epsij2 
            else:
                effective_epsij_min = epsij1 + v2 / (1/(epsij2 - epsij1) + v1/(3*epsij1))
                effective_epsij_max = epsij2 + v1 / (1/(epsij1 - epsij2) + v2/(3*epsij2))

            # Compute concentration factors for piezoelectric load sharing
            epsij_eff = gamma * effective_epsij_max + (1 - gamma) * effective_epsij_min
            effective_properties.append(epsij_eff)
            if epsij1 == epsij2:
                Ceij1 = (1/v1)**2 
                Ceij2 = (1/v2)**2
            else:
                Ceij1 = epsij1/epsij_eff * (1/v1 * (epsij2 - epsij_eff)/(epsij2 - epsij1))**2 
                Ceij2 = epsij2/epsij_eff * (1/v2 * (epsij1 - epsij_eff)/(epsij1 - epsij2))**2
            concentration_factors.append(Ceij1)
            concentration_factors.append(Ceij2)

        # Determine weights based on concentration factor magnitudes
        for i, factor in enumerate(concentration_factors):
            if (factor - TOL) / TOL > 0:
                weights.append(wj)
            else:
                weights.append(0)

        # Cast concentration factors, effective properties and weights to numpy arrays
        concentration_factors = np.array(concentration_factors)
        effective_properties = np.array(effective_properties)
        weights = np.array(weights)

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
        cost = w1*W * np.sum(abs(np.divide(des_props - effective_properties, effective_properties))) + np.sum(np.multiply(weights, abs(np.divide(concentration_factors - TOL, TOL))))

        return cost

    #------ Setter Methods ------#

    def set_dv(self, dv):
        self.dv = dv
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
    