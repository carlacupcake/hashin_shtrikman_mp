import numpy as np
from ga_params_class import GAParams
from hs_logger import logger

class GeneticString:

    def __init__(
            self,
            dv: int = 0,
            values: np.ndarray = np.empty,
            material_properties: list = [],
            desired_properties: list=[],
            ga_params:  GAParams = GAParams(),
            ):
            self.dv = dv
            self.values = np.array(shape=(dv,1)) if values is None else values
            self.material_properties = material_properties
            self.desired_properties = desired_properties
            self.ga_params = ga_params
            
    def get_cost(self):

        '''
        MAIN COST FUNCTION
        '''

        # Extract attributes from self
        TOL = self.ga_params.get_TOL()
        w1 = self.ga_params.get_w1()
        wj = self.ga_params.get_wj()

        # Extract material properties, gamma, and volume fraction from self.values
        [Lam0_mat1, Lam0_mat2, Lam1_mat1, Lam1_mat2,                                             # carrier transport
         Lam2_mat1, Lam2_mat2, Lam3_mat1, Lam3_mat2, Lam4_mat1, Lam4_mat2, Lam5_mat1, Lam5_mat2, # dielectric
         Lam6_mat1, Lam6_mat2, Lam7_mat1, Lam7_mat2, Lam8_mat1, Lam8_mat2,                       # elastic
         Lam9_mat1, Lam9_mat2, Lam10_mat1, Lam10_mat2,                                           # magnetic
         Lam11_mat1, Lam11_mat2,                                                                 # piezoelectric
         Lam12, Lam13] = self.values
        
        gamma = Lam12
        v1 = Lam13
        v2 = 1 - v1
        '''
        Lam0, electrical conductivity, [S/m]
        Lam1, thermal conductivity, [W/m/K]
        Lam2, total dielectric constant, [F/m]
        Lam3, ionic contrib dielectric constant, [F/m]
        Lam4, electronic contrib dielectric constant, [F/m]
        Lam5, dielectric n, [F/m]
        Lam6, bulk modulus, [GPa]
        Lam7, shear modulus, [GPa]
        Lam8, universal anisotropy, []
        Lam9, total magnetization, []
        Lam10, total magnetization normalized volume, []
        Lam11, piezoelectric constant, [C/N or m/V]
        Lam12, gamma, the avergaing parameter, []
        Lam13, volume fraction of phase 1, [] 
        '''

        # Initialize effective property, concentration factor, and weight arrays
        # Initialize to zero so as not to contribute to cost if unchanged
        effective_properties = np.zeros(12)
        concentration_factors = np.zeros(26) 
        weights = np.zeros(26)
        '''
        Concentration factors (26 total):
        Carrier transport (6): CJ1CE1, CJ2CE2, Ct1, Ct2, Cq1, Cq2
        Dielectric (8): CP1E1_tot, CP2E2_tot, CP1E1_ionic, CP2E2_ionic, CP1E1_elec, CP2E2_elec, CP1E1_n, CP2E2_n
        Elastic (6): Ck1, Ck2, Cmu1, Cmu2, CA1, CA2
        Magnetic (4): CM1, CM2, CMvol1, CMvol2
        Piezoelectic (2): Ceij1, Ceij2
        '''
    
        if "carrier-transport" in self.material_properties:

            # Extract electrical conductivities from genetic string (property 2 should be larger)
            sig1 = Lam0_mat1 if Lam0_mat1 <= Lam0_mat2 else Lam0_mat2
            sig2 = Lam0_mat2 if Lam0_mat1 <= Lam0_mat2 else Lam0_mat1

            # Compute effective electrical conductivity mins and maxes with Hashin-Shtrikman
            effective_sig_min = sig1 + v2 / (1/(sig2 - sig1) + v1/(3*sig1))
            effective_sig_max = sig2 + v1 / (1/(sig1 - sig2) + v2/(3*sig2))

            # Compute concentration factors for electrical load sharing
            sig_eff = gamma * effective_sig_max + (1 - gamma) * effective_sig_min
            effective_properties[0] = sig_eff
            CJ1CE1 = sig1/sig_eff * (1/v1 * (sig2 - sig_eff)/(sig2 - sig1))**2 
            CJ2CE2 = sig2/sig_eff * (1/v2 * (sig1 - sig_eff)/(sig1 - sig2))**2 
            concentration_factors[0] = CJ1CE1
            concentration_factors[1] = CJ2CE2

            # Extract thermal conductivities from genetic string
            K1 = Lam1_mat1 if Lam1_mat1 <= Lam1_mat2 else Lam1_mat2
            K2 = Lam1_mat2 if Lam1_mat1 <= Lam1_mat2 else Lam1_mat1

            # Compute effective thermal conductivity mins and maxes with Hashin-Shtrikman
            effective_K_min = K1 + v2 / (1/(K2 - K1) + v1/(3*K1))
            effective_K_max = K2 + v1 / (1/(K1 - K2) + v2/(3*K2))            
            
            # Compute concentration factors for thermal load sharing
            K_eff = gamma * effective_K_max + (1 - gamma) * effective_K_min
            effective_properties[1] = K_eff
            Ct2 = 1/v2 * (K_eff - K1) / (K2 - K1) # Ct2
            Ct1 = 1/v1 * (1 - v2 * Ct2) # Ct1
            Cq2 = K2 * Ct2 * 1/K_eff
            Cq1 = 1/v1 * (1 - v2 * Cq2)  
            concentration_factors[2] = Ct1
            concentration_factors[3] = Ct2
            concentration_factors[4] = Cq1
            concentration_factors[5] = Cq2

        if "dielectric" in self.material_properties:

            # TODO check concentration factor formulas

            # Extract total dielectric constant from genetic string 
            epsTot1 = Lam2_mat1 if Lam2_mat1 <= Lam2_mat2 else Lam2_mat2
            epsTot2 = Lam2_mat2 if Lam2_mat1 <= Lam2_mat2 else Lam2_mat1

            # Compute effective total ielectric constant mins and maxes with Hashin-Shtrikman
            effective_epsTot_min = epsTot1 + v2 / (1/(epsTot2 - epsTot1) + v1/(3*epsTot1))
            effective_epsTot_max = epsTot2 + v1 / (1/(epsTot1 - epsTot2) + v2/(3*epsTot2))

            # Compute concentration factors for total dielectric load sharing
            epsTot_eff = gamma * effective_epsTot_max + (1 - gamma) * effective_epsTot_min
            effective_properties[2] = epsTot_eff
            CP1E1_tot = epsTot1/epsTot_eff * (1/v1 * (epsTot2 - epsTot_eff)/(epsTot2 - epsTot1))**2 
            CP2E2_tot = epsTot2/epsTot_eff * (1/v2 * (epsTot1 - epsTot_eff)/(epsTot1 - epsTot2))**2 
            concentration_factors[6] = CP1E1_tot 
            concentration_factors[7] = CP2E2_tot

            # Extract ionic dielectric constant from genetic string 
            epsIonic1 = Lam3_mat1 if Lam3_mat1 <= Lam3_mat2 else Lam3_mat2
            epsIonic2 = Lam3_mat2 if Lam3_mat1 <= Lam3_mat2 else Lam3_mat1

            # Compute effective ionic dielectric constant mins and maxes with Hashin-Shtrikman
            effective_epsIonic_min = epsIonic1 + v2 / (1/(epsIonic2 - epsIonic1) + v1/(3*epsIonic1))
            effective_epsIonic_max = epsIonic2 + v1 / (1/(epsIonic1 - epsIonic2) + v2/(3*epsIonic2))

            # Compute concentration factors for ionic dielectric load sharing
            epsIonic_eff = gamma * effective_epsIonic_max + (1 - gamma) * effective_epsIonic_min
            effective_properties[3] = epsIonic_eff
            CP1E1_ionic = epsIonic1/epsIonic_eff * (1/v1 * (epsIonic2 - epsIonic_eff)/(epsIonic2 - epsIonic1))**2
            CP2E2_ionic = epsIonic2/epsIonic_eff * (1/v2 * (epsIonic1 - epsIonic_eff)/(epsIonic1 - epsIonic2))**2
            concentration_factors[8] = CP1E1_ionic 
            concentration_factors[9] = CP2E2_ionic

            # Extract electronic dielectric constant from genetic string 
            epsElec1 = Lam4_mat1 if Lam4_mat1 <= Lam4_mat2 else Lam4_mat2
            epsElec2 = Lam4_mat2 if Lam4_mat1 <= Lam4_mat2 else Lam4_mat1

            # Compute effective electronic dielectric constant mins and maxes with Hashin-Shtrikman
            effective_epsElec_min = epsElec1 + v2 / (1/(epsElec2 - epsElec1) + v1/(3*epsElec1))
            effective_epsElec_max = epsElec2 + v1 / (1/(epsElec1 - epsElec2) + v2/(3*epsElec2))

            # Compute concentration factors for electronic dielectric load sharing
            epsElec_eff = gamma * effective_epsElec_max + (1 - gamma) * effective_epsElec_min
            effective_properties[4] = epsElec_eff
            CP1E1_elec = epsElec1/epsElec_eff * (1/v1 * (epsElec2 - epsElec_eff)/(epsElec2 - epsElec1))**2 
            CP2E2_elec = epsElec2/epsElec_eff * (1/v2 * (epsElec1 - epsElec_eff)/(epsElec1 - epsElec2))**2
            concentration_factors[10] = CP1E1_elec
            concentration_factors[11] = CP2E2_elec

            # Extract dielectric n from genetic string 
            epsn1 = Lam5_mat1 if Lam5_mat1 <= Lam5_mat2 else Lam5_mat2
            epsn2 = Lam5_mat2 if Lam5_mat1 <= Lam5_mat2 else Lam5_mat1

            # Compute effective dielectric n mins and maxes with Hashin-Shtrikman
            effective_epsn_min = epsn1 + v2 / (1/(epsn2 - epsn1) + v1/(3*epsn1))
            effective_epsn_max = epsn2 + v1 / (1/(epsn1 - epsn2) + v2/(3*epsn2))

            # Compute concentration factors for dielectric n load sharing
            epsn_eff = gamma * effective_epsn_max + (1 - gamma) * effective_epsn_min
            effective_properties[5] = epsn_eff
            CP1E1_n = epsn1/epsn_eff * (1/v1 * (epsn2 - epsn_eff)/(epsn2 - epsn1))**2 
            CP2E2_n = epsn2/epsn_eff * (1/v2 * (epsn1 - epsn_eff)/(epsn1 - epsn2))**2
            concentration_factors[12] = CP1E1_n
            concentration_factors[13] = CP2E2_n

        if "elastic" in self.material_properties:

            # Extract bulk modulus and shear modulus from genetic string 
            k1  = Lam6_mat1 if Lam6_mat1 <= Lam6_mat2 else Lam6_mat2
            k2  = Lam6_mat2 if Lam6_mat1 <= Lam6_mat2 else Lam6_mat1
            mu1 = Lam7_mat1 if Lam7_mat1 <= Lam7_mat2 else Lam7_mat2
            mu2 = Lam7_mat2 if Lam7_mat1 <= Lam7_mat2 else Lam7_mat1

            # Compute effective property mins and maxes with Hashin-Shtrikman
            effective_k_min  = k1 + v2 / (1/(k2 -k1) + 3*v1/(3*k1 + 4*mu1))
            effective_k_max  = k2 + v1 / (1/(k1 -k2) + 3*v2/(3*k2 + 4*mu2))
            effective_mu_min = mu1 + v2 / (1/(mu2 - mu1) + 6*v1*(k1 + 2*mu1) / (5*mu1*(3*k1 + 4*mu1)))
            effective_mu_max = mu2 + v1 / (1/(mu1 - mu2) + 6*v2*(k2 + 2*mu2) / (5*mu2*(3*k2 + 4*mu2)))

            # Compute concentration factors for mechanical load sharing
            k_eff  = gamma * effective_k_max  + (1 - gamma) * effective_k_min
            mu_eff = gamma * effective_mu_max + (1 - gamma) * effective_mu_min
            effective_properties[6] = k_eff
            effective_properties[7] = mu_eff
            Ck2 = 1/v2 * k2/k_eff * (k_eff - k1) / (k2 - k1) 
            Cmu2 = 1/v2 * mu2/mu_eff * (mu_eff - mu1)/(mu2 - mu1)
            Ck1 = 1/v1 * (1 - v2 * Ck2)
            Cmu1 = 1/v1 * (1 - v2 * Cmu2)
            concentration_factors[14] = Ck1
            concentration_factors[15] = Ck2
            concentration_factors[16] = Cmu1
            concentration_factors[17] = Cmu2

            # Extract universal anisotropy from genetic string 
            A1 = Lam8_mat1 if Lam8_mat1 <= Lam8_mat2 else Lam8_mat2
            A2 = Lam8_mat2 if Lam8_mat1 <= Lam8_mat2 else Lam8_mat1

            # Compute effective univeral anisotropy mins and maxes with Hashin-Shtrikman
            effective_A_min = A1 + v2 / (1/(A2 - A1) + v1/(3*A1))
            effective_A_max = A2 + v1 / (1/(A1 - A2) + v2/(3*A2))

            # Compute concentration factors for anisotropy load sharing
            # TODO check concentration factor formulas
            A_eff = gamma * effective_A_max + (1 - gamma) * effective_A_min
            effective_properties[8] = A_eff
            CA1 = A1/A_eff * (1/v1 * (A2 - A_eff)/(A2 - A1))**2 
            CA2 = A2/A_eff * (1/v2 * (A1 - A_eff)/(A1 - A2))**2
            concentration_factors[18] = CA1
            concentration_factors[19] = CA2

        if "magnetic" in self.material_properties:

            # TODO check concentration factor formulas

            # Extract total magnetization from genetic string 
            M1 = Lam9_mat1 if Lam9_mat1 <= Lam9_mat2 else Lam9_mat2
            M2 = Lam9_mat2 if Lam9_mat1 <= Lam9_mat2 else Lam9_mat1

            # Compute effective magnetization mins and maxes with Hashin-Shtrikman
            effective_M_min = M1 + v2 / (1/(M2 - M1) + v1/(3*M1))
            effective_M_max = M2 + v1 / (1/(M1 - M2) + v2/(3*M2))

            # Compute concentration factors for magnetic load sharing
            M_eff = gamma * effective_M_max + (1 - gamma) * effective_M_min
            effective_properties[9] = M_eff
            CM1 = M1/M_eff * (1/v1 * (M2 - M_eff)/(M2 - M1))**2 
            CM2 = M2/M_eff * (1/v2 * (M1 - M_eff)/(M1 - M2))**2
            concentration_factors[20] = CM1
            concentration_factors[21] = CM2

            # Extract total magnetization normalized volume from genetic string 
            Mvol1 = Lam10_mat1 if Lam10_mat1 <= Lam10_mat2 else Lam10_mat2
            Mvol2 = Lam10_mat2 if Lam10_mat1 <= Lam10_mat2 else Lam10_mat1

            # Compute effective property mins and maxes with Hashin-Shtrikman
            effective_Mvol_min = Mvol1 + v2 / (1/(Mvol2 - Mvol1) + v1/(3*Mvol1))
            effective_Mvol_max = Mvol2 + v1 / (1/(Mvol1 - Mvol2) + v2/(3*Mvol2))

            # Compute concentration factors for magnetic load sharing (normalized by volume)
            Mvol_eff = gamma * effective_Mvol_max + (1 - gamma) * effective_Mvol_min
            effective_properties[10] = Mvol_eff
            CMvol1 = Mvol1/Mvol_eff * (1/v1 * (Mvol2 - Mvol_eff)/(Mvol2 - Mvol1))**2 
            CMvol2 = Mvol2/Mvol_eff * (1/v2 * (Mvol1 - Mvol_eff)/(Mvol1 - Mvol2))**2
            concentration_factors[22] = CMvol1
            concentration_factors[23] = CMvol2

        if "piezoelectric" in self.material_properties:

            # TODO check concentration factor formula
            
            # Extract total magnetization normalized volume from genetic string 
            epsij1 = Lam11_mat1 if Lam11_mat1 <= Lam11_mat2 else Lam11_mat2
            epsij2 = Lam11_mat2 if Lam11_mat1 <= Lam11_mat2 else Lam11_mat1

            # Compute effective property mins and maxes with Hashin-Shtrikman
            effective_epsij_min = epsij1 + v2 / (1/(epsij2 - epsij1) + v1/(3*epsij1))
            effective_epsij_max = epsij2 + v1 / (1/(epsij1 - epsij2) + v2/(3*epsij2))

            # Compute concentration factors for piezoelectric load sharing
            epsij_eff = gamma * effective_epsij_max + (1 - gamma) * effective_epsij_min
            effective_properties[11] = epsij_eff
            Ceij1 = epsij1/epsij_eff * (1/v1 * (epsij2 - epsij_eff)/(epsij2 - epsij1))**2 
            Ceij2 = epsij2/epsij_eff * (1/v2 * (epsij1 - epsij_eff)/(epsij1 - epsij2))**2
            concentration_factors[24] = Ceij1
            concentration_factors[25] = Ceij2

        for i, factor in enumerate(concentration_factors):
            if (factor - TOL) / TOL > 0:
                weights[i] = wj
            else:
                weights[i] = 0

        # Cast desired material properties to numpy array
        des_props = np.array(self.desired_properties)

        # Assemble the cost function
        domains = len(self.material_properties)
        W = 1/domains
        cost = w1*W * np.sum(abs(np.divide(des_props - effective_properties, effective_properties))) + np.sum(np.multiply(weights, abs(np.divide(concentration_factors - TOL, TOL))))
        #logger.info(f"checkpoint cost: {cost}")
        return cost
    