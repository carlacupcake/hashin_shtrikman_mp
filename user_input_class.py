class UserInput:

    def __init__(
        self,
        include_carrier_transport:                     bool  = True,
        include_dielectric:                            bool  = True,
        include_elastic:                               bool  = True,
        include_magnetic:                              bool  = True,
        include_piezoelectric:                         bool  = True,
        desired_elec_cond_300k_low_doping:             float = 1e-5,
        desired_therm_cond_300k_low_doping:            float = 1e-5,
        desired_e_total:                               float = 1e-5,
        desired_e_ionic:                               float = 1e-5,
        desired_e_electronic:                          float = 1e-5,
        desired_n:                                     float = 1e-5,
        desired_bulk_modulus:                          float = 1e-5,
        desired_shear_modulus:                         float = 1e-5,
        desired_universal_anisotropy:                  float = 1e-5,
        desired_total_magnetization:                   float = 1e-5,
        desired_total_magnetization_normalized_volume: float = 1e-5,
        desired_e_ij:                                  float = 1e-5,
        lower_elec_cond_300k_low_doping:               float = 1e-5,
        lower_therm_cond_300k_low_doping:              float = 1e-5,
        lower_e_total:                                 float = 1e-5,
        lower_e_ionic:                                 float = 1e-5,
        lower_e_electronic:                            float = 1e-5,
        lower_n:                                       float = 1e-5,
        lower_bulk_modulus:                            float = 1e-5,
        lower_shear_modulus:                           float = 1e-5,
        lower_universal_anisotropy:                    float = 1e-5,
        lower_total_magnetization:                     float = 1e-5,
        lower_total_magnetization_normalized_volume:   float = 1e-5,
        lower_e_ij:                                    float = 1e-5,
        upper_elec_cond_300k_low_doping:               float = 1e-5,
        upper_therm_cond_300k_low_doping:              float = 1e-5,
        upper_e_total:                                 float = 1e-5,
        upper_e_ionic:                                 float = 1e-5,
        upper_e_electronic:                            float = 1e-5,
        upper_n:                                       float = 1e-5,
        upper_bulk_modulus:                            float = 1e-5,
        upper_shear_modulus:                           float = 1e-5,
        upper_universal_anisotropy:                    float = 1e-5,
        upper_total_magnetization:                     float = 1e-5,
        upper_total_magnetization_normalized_volume:   float = 1e-5,
        upper_e_ij:                                    float = 1e-5
        ):
        self.include_carrier_transport                     = include_carrier_transport
        self.include_dielectric                            = include_dielectric       
        self.include_elastic                               = include_elastic
        self.include_magnetic                              = include_magnetic
        self.include_piezoelectric                         = include_piezoelectric
        self.desired_elec_cond_300k_low_doping             = desired_elec_cond_300k_low_doping
        self.desired_therm_cond_300k_low_doping            = desired_therm_cond_300k_low_doping
        self.desired_e_total                               = desired_e_total
        self.desired_e_ionic                               = desired_e_ionic
        self.desired_e_electronic                          = desired_e_electronic
        self.desired_n                                     = desired_n
        self.desired_bulk_modulus                          = desired_bulk_modulus
        self.desired_shear_modulus                         = desired_shear_modulus
        self.desired_universal_anisotropy                  = desired_universal_anisotropy
        self.desired_total_magnetization                   = desired_total_magnetization
        self.desired_total_magnetization_normalized_volume = desired_total_magnetization_normalized_volume
        self.desired_e_ij                                  = desired_e_ij
        self.lower_elec_cond_300k_low_doping               = lower_elec_cond_300k_low_doping
        self.lower_therm_cond_300k_low_doping              = lower_therm_cond_300k_low_doping
        self.lower_e_total                                 = lower_e_total
        self.lower_e_ionic                                 = lower_e_ionic
        self.lower_e_electronic                            = lower_e_electronic
        self.lower_n                                       = lower_n
        self.lower_bulk_modulus                            = lower_bulk_modulus
        self.lower_shear_modulus                           = lower_shear_modulus
        self.lower_universal_anisotropy                    = lower_universal_anisotropy
        self.lower_total_magnetization                     = lower_total_magnetization
        self.lower_total_magnetization_normalized_volume   = lower_total_magnetization_normalized_volume
        self.lower_e_ij                                    = lower_e_ij
        self.upper_elec_cond_300k_low_doping               = upper_elec_cond_300k_low_doping
        self.upper_therm_cond_300k_low_doping              = upper_therm_cond_300k_low_doping
        self.upper_e_total                                 = upper_e_total
        self.upper_e_ionic                                 = upper_e_ionic
        self.upper_e_electronic                            = upper_e_electronic
        self.upper_n                                       = upper_n
        self.upper_bulk_modulus                            = upper_bulk_modulus
        self.upper_shear_modulus                           = upper_shear_modulus
        self.upper_universal_anisotropy                    = upper_universal_anisotropy
        self.upper_total_magnetization                     = upper_total_magnetization
        self.upper_total_magnetization_normalized_volume   = upper_total_magnetization_normalized_volume
        self.upper_e_ij                                    = upper_e_ij     

    #------ Getter Methods ------#
    def get_include_carrier_transport(self):
        return self.include_carrier_transport
    
    def get_include_dielectric(self):
        return self.include_dielectric
    
    def get_include_elastic(self):
        return self.include_elastic
    
    def get_include_magnetic(self):
        return self.include_magnetic
    
    def get_include_piezoelectric(self):
        return self.include_piezoelectric
    
    def get_desired_elec_cond(self):
        return self.desired_elec_cond_300k_low_doping
    
    def get_desired_therm_cond(self):
        return self.desired_therm_cond_300k_low_doping
    
    def get_desired_e_total(self):
        return self.desired_e_total
    
    def get_desired_e_ionic(self):
        return self.desired_e_ionic
    
    def get_desired_e_elec(self):
        return self.desired_e_electronic
    
    def get_desired_n(self):
        return self.desired_n
    
    def get_desired_bulk_modulus(self):
        return self.desired_bulk_modulus
    
    def get_desired_shear_modulus(self):
        return self.desired_shear_modulus
    
    def get_desired_universal_anisotropy(self):
        return self.desired_universal_anisotropy
    
    def get_desired_total_magnetization(self):
        return self.desired_total_magnetization
    
    def get_desired_total_magnetization_norm_vol(self):
        return self.desired_total_magnetization_normalized_volume
    
    def get_desired_e_ij(self):
        return self.desired_e_ij
    
    def get_lower_elec_cond(self):
        return self.lower_elec_cond_300k_low_doping
    
    def get_lower_therm_cond(self):
        return self.lower_therm_cond_300k_low_doping
    
    def get_lower_e_total(self):
        return self.lower_e_total
    
    def get_lower_e_ionic(self):
        return self.lower_e_ionic
    
    def get_lower_e_elec(self):
        return self.lower_e_electronic
    
    def get_lower_n(self):
        return self.lower_n
    
    def get_lower_bulk_modulus(self):
        return self.lower_bulk_modulus
    
    def get_lower_shear_modulus(self):
        return self.lower_shear_modulus
    
    def get_lower_universal_anisotropy(self):
        return self.lower_universal_anisotropy
    
    def get_lower_total_magnetization(self):
        return self.lower_total_magnetization
    
    def get_lower_total_magnetization_norm_vol(self):
        return self.lower_total_magnetization_normalized_volume
    
    def get_lower_e_ij(self):
        return self.lower_e_ij
    
    def get_upper_elec_cond(self):
        return self.upper_elec_cond_300k_low_doping
    
    def get_upper_therm_cond(self):
        return self.upper_therm_cond_300k_low_doping
    
    def get_upper_e_total(self):
        return self.upper_e_total
    
    def get_upper_e_ionic(self):
        return self.upper_e_ionic
    
    def get_upper_e_elec(self):
        return self.upper_e_electronic
    
    def get_upper_n(self):
        return self.upper_n
    
    def get_upper_bulk_modulus(self):
        return self.upper_bulk_modulus
    
    def get_upper_shear_modulus(self):
        return self.upper_shear_modulus
    
    def get_upper_universal_anisotropy(self):
        return self.upper_universal_anisotropy
    
    def get_upper_total_magnetization(self):
        return self.upper_total_magnetization
    
    def get_upper_total_magnetization_norm_vol(self):
        return self.upper_total_magnetization_normalized_volume
    
    def get_upper_e_ij(self):
        return self.upper_e_ij
    
    #------ Setter Methods ------#
    def set_include_carrier_transport(self, include_carrier_transport):
        self.include_carrier_transport = include_carrier_transport
        return self
    
    def set_include_dielectric(self, include_dielectric):
        self.include_dielectric = include_dielectric
        return self
    
    def set_include_elastic(self, include_elastic):
        self.include_elastic = include_elastic
        return self
    
    def set_include_magnetic(self, include_magnetic):
        self.include_magnetic = include_magnetic
        return self
    
    def set_include_piezoelectric(self, include_piezoelectric):
        self.include_piezoelectric = include_piezoelectric
        return self
    
    def set_desired_elec_cond(self, elec_cond):
        self.desired_elec_cond_300k_low_doping = elec_cond
        return self
    
    def set_desired_therm_cond(self, therm_cond):
        self.desired_therm_cond_300k_low_doping = therm_cond
        return self
    
    def set_desired_e_total(self, e_total):
        self.desired_e_total = e_total
        return self
    
    def set_desired_e_ionic(self, e_ionic):
        self.desired_e_ionic = e_ionic
        return self
    
    def set_desired_e_elec(self, e_elec):
        self.desired_e_elec = e_elec
        return self
    
    def set_desired_n(self, n):
        self.desired_n = n
        return self
    
    def set_desired_bulk_mod(self, bulk_mod):
        self.desired_bulk_modulus = bulk_mod
        return self
    
    def set_desired_shear_mod(self, shear_mod):
        self.desired_shear_modulus = shear_mod
        return self
    
    def set_desired_univ_aniso(self, univ_aniso):
        self.desired_universal_anisotropy = univ_aniso
        return self
    
    def set_desired_tot_mag(self, tot_mag):
        self.desired_total_magnetization = tot_mag
        return self
    
    def set_desired_tot_mag_norm_vol(self, tot_mag_norm_vol):
        self.desired_total_magnetization_normalized_volume = tot_mag_norm_vol
        return self
    
    def set_desired_e_ij(self, e_ij):
        self.desired_e_ij = e_ij
        return self
    
    def set_lower_elec_cond(self, elec_cond):
        self.lower_elec_cond_300k_low_doping = elec_cond
        return self
    
    def set_lower_therm_cond(self, therm_cond):
        self.lower_therm_cond_300k_low_doping = therm_cond
        return self
    
    def set_lower_e_total(self, e_total):
        self.lower_e_total = e_total
        return self
    
    def set_lower_e_ionic(self, e_ionic):
        self.lower_e_ionic = e_ionic
        return self
    
    def set_lower_e_elec(self, e_elec):
        self.lower_e_elec = e_elec
        return self
    
    def set_lower_n(self, n):
        self.lower_n = n
        return self
    
    def set_lower_bulk_mod(self, bulk_mod):
        self.lower_bulk_modulus = bulk_mod
        return self
    
    def set_lower_shear_mod(self, shear_mod):
        self.lower_shear_modulus = shear_mod
        return self
    
    def set_lower_univ_aniso(self, univ_aniso):
        self.lower_universal_anisotropy = univ_aniso
        return self
    
    def set_lower_tot_mag(self, tot_mag):
        self.lower_total_magnetization = tot_mag
        return self
    
    def set_lower_tot_mag_norm_vol(self, tot_mag_norm_vol):
        self.desired_total_magnetization_normalized_volume = tot_mag_norm_vol
        return self
    
    def set_lower_e_ij(self, e_ij):
        self.lower_e_ij = e_ij
        return self
    
    def set_upper_elec_cond(self, elec_cond):
        self.upper_elec_cond_300k_low_doping = elec_cond
        return self
    
    def set_upper_therm_cond(self, therm_cond):
        self.upper_therm_cond_300k_low_doping = therm_cond
        return self
    
    def set_upper_e_total(self, e_total):
        self.upper_e_total = e_total
        return self
    
    def set_upper_e_ionic(self, e_ionic):
        self.upper_e_ionic = e_ionic
        return self
    
    def set_upper_e_elec(self, e_elec):
        self.upper_e_elec = e_elec
        return self
    
    def set_upper_n(self, n):
        self.upper_n = n
        return self
    
    def set_upper_bulk_mod(self, bulk_mod):
        self.upper_bulk_modulus = bulk_mod
        return self
    
    def set_upper_shear_mod(self, shear_mod):
        self.upper_shear_modulus = shear_mod
        return self
    
    def set_upper_univ_aniso(self, univ_aniso):
        self.upper_universal_anisotropy = univ_aniso
        return self
    
    def set_upper_tot_mag(self, tot_mag):
        self.upper_total_magnetization = tot_mag
        return self
    
    def set_upper_tot_mag_norm_vol(self, tot_mag_norm_vol):
        self.upper_total_magnetization_normalized_volume = tot_mag_norm_vol
        return self
    
    def set_upper_e_ij(self, e_ij):
        self.upper_e_ij = e_ij
        return self

    

