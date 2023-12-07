class GAParams:

    def __init__(
            self,
            P: int = 10,
            K: int = 10,
            G: int = 900, # Change it back to 5000 later
            S: int = 200,
            TOL: float = 0.5,
            w1: float = 1.0,
            wj: float = 0.5, 
    ):
       
        self.P = P     # number of design strings to breed
        self.K = K     # number of offspring design strings 
        self.G = G     # maximum number of generations
        self.S = S     # total number of design strings per generation
        self.TOL = TOL # property tolerance for convergence
        self.w1 = w1   # material property individual weight
        self.wj = wj   # concentration tensor weight


    #------ Getter Methods ------#
    def get_P(self):
        return self.P
    
    def get_K(self):
        return self.K
    
    def get_G(self):
        return self.G
    
    def get_S(self):
        return self.S
    
    def get_TOL(self):
        return self.TOL
    
    def get_w1(self):
        return self.w1
    
    def get_wj(self):
        return self.wj
    
    #------ Setter Methods ------#
    def set_P(self, new_P):
        self.P = new_P
        return self
    
    def set_K(self, new_K):
        self.K = new_K
        return self
    
    def set_G(self, new_G):
        self.G = new_G
        return self
    
    def set_S(self, new_S):
        self.S = new_S
        return self
    
    def set_TOL(self, new_TOL):
        self.TOL = new_TOL
        return self
    
    def set_w1(self, new_w1):
        self.w1 = new_w1
        return self
    
    def set_wj(self, new_wj):
        self.wj = new_wj
        return self

    

