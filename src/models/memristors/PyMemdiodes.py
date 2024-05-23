import numpy as np

class NonVolatileMemDiode:
    """
    This class implements the Non-Volatile Memristor model as described in the paper:
    https://doi.org/10.3390/mi13020330

    Author : David Gerard
    Based on the ngspice model of the Non-Volatile Memristor by Fernando Aguirre
    
    """

    def __init__(self, H0=0, Vset = 0.8):
        # Parameters as per the ngspice model
        self.beta = 0.5
        self.imax = 2.0e-5
        self.imin = 2.0e-6
        self.rsmax = 1.0
        self.rsmin = 1.0
        self.alphamin = 6.190
        self.alphamax = 5.220
        self.ch0 = 1.0
        self.etaset = 15.0
        self.vset = Vset
        self.etares = 4.0
        self.vres = -1.3
        self.vt = 0.1
        self.gam = 0.9
        self.gam0 = 0.0
        self.isb = 1.0
        self.EI = 1e-15
        self.RPP = 1e9
        self.H = H0  # Initial condition for H
        self.I = 0

    def I0(self, x):
        return self.imax * x + self.imin * (1 - x)

    def RS(self, x):
        return self.rsmax * x + self.rsmin * (1 - x)

    def Stau(self, Vpn):
        argument = min(max(-self.etaset * (Vpn - self.isb), -67), 67)
        return np.exp(argument)

    def Rtau(self, Vpn):
        argument = min(max(self.etares * self.ISF(self.H) * (Vpn - self.vres), -67), 67)
        return np.exp(argument)
    
    def update_H(self, Vpn,dt = 1e-6):
        if Vpn >0:
            new_h = self.H + dt*(1-self.H)/self.Stau(Vpn)
        else:
            new_h = self.H - dt*self.H/self.Rtau(Vpn) 

        self.H = min(max(new_h, 0), 1)

    def VSB(self, Vpn):
        return self.vt if Vpn > self.isb else self.vset

    def ISF(self, x):
        return np.sign(x) * (np.abs(x)) ** (self.gam) if self.gam != 0 else 1

    def current_through_GD(self, Vdn,dt = 1e-6):
        x = self.H
        current_output = self.I0(x) * np.sinh(self.beta*(Vdn - self.VSB(self.ISF(x)) - self.RS(x) * self.I) ) 
        self.update_H(Vdn - self.VSB(self.ISF(x)),dt)
        self.I = current_output 
        return current_output
    
class VolatileMemDiode:
    """
    This class implements the Non-Volatile Memristor model as described in the paper:
    https://doi.org/10.3390/mi13020330

    Author : David Gerard
    Based on the ngspice model of the Non-Volatile Memristor by Fernando Aguirre
    
    """

    def __init__(self, H0=0, Vset = 0.8, tau_volatility = 50e-6):
        # Parameters as per the ngspice model
        self.beta = 0.5
        self.imax = 2.0e-5
        self.imin = 2.0e-6
        self.rsmax = 1.0
        self.rsmin = 1.0
        self.alphamin = 6.190
        self.alphamax = 5.220
        self.ch0 = 1.0
        self.etaset = 15.0
        self.vset = Vset
        self.etares = 4.0
        self.vres = -1.3
        self.vt = 0.1
        self.gam = 0.9
        self.gam0 = 0.0
        self.isb = 1.0
        self.EI = 1e-15
        self.RPP = 1e9
        self.H = H0  # Initial condition for H
        self.I = 0

        self.tau_volatility = tau_volatility

    def I0(self, x):
        return self.imax * x + self.imin * (1 - x)


    def RS(self, x):
        return self.rsmax * x + self.rsmin * (1 - x)

    def Stau(self, Vpn):
        argument = min(max(-self.etaset * (Vpn - self.isb), -67), 67)
        return np.exp(argument)

    def Rtau(self, Vpn):
        argument = min(max(self.etares * self.ISF(self.H) * (Vpn - self.vres), -67), 67)
        return np.exp(argument)
    
    def update_H(self, Vpn,dt = 1e-6):
        if Vpn >0:
            new_h = self.H + dt*(1-self.H)/self.Stau(Vpn) - dt*self.H/self.tau_volatility
        else:
            new_h = self.H - dt*self.H/self.Rtau(Vpn) - dt*self.H/self.tau_volatility
        self.H = min(max(new_h,0),1)

    def VSB(self, Vpn):
        return self.vt if Vpn > self.isb else self.vset

    def ISF(self, x):
        return np.sign(x) * (np.abs(x)) ** (self.gam) if self.gam != 0 else 1

    def current_through_GD(self, Vdn,dt = 1e-6):
        x = self.H
        current_output = self.I0(x) * np.sinh(self.beta*(Vdn - self.VSB(self.ISF(x)) - self.RS(x) * self.I) ) 
        self.update_H(Vdn - self.VSB(self.ISF(x)),dt)
        self.I = current_output 
        return current_output
    
