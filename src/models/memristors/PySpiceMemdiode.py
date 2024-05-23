from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *
import numpy as np

class NGSpiceMemDiode:
    def __init__(self,memdiode_lib_path,Vset = 0.8, Rleak = 100, Rload = 1):

        self.Rload = Rload
        
        self.circuit = Circuit('memristor testbench')

        circuit_netlist=[]


        self.circuit.include(memdiode_lib_path)

        self.addMemDiode( name = "memristor1", p = 'p', m='m', H='H', H0=0, Vset=Vset, Rleak=Rleak)

        self.circuit.R("load", 'm', '0', Rload@u_Ω)

    def addMemDiode(self, name, p, m, H, H0=0, Vset=0.8, Rleak=10):
        self.circuit.X(name, '', p, m, H, 'memdiode', 'H0='+str(H0), 'Vset='+str(Vset), 'Rleak='+str(Rleak))

    def build_PWL(self,time, V):
        PWL_string = "PWL("
        for idx in range(len(time)):
            PWL_string += str(time[idx]) + " " + str(V[idx]) + " "
        PWL_string += ")"
        return PWL_string

    def get_values_from_pyspice_waveform(self,waveform):
        size_waveform=len(waveform)
        waveform_values=np.zeros((size_waveform,1))
        for idx in range(size_waveform):
            waveform_values[idx]=waveform[idx]
        return waveform_values.astype(float)

    def get_current_response(self,Vinput,t):

        dt = t[1]-t[0]
        max_time = t[-1]
        self.circuit.V(1, 'p', '0', self.build_PWL(t,Vinput))
        simulation = self.circuit.simulator(temperature=25, nominal_temperature=25)
        analysis = simulation.transient(step_time=dt@u_s, end_time=max_time@u_s, use_initial_condition=True)

        Vinput_spice = self.get_values_from_pyspice_waveform(analysis.p)[:,0]
        Ioutput_spice = (self.get_values_from_pyspice_waveform(analysis.m)[:,0])/self.Rload
        Houtput_spice = self.get_values_from_pyspice_waveform(analysis.H)[:,0]
        t = self.get_values_from_pyspice_waveform(analysis.time)[:,0]

        return t,Vinput_spice, Ioutput_spice,Houtput_spice
