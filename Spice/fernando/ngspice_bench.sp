*Netlist for Memdiode Circuit with Voltage Source and Load Resistor

.option METHOD=GEAR
.option RELTOL=1e-2
.option ABSTOL=1.00E-9
.option VNTOL=1.00E-3
.option TRTOL=7
.option ITL4=50
.option CHGTOL=1E-12
.option AUTOSTOP=0

.include /Users/davidgerard/Desktop/Project1/MemReservoir/Spice/fernando/memdiode_ngspice.lib

*Voltage source of 5V connected to the + terminal of the memdiode
*V1 p 0 SIN(0 2 2 0 0) 

*Load resistor of 1k connected from the - terminal of the memdiode to ground
Rload m 0 1 

*Memdiode subcircuit instance
