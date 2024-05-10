# Code by: David Gerard


import pandas as pd
from scipy.interpolate import interp1d

import numpy as np
import matplotlib.pyplot as plt

def relativeSiPhotodiodeResponsivity(wavelengths, reference_wavelength=632.8, path = "src/models/nonidealities/SiResponsivity.csv"):
    """
    Function that returns the responsivity of a silicon photodiode as a function of the wavelength.
    Data extracted from https://doi.org/10.1063/1.5009069 figure 3.f.

    Parameters:
    - wavelengths (np.array): Array of wavelengths in nm.
    - reference_wavelength (float, optional): Reference wavelength in nm. If provided, the responsivity will be normalized with respect to the responsivity at the reference wavelength.

    Returns:
    - responsivity (np.array): Array of responsivity in A/W.

    Raises:
    - ValueError: If the maximum wavelength in `wavelengths` is greater than 1062 nm or the minimum wavelength is less than 333 nm.

    """
    if np.max(wavelengths) > 1062 or np.min(wavelengths) < 333:
        raise ValueError("Wavelength out of range, must be between 333 nm and 1062 nm")

    # Load data extracted from paper
    data = pd.read_csv(path, header=None, names=['Wavelength[nm]', 'Responsivity[A/W]'])

    # Interpolate data
    function = interp1d(data['Wavelength[nm]'], data['Responsivity[A/W]'], kind='cubic')

    if reference_wavelength is None:
        return function(wavelengths)
    else:
        if reference_wavelength > 1062 or reference_wavelength < 333:
            raise ValueError("Reference wavelength out of range, must be between 333 nm and 1062 nm")
        return function(wavelengths) / function(reference_wavelength)
    

def getVset(powernW,wavelengths = None,  Vset0 = None,path_power = "src/models/nonidealities/Vset_Power_632nm.csv", path_SiResponsivity = "src/models/nonidealities/SiResponsivity.csv"):

    """
    Compute the Vset value for a given lign intensity power in nW and responsivity of the silicon substrate.
    Data extracted from https://doi.org/10.1063/1.5009069 figure 3.e.

    Parameters:
    - powernW (float): Power in nW.
    - responivity (float, optional): Relative responsivity of the silicon photodiode compared to the responsivity at the reference wavelength (632.8nm).
    - Vset0 (float, optional): Vset value of the memristor model used, that becomes the value in absance of illumination.
    - path (str, optional): Path to the file containing the Vset vs Power data for the reference wavelength (632.8nm).
    
    """

    data = pd.read_csv(path_power, header=None, names=['Power[nW]', 'Vset[V]'])
    coefs = np.polyfit(data['Power[nW]'], data['Vset[V]'], 1)

    if wavelengths is None:
        responivity = 1
        if Vset0 is None:
            return responivity*powernW*coefs[0] + coefs[1]
        else:
            scale_factor = Vset0/coefs[1]
            scaled_coefs = coefs*scale_factor
            return responivity*powernW*scaled_coefs[0] + scaled_coefs[1]
        
    else:
        responivity = relativeSiPhotodiodeResponsivity(wavelengths=wavelengths, path= path_SiResponsivity)
        if Vset0 is None:
            return np.outer(responivity,powernW)*coefs[0] + coefs[1]
        else:
            scale_factor = Vset0/coefs[1]
            scaled_coefs = coefs*scale_factor
            return np.outer(responivity,powernW)*scaled_coefs[0] + scaled_coefs[1]


if __name__ == "__main__":
    # Example of how to use the functions
    from src.utils.matplotlibTools import generate_colormap

    # Vset value of the memristor model used, that becomes the value in absance of illumination.
    Vset0 = 0.8
    
    # Generate wavelength values
    N = 10
    sample_wavelengths = np.linspace(500, 1050, N)
    colors = generate_colormap(sample_wavelengths)

    # Generate power values (in nW)
    power = np.linspace(-5, 45, 100)

    # Compute Vset values
    Vset = getVset(power, wavelengths=  sample_wavelengths, Vset0=Vset0)

    # Plot Vset vs Power for different wavelengthss
    for i in range(N):
        plt.plot(power, Vset[i], color=colors[i], label = str(int(sample_wavelengths[i]))+"nm")

    plt.legend(fontsize = "small")
    plt.xlabel('Power [nW]')
    plt.ylabel('Vset [V]')
    plt.title('Vset vs Power for different wavelengths')

    plt.show()