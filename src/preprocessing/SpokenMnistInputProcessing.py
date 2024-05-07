# Code by David Gerard https://github.com/David-GERARD

from src.utils.lyon.calc import LyonCalc
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d



class JohnMoonEtAlPreprocessor:
    """
    Processes audio data from the spoken MNIST dataset to be used as input for a memristor-based reservoir computing system.
    data: https://github.com/Jakobovski/free-spoken-digit-dataset

    This preprocessing method is introduced in "Temporal data classification and forecasting using a memristor-based reservoir computing system" by John Moon, et al.
    https://doi.org/10.1038/s41928-019-0313-3

    Args:
        decimation_factor (int): The decimation factor for the Lyon auditory model. Default is 100.
        step_factor (float): The step factor for the Lyon auditory model. Default is 0.4.
        digitialize_threshold (float): The threshold for digitizing the cochleagram. Default is 0.08.
        spike_amplitude (float): The amplitude of the generated spike. Default is 3.0.
        spike_width (float): The width of the generated spike. Default is 10e-6.
        step_length (float): The step length for generating spike trains. Default is 250e-6.

    Methods:
        lyon_passive_ear(waveform, sample_rate): Compute the cochleagram of a waveform using the Lyon passive ear model.
        digitialize(coch): Digitize the cochleagram.
        generate_spike_trains(digitized_coch, dt): Generate a spike train from the digitized cochleagram.
        process_audio(audio, dt): Process the audio data for the spoken MNIST dataset.
    """
    def __init__(self, decimation_factor=100, step_factor=0.4, digitialize_threshold=0.08, spike_amplitude=3.0, spike_width=10e-6, step_length=250e-6):
        """
        Initialize the JohnMoonEtAlPreprocessor.
        
        Parameters:
        - decimation_factor (int): The decimation factor for the Lyon auditory model.
        - step_factor (float): The step factor for the Lyon auditory model.
        - digitialize_threshold (float): The threshold for digitizing the cochleagram.
        - spike_amplitude (float): The amplitude of the generated spike.
        - spike_width (float): The width of the generated spike.
        - step_length (float): The step length for each of the steps outputted by the cochleagram.
        
        """
        # Lyon auditory model parameters
        self.decimation_factor = decimation_factor
        self.step_factor = step_factor

        # Digitization parameters
        self.digitialize_threshold = digitialize_threshold

        # Spike generation parameters
        self.spike_amplitude = spike_amplitude # V
        self.spike_width = spike_width # s
        self.step_length = step_length # s
    
    def lyon_passive_ear(self, waveform, sample_rate):
        """
        Compute the cochleagram of a waveform using the Lyon passive ear model. 

        Parameters:
        - waveform (numpy.ndarray): The audio waveform

        Returns:
        - coch (numpy.ndarray): The cochleagram
        """
        calc = LyonCalc()
        coch = calc.lyon_passive_ear(waveform, sample_rate = sample_rate, decimation_factor = self.decimation_factor, step_factor = self.step_factor).T
        return coch
    
    def digitialize(self, coch):
        """
        Digitize the cochleagram.

        Parameters:
        - coch (numpy.ndarray): The cochleagram

        Returns:
        - digitized_coch (numpy.ndarray): The digitized cochleagram
        """
        return np.where(coch > self.digitialize_threshold, 1, 0)
    
    def generate_spike_trains(self, digitized_coch, dt):
        """
        Generate a spike train from the digitized cochleagram.

        Parameters:
        - digitized_coch (numpy.ndarray): The digitized cochleagram
        - dt (float): The time step for generating spike trains

        Returns:
        - t (numpy.ndarray): The time array
        - spikes (numpy.ndarray): The generated spike trains
        """
        t = np.linspace(0, digitized_coch.shape[1]*self.step_length, int(digitized_coch.shape[1]*self.step_length/dt))
        spikes = np.zeros((digitized_coch.shape[0], len(t)))

        for i in range(digitized_coch.shape[0]): # for each channel
            for j in range(digitized_coch.shape[1]):
                if digitized_coch[i, j] == 1:
                    spikes[i, int(j*self.step_length/dt):int(j*self.step_length/dt)+int(self.spike_width/dt)] = self.spike_amplitude # generate a spike if the digitalized cochleagram value is 1
            
        return t,spikes
    
    def plot_cochleagram(self, coch):
        """
        Plot the cochleagram.

        Parameters:
        - coch (numpy.ndarray): The cochleagram
        """
        plt.figure(figsize=(10, 5))
        plt.imshow(coch, aspect='auto', origin='lower', interpolation='nearest')
        plt.colorbar()
        plt.show()

    def plot_digitized_cochleagram(self, digitized_coch):
        """
        Plot the digitized cochleagram.

        Parameters:
        - digitized_coch (numpy.ndarray): The digitized cochleagram
        """
        plt.figure(figsize=(10, 5))
        plt.imshow(digitized_coch, aspect='auto', origin='lower', interpolation='nearest')
        plt.colorbar()
        plt.show()



    def process_audio(self, audio, sample_rate=8000, dt=5e-6, plot_coch=False, plot_digitized_coch=False): 
        """
        Parameters:
        - audio (numpy.ndarray): The audio data to process. 
        - sample_rate (int): The sample rate of the audio data, defaults to 8000 Hz
        - dt (float): The time step for generating spike trains, defaults to 5e-6

        Returns:
        - t (numpy.ndarray): The time array
        - spikes (numpy.ndarray): The generated spike trains
        """

        coch = self.lyon_passive_ear(audio,sample_rate)
        digitized_coch = self.digitialize(coch)
        t, spikes = self.generate_spike_trains(digitized_coch, dt)
        if plot_coch:
            self.plot_cochleagram(coch)
        if plot_digitized_coch:
            self.plot_digitized_cochleagram(digitized_coch)
        return t, spikes



class YananZhongEtAlPreprocessor:
    """
    Processes audio data from the spoken MNIST dataset to be used as input for a memristor-based reservoir computing system.
    data: https://github.com/Jakobovski/free-spoken-digit-dataset

    This preprocessing method is introduced in "Dynamic memristor-based reservoir computing for high-efficiency temporal signal processing" by Yanan Zhong, et al.
    https://doi.org/10.1038/s41467-020-20692-1
    
    Args:
        decimation_factor (int): The decimation factor for the Lyon auditory model. Default is 100.
        step_factor (float): The step factor for the Lyon auditory model. Default is 0.4.
        digitialize_threshold (float): The threshold for digitizing the cochleagram. Default is 0.08.
        spike_amplitude (float): The amplitude of the generated spike. Default is 3.0.
        spike_width (float): The width of the generated spike. Default is 10e-6.
        step_length (float): The step length for generating spike trains. Default is 250e-6.

    Methods:
        lyon_passive_ear(waveform, sample_rate): Compute the cochleagram of a waveform using the Lyon passive ear model.
        digitialize(coch): Digitize the cochleagram.
        generate_spike_trains(digitized_coch, dt): Generate a spike train from the digitized cochleagram.
        process_audio(audio, dt): Process the audio data for the spoken MNIST dataset.
    """
    def __init__(self, decimation_factor=70, step_factor=0.25, N=100, tau=1.2e-3, voltage_amplitude=2):
        """
        Initialize the YananZhongEtAlPreprocessor.

        Parameters:
        - decimation_factor (int): The decimation factor for the Lyon auditory model.
        - step_factor (float): The step factor for the Lyon auditory model.
        - N (int): The number of reservoir nodes.
        - tau (float): The time constant for a reservoir cycle (in s).
        - voltage_amplitude (int): The voltage amplitude for the input.

        """
        # Lyon auditory model parameters
        self.decimation_factor = decimation_factor
        self.step_factor = step_factor

        # Time multiplexing parameters
        self.N = N
        self.tau = tau
        self.theta = tau / N  # time delay between 2 nodes

        # Voltage input parameters
        self.voltage_amplitude = voltage_amplitude

        
    
    def lyon_passive_ear(self, waveform, sample_rate):
        """
        Compute the cochleagram of a waveform using the Lyon passive ear model. 

        Parameters:
        - waveform (numpy.ndarray): The audio waveform
        - sample_rate (int): The sample rate of the waveform

        Returns:
        - coch (numpy.ndarray): The cochleagram, a 2D array representing the spectral content of the waveform
        """
        calc = LyonCalc()
        coch = calc.lyon_passive_ear(waveform, sample_rate=sample_rate, decimation_factor=self.decimation_factor, step_factor=self.step_factor).T
        return coch
    
    
    def plot_cochleagram(self, coch):
        """
        Plot the cochleagram.

        Parameters:
        - coch (numpy.ndarray): The cochleagram
        """
        plt.figure(figsize=(10, 5))
        plt.imshow(coch, aspect='auto', origin='lower', interpolation='nearest')
        plt.colorbar()
        plt.show()

    def sampleAndHold(self, coch, N):
        """
        Sample and hold the cochleagram to a desired length.

        Parameters:
        - coch (numpy.ndarray): The cochleagram
        - N (int): The desired length of the cochleagram
        
        Returns:
        - sampled_coch (numpy.ndarray): The sampled cochleagram

        """
        coch_length = coch.shape[1]

        if N == coch_length:
            return coch

        indices = np.linspace(0, coch_length-1, N).astype(int)
        sampled_coch = coch[:, indices]

        return sampled_coch
    
    def sampleAndInterpolate(self, coch, N):
        """
        Sample and interpolate the cochleagram to a desired length.

        Parameters:
        - coch (numpy.ndarray): The cochleagram
        - N (int): The desired length of the cochleagram

        Returns:
        - upsampled_coch (numpy.ndarray): The upsampled cochleagram

        """
        coch_length = coch.shape[1]

        if N == coch_length:
            return coch

        x = np.arange(coch_length)
        x_new = np.linspace(0, coch_length-1, N)
        upsampled_coch = np.zeros((coch.shape[0], N))

        for i in range(coch.shape[0]):
            f = interp1d(x, coch[i], kind='linear')
            upsampled_coch[i] = f(x_new)

        return upsampled_coch


    def generateMask(self, n_channels, N):
        """
        Generates a random mask array of size N, where each element is either 1 or -1, chosen randomly.
        
        Parameters:
        - n_channels (int): The number of channels of the cochleogram.
        - N (int): The size of the mask array to generate, aka the number of vitual reservoir nodes.
        
        Returns:
        - numpy.ndarray: An array of shape (n_channels, N), where each element is randomly set to either 1 or -1.
        
        This function utilizes numpy's random.choice method to fill the array with -1 or 1 values.
        """
        M = np.random.choice([-1, 1], size=(n_channels, N))
        return M
    
    def applyMask(self, I, M):
            """
            Applies a mask `M` to an input signal `I`, element-wise, with periodic extension of the mask if needed.
            
            Parameters:
            - I (numpy.ndarray): The input signal to mask.
            - M (numpy.ndarray): The mask to apply, where each element is either 1 or -1.
            
            Returns:
            - J (numpy.ndarray): The masked signal, which is the element-wise multiplication of `I` by `M`, extending `M` periodically if `I` is longer than `M`.
            
            This method applies the mask `M` to the input signal `I` by performing an element-wise multiplication. If the shapes of `I` and `M` are not equal, a `ValueError` is raised. The mask `M` is extended periodically if `I` is longer than `M`. The resulting masked signal is returned as a torch tensor.
            """
            if I.shape != M.shape:
                raise ValueError("I and M must have the same shape.")

            J = np.multiply(I, M)
            
            return J


    def process_audio(self, audio, sample_rate=8000, dt=500e-6, n_cycles=5, plot_coch=False):
        """
        Process the audio data and generate spike trains.

        Parameters:
        - audio (numpy.ndarray): The audio data to process.
        - sample_rate (int): The sample rate of the audio data, defaults to 8000 Hz.
        - dt (float): The time step for generating voltage input, defaults to 5e-6.
        - n_cycles (int): The number of cycles to repeat the input voltages, defaults to 5.
        - plot_coch (bool): Whether to plot the cochleagram, defaults to False.

        Returns:
        - t (numpy.ndarray): The time array.
        - input_voltages (numpy.ndarray): The generated input voltages as spike trains.
        """

        # Compute the cochleagram
        self.coch = self.lyon_passive_ear(audio, sample_rate)
        amplitude_coef = self.voltage_amplitude / np.max(self.coch)
        if plot_coch:
            self.plot_cochleagram(self.coch)

        # Sample and hold
        sampled_coch = self.sampleAndHold(self.coch, self.N)

        # Masking
        M = self.generateMask(sampled_coch.shape[0], self.N)
        print(M.shape, sampled_coch.shape)
        masked_coch = self.applyMask(sampled_coch, M)

        # Time upsampling
        n_points = int(self.theta / dt) * masked_coch.shape[1]
        print(n_points)
        upsampled_coch = self.sampleAndInterpolate(masked_coch, n_points)  # TODO: question about upsampling method

        # Generate input voltages by repeating the upsampled cochleagram over n_cycles
        input_voltages = np.zeros((upsampled_coch.shape[0], n_points * n_cycles))
        for i in range(n_cycles):
            input_voltages[:, i * n_points:(i + 1) * n_points] = upsampled_coch * amplitude_coef

        # Generate the time array
        t = np.linspace(0, n_cycles * self.tau, n_points * n_cycles)

        return t, input_voltages