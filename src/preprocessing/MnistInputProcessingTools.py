from src.preprocessing.MnistInputProcessors import SpikeTrainPreprocessor
from src.utils.pickleFilesTools import saveProcessedItemToPickle
import numpy as np


def processSample(X,Y,voltage_output_params = None, save_directory_path = None):

    if len(X.shape) != 3:
        raise ValueError("Input data must be 3D, with shape (n_samples, height, width)")

    if X.shape[0] != len(Y):
        raise ValueError("Input data and target data must have the same number of samples")
    
    if voltage_output_params == None:
        process = SpikeTrainPreprocessor()
    else:
        if "t_pixel" in voltage_output_params:
            t_pixel = voltage_output_params["t_pixel"]
        else:
            t_pixel = 3e-3

        if "spike_amplitude" in voltage_output_params:
            spike_amplitude = voltage_output_params["spike_amplitude"]
        else:
            spike_amplitude = 3.0

        if "spike_duration" in voltage_output_params:
            spike_duration = voltage_output_params["spike_duration"]
        else:
            spike_duration = 1e-3

        if "digitalize" in voltage_output_params:
            digitalize = voltage_output_params["digitalize"]
        else:
            digitalize = True

        process = SpikeTrainPreprocessor(t_pixel=t_pixel, spike_amplitude=spike_amplitude, spike_duration=spike_duration, digitalize = digitalize)


    
    n_samples = X.shape[0]
    

    processed_sample = []

    for i in range(n_samples):
        im = X[i]

        if voltage_output_params == None or "dt" not in voltage_output_params:
            dt = 1e-4
            t, spikes = process.process_image(im,dt )
        else:
            dt = voltage_output_params["dt"]

            t, spikes = process.process_image(im, dt)

        processed_item = {
            'label': Y[i],
            'dt':dt,
            't': t,
            'channels':spikes
            }
        
        if save_directory_path is not None:
            saveProcessedItemToPickle(processed_item, save_directory_path)

        processed_sample.append(processed_item)

    return processed_sample


def processSample_V2(X,Y,voltage_output_params = None, save_directory_path = None, pixel_batch_size = 4):

    if len(X.shape) != 3:
        raise ValueError("Input data must be 3D, with shape (n_samples, height, width)")

    if X.shape[0] != len(Y):
        raise ValueError("Input data and target data must have the same number of samples")
    
    if voltage_output_params == None:
        process = SpikeTrainPreprocessor()
    else:
        if "t_pixel" in voltage_output_params:
            t_pixel = voltage_output_params["t_pixel"]
        else:
            t_pixel = 3e-3

        if "spike_amplitude" in voltage_output_params:
            spike_amplitude = voltage_output_params["spike_amplitude"]
        else:
            spike_amplitude = 3.0

        if "spike_duration" in voltage_output_params:
            spike_duration = voltage_output_params["spike_duration"]
        else:
            spike_duration = 1e-3

        if "digitalize" in voltage_output_params:
            digitalize = voltage_output_params["digitalize"]
        else:
            digitalize = True

        process = SpikeTrainPreprocessor(t_pixel=t_pixel, spike_amplitude=spike_amplitude, spike_duration=spike_duration, digitalize = digitalize)


    
    n_samples = X.shape[0]
    

    processed_sample = []

    for i in range(n_samples):
        im = X[i]

        if voltage_output_params == None or "dt" not in voltage_output_params:
            dt = 1e-4
            spikes_rate_1 = []
            for i in range(im.shape[1]//pixel_batch_size):
                t, spikes = process.process_image(im[:,i*pixel_batch_size:(i+1)*pixel_batch_size], dt)
                spikes_rate_1.append(np.array(spikes))
            spikes = np.array(spikes_rate_1)

        else:
            dt = voltage_output_params["dt"]

            spikes_rate_1 = []
            for i in range(im.shape[1]//pixel_batch_size):
                t, spikes = process.process_image(im[:,i*pixel_batch_size:(i+1)*pixel_batch_size], dt)
                spikes_rate_1.append(np.array(spikes))
            spikes = np.array(spikes_rate_1)

        processed_item = {
            'label': Y[i],
            'dt':dt,
            't': t,
            'channels':spikes
            }
        
        if save_directory_path is not None:
            saveProcessedItemToPickle(processed_item, save_directory_path)

        processed_sample.append(processed_item)

    return processed_sample


def add_read_pulse(V_input,dt,pulse_amplitude=1,pulse_duration=1e-3):
    """
    Add a read pusle at the end of the input signal

    Parameters:
    - V_input: input signal
    - dt: time step
    - pulse_amplitude: amplitude of the pulse
    - pulse_duration: duration of the pulse
    
    """
    t = np.arange(0,len(V_input)*dt,dt)


    pulse_end = t[-5]
    pulse_start = pulse_end - pulse_duration
    V_input[t>=pulse_start] = pulse_amplitude


    return V_input