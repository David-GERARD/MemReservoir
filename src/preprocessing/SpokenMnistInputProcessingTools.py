import pickle
import os
from datetime import datetime
from .SpokenMnistInputProcessors import YananZhongEtAlPreprocessor

from src.utils.pickleFilesTools import saveProcessedItemToPickle

def processSample(sample, preprocessor = "YananZhongEtAlPreprocessor", preprocessor_params = None, voltage_output_params = None, save_directory_path = None):
    """
    
    This function processes a sample of audio waveforms using a preprocessor and returns the processed sample.

    Parameters:
    - sample (list of dictionaries): each dictionary contains the following
        - 'audio': torch.tensor, audio waveform
        - 'spokenMNISTindex': int, index of the spoken MNIST dataset
        - 'speakers': int, speaker index
        - 'labels': int, label index

    - preprocessor (str): name of the preprocessor to be used

    - preprocessor_params (dict): parameters for the preprocessor
        - YananZhongEtAlPreprocessor:
            - decimation_factor (int): downsampling factor
            - step_factor (float): involved in the number of final channels
            - N (int): number of nodes
            - tau (int): sampling period 

        - JohnMoonEtAlPreprocessor:
            - digitialize_threshold (float): threshold for digitization

    - voltage_output_params (dict): parameters for the voltage output
        - sample_rate (int): sample rate of the audio waveform to be processed
        - dt (float): sample rate of the voltage input to the reservoir to be generated from the audio waveform
        - n_cycles (int): number of times we want to repeat the input
        - normalize (bool): normalize the cochleogram to have its values range from 0 to 1
        - plot_coch (bool): plot the cochleogram (illadvised when processing multiple audio waveform)

    - save_directory_path (str, optional): path to the directory where the processed sample will be saved as pickle files

    Returns:

    - processed_sample (list of dictionaries): each dictionary contains the following
        - 'spokenMNISTindex': int, index of the spoken MNIST dataset
        - 'speaker': int, speaker index
        - 'label': int, label index
        - 't': numpy array, time array
        - 'channel_0': numpy array, input voltages for channel 0
        - 'channel_1': numpy array, input voltages for channel 1
        - ...
        - 'channel_n': numpy array, input voltages for channel n


    """


    if preprocessor_params == None:
        if preprocessor == "YananZhongEtAlPreprocessor":
            processing = YananZhongEtAlPreprocessor()

        elif preprocessor == "JohnMoonEtAlPreprocessor": # TODO: add support for JohnMoonEtAlPreprocessor
            print("JohnMoonEtAlPreprocessor not supported yet")
            return 0

        else:
            raise ValueError("Wrong preprocessor name, currenlty supported are: YananZhongEtAlPreprocessor, JohnMoonEtAlPreprocessor")

    else:
        if preprocessor == "YananZhongEtAlPreprocessor":

            if "decimation_factor" in preprocessor_params:
                decimation_factor = preprocessor_params["decimation_factor"]
            else:
                decimation_factor = 70 # default value

            if "step_factor" in preprocessor_params:
                step_factor = preprocessor_params["step_factor"]
            else:
                step_factor = 0.25 # default value

            if "N" in preprocessor_params:
                N = preprocessor_params["N"]
            else:
                N = 100 # default value

            if "tau" in preprocessor_params:
                tau = preprocessor_params["tau"]
            else:
                tau = 10 # default value

            if "voltage_amplitude" in preprocessor_params:
                voltage_amplitude = preprocessor_params["voltage_amplitude"]
            else:
                voltage_amplitude = 2 # default value

            processing = YananZhongEtAlPreprocessor(decimation_factor=decimation_factor, step_factor=step_factor, N = N, tau = tau,voltage_amplitude=voltage_amplitude)

        elif preprocessor == "JohnMoonEtAlPreprocessor": # TODO: add support for JohnMoonEtAlPreprocessor
            print("JohnMoonEtAlPreprocessor not supported yet")
            return 0

        else:
            raise ValueError("Wrong preprocessor name, currenlty supported are: YananZhongEtAlPreprocessor, JohnMoonEtAlPreprocessor")

    processed_sample = []

    for item in sample:
        audio = item['audio'].numpy() #8kH audio array

        if preprocessor == "YananZhongEtAlPreprocessor":
            if voltage_output_params == None:
                t, input_voltages = processing.process_audio(audio)
            else:

                # Sample rate of the audio waveform to be processed
                if "sample_rate" in voltage_output_params: 
                    sample_rate = voltage_output_params["sample_rate"]
                else:
                    sample_rate = 8000 # default value

                # Sample rate of the voltage input to the reservoir to be generated from the audio waveform
                if "dt" in voltage_output_params: 
                    dt = voltage_output_params["dt"]
                else:
                    dt = 500e-6 # default value

                # Number of times we want to repeat the input
                if "n_cycles" in voltage_output_params: 
                    n_cycles = voltage_output_params["n_cycles"]
                else:
                    n_cycles = 5 # default value
                
                # Normalize the cochleogram to have its values range from 0 to 1
                if "normalize" in voltage_output_params: 
                    normalize = voltage_output_params["normalize"]
                else:
                    normalize = True # default value

                # Plot the cochleogram (illadvised when processing multiple audio waveform)
                if "plot_coch" in voltage_output_params: 
                    plot_coch = voltage_output_params["plot_coch"]
                else:
                    plot_coch = False # default value

                t, input_voltages = processing.process_audio(audio,sample_rate=sample_rate, dt=dt, n_cycles=n_cycles, normalize=normalize, plot_coch=plot_coch)
                


        elif preprocessor == "JohnMoonEtAlPreprocessor": # TODO: add support for JohnMoonEtAlPreprocessor

            print("JohnMoonEtAlPreprocessor not supported yet")
            return 0

        else:
            raise ValueError("Wrong preprocessor name, currenlty supported are: YananZhongEtAlPreprocessor, JohnMoonEtAlPreprocessor")


        processed_item = {
            'spokenMNISTindex': item['spokenMNISTindex'],
            'speaker': item['speakers'],
            'label': item['labels'],
            't': t
            }
        channels = []

        for channel in range(len(input_voltages)):
            channels.append(input_voltages[channel])
            
        processed_item['channels'] = channels

        if save_directory_path is not None:
            saveProcessedItemToPickle(processed_item, save_directory_path)
                
        processed_sample.append(processed_item)

    if save_directory_path is not None:

        # Get the current date and time
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Create a string representation of the parameters
        parameters_str = f"Processing date: {current_datetime}"
        parameters_str += f"preprocessor_params: {preprocessor_params}\n"
        parameters_str += f"voltage_output_params: {voltage_output_params}\n"
        


        # Create the file path for the parameters text file
        parameters_file_path = os.path.join(save_directory_path, "processing_parameters.txt")

        # Write the parameters to the text file
        with open(parameters_file_path, 'w') as file:
            file.write(parameters_str)

    return processed_sample


