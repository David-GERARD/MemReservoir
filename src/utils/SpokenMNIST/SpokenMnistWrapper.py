# Code by David Gerard https://github.com/David-GERARD 

import hub
import numpy as np
from .trimmer import trim_silence
import os
import pandas as pd
from scipy.io import wavfile
import torch
import random

class SpokenMnistWrapperAPI:
    """
    A wrapper class for accessin the Spoken MNIST dataset via the hub API.

    Attributes:
    - dataset: The Spoken MNIST dataset loaded from the ActiveLoop hub.

    Methods:
    - get_audio_from_index(index): Returns the audio data for a given index.
    - get_spectrogram_from_index(index): Returns the spectrogram data for a given index.
    - get_label_from_index(index): Returns the label for a given index.
    - get_speaker_from_index(index): Returns the speaker name for a given index.
    - get_item(index): Returns a dictionary containing audio, spectrogram, label, and speaker data for a given index.
    - get_speakers(): Returns an array of unique speaker names in the dataset.
    - get_labels(): Returns an array of unique labels in the dataset.
    - get_sample(sample_size, digits=None, speakers=None, min_length=None, max_length=None): Returns a sample of items from the dataset that match the specified criteria.

    Example usage:

    # load the full dataset from hub
    data = SpokenMnistWrapperAPI(path, trimmed = True)

    # load the unique speakers contained in the dataset
    speakers = data.get_speakers()

    # load a sample of the spoken MNIST dataset based on criterions
    sample = data.get_sample(3, digits = [0,1], speakers = ['george'], min_length = 1000)
    """

    def __init__(self, trimmed=False):
        ds = hub.load("hub://activeloop/spoken_mnist")

        
        self.sampling_rate = 8000 # All recordings should be mono 8kHz

        if trimmed:
            rows_list = []
            for i, sample in enumerate(ds):
                audio =  ds['audio'][i]
                trimmed_audio = trim_silence(audio.numpy())
                rows_list.append({'labels': ds['labels'][i].numpy()[0], 'speakers': ds['speakers'][i].numpy()[0], 'audio': torch.tensor(trimmed_audio, dtype=torch.float64),'spectrograms': ds['spectrograms'][i]})
            self.dataset = pd.DataFrame(rows_list, columns=['labels', 'speakers', 'audio','spectrograms'])
        else:
            rows_list = []
            for i, sample in enumerate(ds):
                rows_list.append({'labels': ds['labels'][i].numpy()[0], 'speakers': ds['speakers'][i].numpy()[0], 'audio': ds['audio'][i],'spectrograms': ds['spectrograms'][i]})
            self.dataset = pd.DataFrame(rows_list, columns=['labels', 'speakers', 'audio','spectrograms'])

    def get_audio_from_index(self, index):
        """
        Returns the audio data for a given index.

        Parameters:
        - index (int): The index of the item in the dataset.

        Returns:
        - audio (ndarray): The audio data.
        """
        return self.dataset['audio'][int(index)]

    def get_spectrogram_from_index(self, index):
        """
        Returns the spectrogram data for a given index.

        Parameters:
        - index (int): The index of the item in the dataset.

        Returns:
        - spectrogram (ndarray): The spectrogram data.
        """
        return self.dataset['spectrograms'][int(index)]

    def get_label_from_index(self, index):
        """
        Returns the label for a given index.

        Parameters:
        - index (int): The index of the item in the dataset.

        Returns:
        - label (int): The label.
        """
        return self.dataset['labels'][int(index)]

    def get_speaker_from_index(self, index):
        """
        Returns the speaker name for a given index.

        Parameters:
        - index (int): The index of the item in the dataset.

        Returns:
        - speaker (str): The speaker name.
        """
        return self.dataset['speakers'][int(index)]

    def get_item_from_index(self, index):
        """
        Returns a dictionary containing audio, spectrogram, label, and speaker data for a given index.

        Parameters:
        - index (int): The index of the item in the dataset.

        Returns:
        - item (dict): A dictionary containing audio, spectrogram, label, and speaker data.
        """
        return {
            'audio': self.get_audio_from_index(index),
            'spectrograms': self.get_spectrogram_from_index(index),
            'labels': self.get_label_from_index(index),
            'speakers': self.get_speaker_from_index(index),
        }
    
    def get_speakers(self):
        """
        Returns an array of unique speaker names in the dataset.

        Returns:
        - speakers (ndarray): An array of unique speaker names.
        """
        return np.unique(self.dataset['speakers'].to_numpy())
    
    def get_labels(self):
        """
        Returns an array of unique labels in the dataset.

        Returns:
        - labels (ndarray): An array of unique labels.
        """
        return np.unique(self.dataset['labels'].to_numpy())

    def get_sample(self, sample_size = None, digits=None, speakers=None, min_length=None, max_length=None):
        """
        Returns a sample of items from the dataset that match the specified criteria.

        Parameters:
        - sample_size (int): The number of items to include in the sample.
        - digits (list, optional): A list of digits that should be included in the sample. If not specified, any digit can be included.
        - speakers (list, optional): A list of speaker names that should be included in the sample. If not specified, any speaker can be included.
        - min_length (int, optional): The minimum length of audio that should be included in the sample. If not specified, there is no minimum length requirement.
        - max_length (int, optional): The maximum length of audio that should be included in the sample. If not specified, there is no maximum length requirement.

        Returns:
        - sample (list): A list of items from the dataset that match the specified criteria.

        Raises:
        - ValueError: If there are not enough items in the dataset that match the specified criteria.
        """
        sample = []
        indices = []


        idx_shuffled = np.arange(len(self.dataset))
        np.random.shuffle(idx_shuffled)

        j = 0

        if sample_size is None:
            sample_size = len(self.dataset)
        else:
            if sample_size == len(self.dataset):
                raise ValueError("The sample size is equal to the size of the dataset. Please specify a smaller sample size, or leave the field empty if you want all the samples fitting the other criterions.")

        while len(indices) < sample_size and j < len(self.dataset):
            i = idx_shuffled[j]
            valid = True

            if digits is not None:
                label = self.get_label_from_index(i)
                if label not in digits:
                    valid = False

            if speakers is not None:
                speaker = self.get_speaker_from_index(i)
                if speaker not in speakers:
                    valid = False
            
            if min_length is not None:
                audio = self.get_audio_from_index(i).numpy()
                if len(audio) < min_length:
                    valid = False
            
            if max_length is not None:
                audio = self.get_audio_from_index(i).numpy()
                if len(audio) > max_length:
                    valid = False

            if valid:
                indices.append(i) 
            
            j+=1

        if len(indices) < sample_size and sample_size < len(self.dataset): # the second condition is false if the user didn't specify a sample size
            raise ValueError("Not enough items in the dataset that match the specified criteria.")

        sample_indices = np.array(indices)

        for index in sample_indices:
            sample.append(self.get_item_from_index(index))

        return sample
    

class SpokenMnistWrapperLocal:
    """
    A wrapper class for accessing the Spoken MNIST dataset via a local file.

    Attributes:
    - dataset: The Spoken MNIST dataset loaded from a local file.

    Methods:
    - get_audio_from_index(index): Returns the audio data for a given index.
    - get_spectrogram_from_index(index): Returns the spectrogram data for a given index.
    - get_label_from_index(index): Returns the label for a given index.
    - get_speaker_from_index(index): Returns the speaker name for a given index.
    - get_item(index): Returns a dictionary containing audio, spectrogram, label, and speaker data for a given index.
    - get_speakers(): Returns an array of unique speaker names in the dataset.
    - get_labels(): Returns an array of unique labels in the dataset.
    - get_sample(sample_size, digits=None, speakers=None, min_length=None, max_length=None): Returns a sample of items from the dataset that match the specified criteria.

    Example usage:

    # load all the wav files having the right naming convetion in a target folder
    path = "path/to/wav/files/audio_data/folder"
    data = SpokenMnistWrapperLocal(path, trimmed = True)

    # load the unique speakers contained in the dataset
    speakers = data.get_speakers()

    # load a sample of the spoken MNIST dataset based on criterions
    sample = data.get_sample(3, digits = [0,1], speakers = ['george'])

    """

    def __init__(self, path, trimmed=False):
        """
        Initializes the SpokenMnistWrapperLocal object.

        Parameters:
        - path (str): The path to the directory containing the Spoken MNIST dataset files.
        - trimmed (bool, optional): Whether to trim silence from the audio data. Defaults to False.

        Raises:
        - ValueError: If the specified path does not exist or does not contain any wav files.

        """
        if os.path.exists(path):
            rows_list = []
            wav_file_count = 0
            for file in os.listdir(path):
                if file.endswith(".wav"):
                    file_parts = file.split('_')
                    if len(file_parts) != 3:
                        raise ValueError("The wav file name is not in the correct format <digit>_<speaker>_<attempt>.wav.")
                    else:
                        label = int(file_parts[0])
                        speaker = file_parts[1]
                        samplerate, audio = wavfile.read(os.path.join(path, file))

                        if trimmed:
                            audio = trim_silence(audio)

                        audio = torch.tensor(audio, dtype=torch.float64)

                        wav_file_count += 1

                        rows_list.append({'labels': label, 'speakers': speaker, 'audio': audio , 'spectrograms': None})

            if wav_file_count == 0:
                raise ValueError("The specified path does not contain any wav files.")
            else:
                self.dataset = pd.DataFrame(rows_list, columns=['labels', 'speakers', 'audio','spectrograms'])
        else:
            raise ValueError("The specified path does not exist.")

        self.sampling_rate = 8000  # All recordings should be mono 8kHz

    def get_audio_from_index(self, index):
        """
        Returns the audio data for a given index.

        Parameters:
        - index (int): The index of the item in the dataset.

        Returns:
        - audio (ndarray): The audio data.
        """
        return self.dataset['audio'][int(index)]

    def get_spectrogram_from_index(self, index):
        """
        Returns the spectrogram data for a given index.

        Parameters:
        - index (int): The index of the item in the dataset.

        Returns:
        - spectrogram (ndarray): The spectrogram data.
        """
        return self.dataset['spectrograms'][int(index)]

    def get_label_from_index(self, index):
        """
        Returns the label for a given index.

        Parameters:
        - index (int): The index of the item in the dataset.

        Returns:
        - label (int): The label.
        """
        return self.dataset['labels'][int(index)]

    def get_speaker_from_index(self, index):
        """
        Returns the speaker name for a given index.

        Parameters:
        - index (int): The index of the item in the dataset.

        Returns:
        - speaker (str): The speaker name.
        """
        return self.dataset['speakers'][int(index)]

    def get_item_from_index(self, index):
        """
        Returns a dictionary containing audio, spectrogram, label, and speaker data for a given index.

        Parameters:
        - index (int): The index of the item in the dataset.

        Returns:
        - item (dict): A dictionary containing audio, spectrogram, label, and speaker data.
        """
        return {
            'audio': self.get_audio_from_index(index),
            'spectrograms': self.get_spectrogram_from_index(index),
            'label': self.get_label_from_index(index),
            'speaker': self.get_speaker_from_index(index),
        }

    def get_speakers(self):
        """
        Returns an array of unique speaker names in the dataset.

        Returns:
        - speakers (ndarray): An array of unique speaker names.
        """
        return np.unique(self.dataset['speakers'].to_numpy())

    def get_labels(self):
        """
        Returns an array of unique labels in the dataset.

        Returns:
        - labels (ndarray): An array of unique labels.
        """
        return np.unique(self.dataset['labels'].to_numpy())

    def get_sample(self, sample_size = None, digits=None, speakers=None, min_length=None, max_length=None):
        """
        Returns a sample of items from the dataset that match the specified criteria.

        Parameters:
        - sample_size (int): The number of items to include in the sample.
        - digits (list, optional): A list of digits that should be included in the sample. If not specified, any digit can be included.
        - speakers (list, optional): A list of speaker names that should be included in the sample. If not specified, any speaker can be included.
        - min_length (int, optional): The minimum length of audio that should be included in the sample. If not specified, there is no minimum length requirement.
        - max_length (int, optional): The maximum length of audio that should be included in the sample. If not specified, there is no maximum length requirement.

        Returns:
        - sample (list): A list of items from the dataset that match the specified criteria.

        Raises:
        - ValueError: If there are not enough items in the dataset that match the specified criteria.
        """
        sample = []
        indices = []


        idx_shuffled = np.arange(len(self.dataset))
        np.random.shuffle(idx_shuffled)

        j = 0

        if sample_size is None:
            sample_size = len(self.dataset)
        else:
            if sample_size == len(self.dataset):
                raise ValueError("The sample size is equal to the size of the dataset. Please specify a smaller sample size, or leave the field empty if you want all the samples fitting the other criterions.")

        while len(indices) < sample_size and j < len(self.dataset):
            i = idx_shuffled[j]
            valid = True

            if digits is not None:
                label = self.get_label_from_index(i)
                if label not in digits:
                    valid = False

            if speakers is not None:
                speaker = self.get_speaker_from_index(i)
                if speaker not in speakers:
                    valid = False
            
            if min_length is not None:
                audio = self.get_audio_from_index(i)
                if len(audio) < min_length:
                    valid = False
            
            if max_length is not None:
                audio = self.get_audio_from_index(i)
                if len(audio) > max_length:
                    valid = False

            if valid:
                indices.append(i) 
            
            j+=1

        if len(indices) < sample_size and sample_size < len(self.dataset): # the second condition is false if the user didn't specify a sample size
            raise ValueError("Not enough items in the dataset that match the specified criteria.")

        sample_indices = np.array(indices)

        for index in sample_indices:
            sample.append(self.get_item_from_index(index))

        return sample
    


if __name__ == "__main__":
    # Example usage
    # Load the Spoken MNIST dataset
    spoken_mnist = SpokenMnistWrapperAPI()

    # Get the names of the speakers in the dataset
    speakers = spoken_mnist.get_speakers()

    # Get 5 samples of digits 0 and 1 spoken by speakers 0 and 1
    print(spoken_mnist.get_sample(5, digits=[0, 1], speakers=speakers[:2]))