import numpy as np

class SpikeTrainPreprocessor:
    """
    Processes audio data from the MNIST 784 dataset to be used as input for a memristor-based reservoir computing system.
    data: https://www.openml.org/search?type=data&sort=runs&id=554
    
    Import with:
    from sklearn.datasets import fetch_openml

    # Load the MNIST dataset
    mnist = fetch_openml('mnist_784')

    # Access the data and target variables
    X = mnist.data.to_numpy().reshape(-1, 28, 28)
    y = mnist.target
    

    This preprocessing method is introduced in "Reservoir computing using dynamic memristors for temporal information processing" by Chao Du, et al.
    https://doi.org/10.1038/s41467-017-02337-y

    """
    def __init__(self, t_pixel=3e-3, spike_amplitude=3.0, spike_duration=1e-3, digitalize = True):
        """
        
        Parameters:
        - t_pixel: time duration of a pixel in seconds
        - spike_amplitude: amplitude of the spike in Volts
        - spike_duration: duration of the spike in seconds
        - digitalize: if True, the spike amplitude is the same for all pixels, otherwise the amplitude is proportional to the pixel value
        """
        
        self.t_pixel = t_pixel
        self.spike_amplitude = spike_amplitude #V
        self.spike_duration = spike_duration #s

        self.digitalize = digitalize


    def generate_spikes(self, row , dt):
        """
        Generates a spike train for a row of pixels

        Parameters:
        - row: a row of pixels
        - dt: time step for the spike train

        Returns:
        - train: a spike train for the row
        """
        train = np.zeros(int(len(row)*self.t_pixel//dt)+1)
        for i, pixel in enumerate(row):
            if self.digitalize and pixel > 0:
                train[int(i*self.t_pixel//dt):int(i*self.t_pixel//dt +int(self.spike_duration//dt))] = self.spike_amplitude
            elif not self.digitalize:
                train[int(i*self.t_pixel//dt):int(i*self.t_pixel//dt +int(self.spike_duration//dt))] = self.spike_amplitude*pixel

        return train
    
    def normalize(self, image):
        """
        Normalizes the pixels of the image to be between 0 and 1

        Parameters:
        - image: a 28x28 pixel image

        Returns:
        - image: the normalized image
        """
        return image / 255.0
        

    def process_image(self, image, dt): 
        """
        Processes an image to generate a spike train

        Parameters:
        - image: a 28x28 pixel image
        - dt: time step for the spike train

        Returns:
        - t: time array for the spike train
        - spikes: a list of spike trains for each row of the image
        """

        image = self.normalize(image)
        
        t = np.arange(0, self.t_pixel*image.shape[1], dt)
        spikes = []
        for row in range(image.shape[0]):
            spikes.append(self.generate_spikes(image[row],  dt))
        return t, spikes