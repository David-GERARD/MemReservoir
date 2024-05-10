import numpy as np
import matplotlib.pyplot as plt

def generate_colormap(x, colormap = 'turbo'):
    """
    Function that generates a colormap for a given array of values, with color being associated to x coordinate value

    Parameters:
    - x (np.array): Array of x coordinates values to be used to generate the colormap.
    - colormap (str, optional): Name of the colormap to be used. Default is 'turbo'.

    Returns:
    - colors (np.array): Array of colors associated with the x values.

    """

    # Define the color map
    cmap = plt.cm.get_cmap(colormap)

    # Normalize the X values to the range [0, 1]
    normalized_x = (x - np.min(x)) / (np.max(x) - np.min(x))

    # Generate a list of colors based on the normalized X values
    colors = cmap(normalized_x)

    return colors
