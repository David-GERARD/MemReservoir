from sklearn.datasets import fetch_openml
import numpy as np

def load_mnist():
    # Load the MNIST dataset
    mnist = fetch_openml('mnist_784')

    # Access the data and target variables
    X = mnist.data.to_numpy().reshape(-1, 28, 28)
    y = mnist.target

    return X, y
