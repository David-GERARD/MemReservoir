# This file contains the code to encode signals to be fed to a delay-dynamical reservoir computing system, as described in the paper by Appeltant et al. (2011).
# https://doi.org/10.1038/ncomms1476

# Code by David Gerard
# Email: david.gerard.23@ucl.ac.uk



import torch
import numpy as np
import sys
import os
import pandas as pd
import argparse



def sampleAndHold(u, t, tau, ts=None):
    """
    Samples an input signal `u` at specified intervals `tau` and optionally resamples the signal to a new timestep `ts`.
    
    Parameters:
    - u (torch.Tensor or numpy.ndarray): The input signal values.
    - t (torch.Tensor or numpy.ndarray): The time points corresponding to the signal values, in milliseconds.
    - tau (int): The sampling period in milliseconds, determining how often to sample the input signal.
    - ts (int, optional): The new timestep in milliseconds for resampling the signal. If not specified, no resampling is done.
    
    Returns:
    - torch.Tensor: The sampled (and optionally resampled) signal.
    - torch.Tensor: The time points corresponding to the sampled (and optionally resampled) signal.
    
    The function first checks if the input arrays are torch tensors, converting them if necessary. It then samples the
    input signal `u` at the specified intervals `tau`, and if `ts` is provided, it resamples this sampled signal to have
    each point separated by `ts` milliseconds.
    """

    # Check if inputs are torch tensors, convert if they are numpy arrays
    if not torch.is_tensor(u):
        u = torch.tensor(u)
    if not torch.is_tensor(t):
        t = torch.tensor(t)

    # Initialize the sampled signal with zeros
    sampled_signal = torch.zeros_like(u)

    # Perform sampling
    last_sampled_value = 0
    for i in range(len(t)):
        if i == 0 or t[i] % tau == 0:
            last_sampled_value = u[i]
        sampled_signal[i] = last_sampled_value

    # Resample if ts is provided
    if ts is not None:
        # Calculate the number of points in the resampled signal
        num_points = int((t[-1] - t[0]).item() / ts) + 1
        resampled_signal = torch.zeros(num_points)
        resampled_times = torch.linspace(t[0].item(), t[-1].item(), num_points)

        # Fill in the resampled signal values
        j = 0
        for i in range(len(resampled_signal)):
            while j + 1 < len(t) and t[j + 1] <= resampled_times[i]:
                j += 1
            resampled_signal[i] = sampled_signal[j]

        return resampled_signal,resampled_times

    return sampled_signal,t

def generateMask(N):
    """
    Generates a random mask array of size N, where each element is either 1 or -1, chosen randomly.
    
    Parameters:
    - N (int): The size of the mask array to generate.
    
    Returns:
    - numpy.ndarray: An array of size N, where each element is randomly set to either 1 or -1.
    
    This function utilizes numpy's random.choice method to fill the array with -1 or 1 values.
    """
    M = np.random.choice([-1, 1], size = N)
    return M

def applyMask(I,M):
    """
    Applies a mask `M` to an input signal `I`, element-wise, with periodic extension of the mask if needed.
    
    Parameters:
    - I (torch.Tensor or numpy.ndarray): The input signal to mask.
    - M (torch.Tensor or numpy.ndarray): The mask to apply, where each element is either 1 or -1.
    
    Returns:
    - torch.Tensor: The masked signal, which is the element-wise multiplication of `I` by `M`, extending `M` periodically if `I` is longer than `M`.
    
    The function first checks if the input `I` and mask `M` are torch tensors, converting them if necessary. It then applies the mask `M` to the input signal `I`, extending the mask periodically if `I` is longer than `M`.
    """
    # Check if inputs are torch tensors, convert if they are numpy arrays
    if not torch.is_tensor(I):
        I = torch.tensor(I)
    if not torch.is_tensor(M):
        M = torch.tensor(M)

    N = len(M)
    # Initialize the sampled signal with zeros
    J = torch.zeros_like(I)

    for i in range(len(I)):
        J[i] = I[i]*M[i%N]

    return J




# if used as a script
if __name__ == "__main__":
    # Parsing arguments
    parser=argparse.ArgumentParser(description="Preprocessing script for memristor based reservoir computing")
    parser.add_argument("input_file",  help='path to the file containing the time series to preprocess')
    parser.add_argument("output_folder", help='path to the folder in which to store the output of the preprocessing pipeline')
    parser.add_argument("N", type=int, help='Number of virtual nodes in the reservoir')
    parser.add_argument("tau", type=int, help='Sampling period')

    parser.add_argument("-ts", type=int, help='Desired output sampling period (for upsampling, must be smaller than tau/N)')

    args=parser.parse_args()

    if not os.path.exists(args.input_file):
        input_file = args.input_file
        sys.exit(f"File {input_file} does not exist")

    if not os.path.exists(args.output_folder):
        output_folder = args.output_folder
        sys.exit(f"File {output_folder} does not exist")

    if args.ts is not None and args.ts > tau/N:
        sys.exit("ts must be smaller than tau/N")

    # Opening file
    data = pd.read_csv(args.input_file, sep=" ", header=None).to_numpy()

    # Check that file has the right format
    if data.shape[1] != 2:
        sys.exit("Input error: the input file must contain two columns, the first one with the time in ms and the second one with the input values")
    
    # Execute pipeline
    input = data[:,1]
    times = data[:,0]

    N = args.N
    tau =args.tau
    theta = tau/N

    sampled_signal,sampled_times = sampleAndHold(u, t, tau, ts=args.ts)
    mask = generateMask(N)
    J = applyMask(sampled_signal,mask)

    prepocessed_input = torch.t(torch.cat([sampled_times,J],0))

    pd.DataFrame(prepocessed_input.numpy()).to_csv(args.output_folder + "preprocessed_"+os.path.basename(args.input_file))


        

