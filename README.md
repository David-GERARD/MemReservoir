# Memristor-Enabled Reservoir Computing for Spoken-MNIST Digit Recognition

## Project Overview
In this project, we present a Python pipeline system designed to simulate memristor behavior within reservoir computing frameworks. By optimizing memristor model parameters, our objective is to achieve superior accuracy and reduce output layer training time.

### Memristors
A memristor is a two-terminal electronic component whose resistance changes based on the history of the electric charge that has flowed through it, exhibiting memory-like behavior.

The concept of the memristor was theorized by Leon Chua in 1971. However, the first physical realization of a memristor was reported by HP Labs researchers in 2008.

### Reservoir computing 
Reservoir Computing (RC) is a computational framework for processing sequential data, particularly suited for tasks like time-series prediction and pattern recognition. It involves feeding input data into a fixed, often randomly initialized, high-dimensional dynamical system called the "reservoir." The reservoir's complex dynamics transform the input data nonlinearly, creating rich representations that can be further processed to achieve desired outputs through a trainable readout layer. This separation of feature extraction (done by the reservoir) and output generation (done by the readout layer) simplifies training and often leads to efficient and high-performing models.


## Key Objectives
- Memristor Simulation: Develop a flexible simulator capable of emulating memristor functionality within RC architectures.
- Parameter Optimization: Explore and fine-tune memristor model parameters to enhance system performance.
- Impact Assessment: Investigate the effects of volatility and light sensitivity on RC system behavior and accuracy.
- Benchmarking: Evaluate the effectiveness of the memristor-enabled RC system against the spoken-MNIST digit recognition task, leveraging Lyon's auditory model for input processing.

## Conda enviroment

- Open a terminal, navigate to the root folder of the repository.
- Run the command `make env`, enter the password of your session, press enter.
- If you add a package to requirements.txt, or want to update your packages to their latest version, run the command `make updates`

## TBD 
This repository is a work in progress. The following need to be finished in order to achieve a stable version.

- Add the compiling of the C and Matlab code for the Lyon auditory model python wrapper functions to the Makefile.
- Implement memristor models (with PySpice or directly in python)
- Implement output layers of the RC systems.
- Run full pipeline for training.
- Perform parameter fine tuning.
- Implement an automated procedure for finding memristor model parameters that optimize RC system performances (accuracy and training time).
