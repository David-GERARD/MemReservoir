# Memristor-Enabled Reservoir Computing for Spoken-MNIST Digit Recognition

## Project Overview
In this project, we present a Python pipeline system designed to simulate memristor behavior within RC frameworks. By optimizing memristor model parameters, our objective is to achieve superior accuracy and reduce output layer training time.

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
