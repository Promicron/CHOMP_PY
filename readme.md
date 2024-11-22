# CHOMP-PY

This repository contains a Python implementation of the CHOMP (Covariant Hamiltonian Optimization for Motion Planning) algorithm. It uses the PyBullet environment for simulation.

## Features

- Motion planning for a robotic arm with 7 degrees of freedom (Franka Panda URDF).
- Optimization of trajectories considering both:
  Smoothness Cost: Penalizing abrupt changes in trajectory.
  Collision Cost: Avoiding obstacles in the environment.
- Linear interpolation for initializing trajectories.
- Visualization of the optimization process with plots for smoothness, collision, and overall cost values.

## Dependencies

- Python 3.10
- numpy
- pybullet
- matplotlib

## Installation

### Using Anaconda:
Run this on the terminal to create a new environment and install the dependencies:
`conda create -n chomp_py python=3.10`
`conda activate pybullet_robot_base`
`conda install -c conda-forge pybullet numpy matplotlib`

## Usage
Run the Script: Execute the script in your terminal:

`python CHOMP_T1.py`
