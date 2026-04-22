# SpecUNet

SpecUNet is a repository for denoising and analyzing spectroscopic single-molecule localization microscopy (sSMLM) data using a U-Net-based deep learning workflow. This project includes both MATLAB and Python implementations of the SpecUNet framework for simulation, training, testing, and evaluation.

The work is associated with the following article:

[Mao, H. et al. “Framework for Accurate Single-Molecule Spectroscopic Imaging Analyses Using Monte Carlo Simulation and Deep Learning.” *Analytical Chemistry* (2025)](https://pubs.acs.org/doi/full/10.1021/acs.analchem.5c01486)

## Overview

SpecUNet is designed to support the following tasks:

- Simulate Monte Carlo training and testing data for sSMLM image denoising
- Train a U-Net-based model to predict background and recover denoised spectral images
- Evaluate model performance on simulated and experimental datasets
- Support reproducible workflows in both MATLAB and Python

## Repository Structure

```text
.
├── MATLAB/     # MATLAB implementation and detailed README
├── Python/     # Python implementation and detailed README
└── README.md   # Main repository overview