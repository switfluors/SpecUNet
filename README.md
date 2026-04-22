# SpecUNet

SpecUNet is a repository for denoising and analyzing spectroscopic single-molecule localization microscopy (sSMLM) data using a U-Net-based deep learning workflow. The project is based on a framework that combines Monte Carlo simulation, supervised learning, and quantitative evaluation for accurate single-molecule spectral image analysis. :contentReference[oaicite:0]{index=0}

This repository contains two implementations of the SpecUNet workflow:

- **MATLAB/** — MATLAB scripts for data simulation, network training, and evaluation
- **Python/** — A configurable Python implementation for training, testing, and metric-based evaluation workflows :contentReference[oaicite:1]{index=1}

## Overview

SpecUNet is designed to support the following tasks:

- Simulate training and testing data for sSMLM image denoising
- Train a U-Net-based model to predict background and recover denoised spectral images
- Evaluate model performance on simulated or experimental datasets
- Provide a foundation for reproducible development across MATLAB and Python implementations 

## Repository Structure

```text
.
├── MATLAB/     # MATLAB implementation and documentation
├── Python/     # Python implementation and documentation
└── README.md   # Main repository overview

