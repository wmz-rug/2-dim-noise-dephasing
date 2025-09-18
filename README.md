# 2-dim-noise-dephasing

## Project Overview

This repository contains the simulation code accompanying the paper:

**Destructive Interference of Inertial Noise in Matter-wave Interferometry**

Accepted by Phys.Rev.D (2025)

DOI: *TBD*

arxiv: [2507.00280] https://arxiv.org/abs/2507.00280v2

## Repository Structure

```plaintext
.
├── simulation.py            # Main simulation Python file for Spyder
├── simulation.ipynb         # Main simulation notebook (same as simulation.py)
├── environment.yml          # Conda environment file 
├── LICENSE                  # Open-source license 
└── README.md                # Project documentation
```

## Environment Setup

You can set up the environment in one of two ways:

Option 1:
```
conda env create -n sim-env spyder numpy matplotlib scipy
conda activate sim-env
```

Option 2:
```
conda env create -f environment.yml
conda activate sim-env
```

## Contact

For questions or feedback, feel free to reach out via [mengzhi.wu@rug.nl] or open an issue in this repository.
