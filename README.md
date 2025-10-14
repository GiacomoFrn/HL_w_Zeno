# Hamiltonian Learning with Quantum Zeno Effect

A tensor network implementation for learning Hamiltonian parameters of spin chains using quantum process tomography with the Quantum Zeno effect.

## Overview

This project implements Hamiltonian learning for spin chain systems by:
- Simulating quantum evolution with repeated Zeno measurements
- Performing quantum process tomography on adjacent qubit pairs
- Reconstructing Hamiltonian coefficients from measurement probabilities
- Using tensor networks (TeNPy) for efficient simulation of many-body systems

## Requirements

- Python 3.11
- numpy, scipy, matplotlib
- physics-tenpy (tensor network library)
- cvxpy, sympy, dask
- tqdm (progress bars)

Install dependencies via conda:
```bash
conda env create -f environment.yml
conda activate hl_w_zeno
```

## Usage

Run with default parameters:
```bash
python hl_w_zeno_TN.py
```

Customize parameters:
```bash
python hl_w_zeno_TN.py --N 7 --T 0.01 --N_shots 100000 --seed 43
```

Or use a JSON config file:
```bash
python hl_w_zeno_TN.py --config config.json
```

### Key Parameters

- `--N`: Number of spins in the chain (default: 7)
- `--N_QPT`: Number of qubits for process tomography (default: 2)
- `--T`: Total evolution time (default: 1e-2)
- `--n_steps`: Number of evolution steps (default: 1e-5)
- `--N_shots`: Number of measurement shots (default: 100000)
- `--ising_like`: Use Ising-like Hamiltonian instead of random (default: False)
- `--plot`: Display reconstruction results (default: True)
- `--save`: Save results to text file (default: False)

## Output

The script outputs:
- Reconstructed Hamiltonian coefficients
- Comparison with true coefficients
- Logarithmic plot of reconstruction errors (if `--plot True`)
- Optional text file with detailed results (if `--save True`)

## Components

- **hl_w_zeno_TN.py**: Main script implementing Zeno protocol and Hamiltonian learning
- **utilities.py**: Hilbert space, density matrix, and quantum state utilities
- **tutorial.ipynb**: Jupyter notebook with examples and demonstrations
