# Hamiltonian learning via the quantum Zeno effect

This repository contains simulation code, tutorials, and IBM Quantum notebooks for Hamiltonian learning using unitary Zeno kicks and local quantum process tomography.

The main simulation reconstructs the coefficients of a nearest-neighbor spin-chain Hamiltonian. Z kicks act on separator qubits, and tomography reconstructs the dynamics of the small patches between them. The patch estimates are combined with corrections for separator-induced coefficient shifts.

Two small-scale extensions demonstrate reconstruction for a chain with three-body interactions and a two-dimensional square lattice.

Start with [tutorial.ipynb](tutorial.ipynb) for the main simulation, data-analysis, and plotting workflow. The two extension examples are presented in [tutorial_extensions.ipynb](tutorial_extensions.ipynb).

## Installation

From the repository root, create and activate the Conda environment provided in `environment.yml`:

```sh
conda env create --name hl_w_zeno --file environment.yml
conda activate hl_w_zeno
```

Open the tutorials in your notebook editor and select a Python kernel from the `hl_w_zeno` environment. Keep the main tutorial's working directory at the repository root so its relative file paths resolve correctly.

## Main tutorial

[tutorial.ipynb](tutorial.ipynb) runs the chain simulation for selected system sizes, shot budgets, and Hamiltonian seeds. It then reads the saved estimates and plots the Euclidean coefficient error against the total number of measurement copies.

The notebook retains the original grid of 420 runs. For a small first run, change these values in the **Parameter Setup** cell, leaving the parameter-generation loop below them in place:

```python
shots_vec = [1e2, 1e3, 1e4]
N_vec = [3]
n_averages = 1
```

Run the notebook cells in order. The batch creates its output directory automatically. The final cells plot the mean reconstruction error for each chain length and number of measurement copies.

The batch stores its results in `data/paper/`, with filenames identifying the system size, evolution time, shot count, and seed. Repeating a run with the same filename replaces the previous output.

## Running a single simulation

The main script also accepts a JSON configuration. For example, save the following as `config.json` in the repository root:

```json
{
  "N": 3,
  "N_QPT": 2,
  "T": 0.01,
  "n_steps": 10,
  "N_shots": 10000,
  "seed": 0,
  "plot": false,
  "save": true,
  "ising_like": false
}
```

Then run:

```sh
python hl_w_zeno_TN.py --config config.json
```

| Parameter | Meaning |
| --- | --- |
| `N` | Total number of qubits in the chain. |
| `N_QPT` | Number of qubits in each tomography patch; keep this at `2` for the main script. |
| `T` | Total Hamiltonian evolution time. |
| `n_steps` | Integer number of evolution segments and kick rounds; the example uses an even count of `10`. |
| `N_shots` | Number of samples per binary tomographic setting, rather than the total copy budget. |
| `seed` | Random seed used by the simulation. |
| `plot`, `save` | Whether to display the coefficient-error plot and write the results. |
| `ising_like` | Whether to use the restricted Z-field and XX-coupling model instead of the general nearest-neighbor Pauli model. |

With `save` enabled, the script writes `results.txt` in the working directory. It contains reconstructed coefficients, target coefficients, absolute coefficient differences, and the patch Choi matrices. The tutorial renames this file after each batch run.

## Extension examples

[tutorial_extensions.ipynb](tutorial_extensions.ipynb) presents two small-system examples:

- An eight-site chain with interactions involving up to three qubits within three consecutive sites.
- An open three-by-three square lattice with on-site terms and nearest-neighbor two-qubit interactions.

Each example shows the model and patches, followed by the explicit evolution, patch tomography and coefficient-assembly steps. The substantial numerical functions are in `extension_simulations.py`, which reuses the main simulator's fiducial-state and polar-projection functions. The examples use `T=0.01`, 12 kick layers and approximately ten billion total copies per Hamiltonian. Their executed plots are saved in the notebook. The JSON files in `data/extensions/` provide the reported ten-seed results.

## Experimental data and IBM notebooks

The recorded input [datos_100q_RC.json](IBM_experiment/datos_100q_RC.json) is included in `IBM_experiment/`. See the [hardware notebook guide](IBM_experiment/README.md) for the data files and setup instructions.

The [color-map notebook](IBM_experiment/plot_color_map.ipynb) includes a short saved-results plotting section using [params_100q_RC.json](IBM_experiment/params_100q_RC.json). It displays the original-count coefficient deviations without an IBM account. Its earlier SDK cells show reconstruction from the recorded shots.

The resampling notebook implements the paper's 1,000-resample procedure using the recorded shots and supplied original-count unitary fits. Its numerical helpers are included, and it runs without an IBM account or Qiskit. See the hardware guide for the seed, fitting procedure and uncertainty definitions. The circuit-execution notebook submits new hardware experiments and requires the author's IBM Quantum configuration and additional acquisition dependencies.

The complete simulation and experimental datasets are associated with [Zenodo, DOI 10.5281/zenodo.17351462](https://doi.org/10.5281/zenodo.17351462). The hardware input needed by these notebooks is supplied directly here.

## Code Overview

### Simulation Code

- **`hl_w_zeno_TN.py`** — main tensor-network simulation script for Hamiltonian learning with repeated unitary Z kicks.
- **`utilities.py`** — helper functions for state evolution, measurement operations, and data handling.
- **`tutorial.ipynb`** — interactive notebook demonstrating the full workflow: setup → simulation → analysis.
- **`tutorial_extensions.ipynb`** — small-scale examples for a chain with three-body interactions and a two-dimensional square lattice.
- **`extension_simulations.py`** — shared evolution, patch-tomography and coefficient-assembly functions used by the extension examples.
- **`environment.yml`** — conda environment setup file.

### IBM Quantum Experiments (`IBM_experiment/`)

- **`execute_hamiltonian_learning.ipynb`** — performs Hamiltonian learning on IBM Quantum backends.
- **`execute_montecarlo_resampling.ipynb`** — applies Monte Carlo resampling to estimate uncertainties from experimental data.
- **`plot_color_map.ipynb`** — visualizes experimental results as color maps for easier interpretation.
