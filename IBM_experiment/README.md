# Hardware experiment notebooks

Run these notebooks with `IBM_experiment/` as the working directory.

## Data

- `datos_100q_RC.json`: recorded shots (972 settings, 900 shots each).
- `params_100q_RC.json`: coefficient estimates from the recorded data, used by the color map.
- `original_count_unitaries.npz`: fitted unitaries used to initialize resampling, with their patch order and qubit layout.

## Notebooks

- **`execute_montecarlo_resampling.ipynb`** estimates coefficient uncertainties using the paper's 1,000-resample procedure. Run cells in order; reduce `num_resamples` for a shorter trial. It writes `params_100q_RC_mc1000.json` with the original coefficient estimates and sample standard deviations.
- **`plot_color_map.ipynb`** displays coefficient deviations. Its **Plot the saved experimental estimates** section runs independently using the supplied coefficient JSON.
- **`execute_hamiltonian_learning.ipynb`** submits new hardware experiments and requires IBM/Qiskit packages and an IBM backend configuration. Additional acquisition helpers are available from the authors on request.

Offline resampling needs NumPy and SciPy; plotting also needs Matplotlib. Both use the root environment without an IBM account. The numerical helpers are in `main/tomography.py` and `main/resampling.py`.

The original `params_100q_RC.json` stores raw overlap scores under `lambda`. The resampling output reports `depolarizing_strength` with explicit `median` and `bootstrap_sd` entries.
