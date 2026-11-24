"""Small-system extensions of the Zeno Hamiltonian-learning tutorial."""

import os
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, groupby, product
from types import SimpleNamespace

for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import numpy as np
from scipy.linalg import eigh

from hl_w_zeno_TN import get_fiducial_states_arr, complex2unitary

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.diag([1, -1]).astype(complex)
PAULI = [I, X, Y, Z]
FIDUCIALS = np.asarray(get_fiducial_states_arr(1, False))
SINGLE_SITE_DUALS = ((I-X-Y+Z)/2, (I-X-Y-Z)/2, X, Y)
PAULI_BY_LABEL = dict(zip("IXYZ", PAULI))
MEASUREMENT_BASES = [np.stack((ket.conj(), [-ket[1], ket[0]])) for ket in FIDUCIALS]

def kron_all(items):
    """Return the tensor product in site order."""
    out=np.array([1],complex)
    for item in items: out=np.kron(out,item)
    return out

def full_basis(n=2):
    """Return nonidentity Pauli matrices and labels in product order."""
    matrices=[];labels=[]
    for inds in product(range(4),repeat=n):
        if not any(inds): continue
        matrices.append(kron_all([PAULI[i] for i in inds]));labels.append(''.join('IXYZ'[i] for i in inds))
    return np.array(matrices),labels

def coefficients(U,T,basis):
    """Extract normalized Pauli traces of the phase-fixed unitary logarithm."""
    U=phase(U)
    vals,vecs=np.linalg.eig(U)
    A=(vecs*(-np.angle(vals)))@np.linalg.inv(vecs)
    A=(A+A.conj().T)/2
    return np.real(np.einsum('aij,ji->a',basis,A))/(len(U)*T)

def phase(U):
    """Choose positive real trace for the unitary representative."""
    trace = np.trace(U)
    if abs(trace) <= len(U) * np.finfo(float).eps:
        raise ValueError("Unitary trace is too small to fix its global phase.")
    return U * (np.conj(trace) / abs(trace))


def ep(J):
    """Recover the unitary polar factor of the leading Choi eigenvector."""
    values, vectors = np.linalg.eigh((J + J.conj().T)/2)
    dimension = int(round(np.sqrt(len(J))))
    raw = np.sqrt(dimension) * vectors[:, -1].reshape(dimension, dimension)
    return np.asarray(complex2unitary(raw), dtype=complex), values

@dataclass(frozen=True)
class Layout:
    """Store the active patches and two separator classes for one schedule."""
    patches: tuple[tuple[int, ...], ...]
    separators: tuple[tuple[int, ...], tuple[int, ...]]


@dataclass(frozen=True)
class Design:
    """Store the product-state tomography design for a patch."""
    states: np.ndarray
    effects: np.ndarray
    duals: np.ndarray
    paulis: np.ndarray
    labels: list[str]


@lru_cache(maxsize=None)
def tomography_design(size):
    """Build and cache the fiducial states, duals and Pauli basis."""
    states = np.array([kron_all(row) for row in product(FIDUCIALS, repeat=size)])
    effects = np.einsum("ei,ej->eij", states, states.conj())
    duals = np.array([kron_all(row) for row in product(SINGLE_SITE_DUALS, repeat=size)])
    paulis, labels = full_basis(size)
    return Design(states, effects, duals, paulis, labels)


def recover_patch(design, probabilities, duration):
    # Normalized Choi matrix in output-input ordering, matching the EP routine.
    """Recover local Hamiltonian coefficients from a patch probability table."""
    outputs = np.einsum("ez,eij->zij", probabilities, design.duals, optimize=True)
    size = design.states.shape[1]
    choi = np.einsum("zij,zkl->ikjl", outputs, design.duals.transpose(0, 2, 1))
    unitary, _ = ep(choi.reshape(size**2, size**2) / size)
    return dict(zip(design.labels, coefficients(unitary, duration, design.paulis)))


def assembly_rules(labels, layouts):
    """Identify repeated observations and their higher-weight boundary donors."""
    supports = [{i for i, factor in enumerate(label) if factor != "I"} for label in labels]
    exact_observations = [[] for label in labels]
    sources = {}
    for layout_id, layout in enumerate(layouts):
        partition = [site for block in (*layout.patches, *layout.separators) for site in block]
        if sorted(partition) != list(range(len(labels[0]))):
            raise ValueError(f"Layout {layout_id} must partition the system")
        separators = set().union(*layout.separators)
        patch_sites = [set(patch) for patch in layout.patches]
        for index, (label, support) in enumerate(zip(labels, supports)):
            if any(len(support.intersection(color)) > 1 for color in layout.separators):
                raise ValueError(f"Two separator sites of one color in term {label}, layout {layout_id}")
            touched = [i for i, patch in enumerate(patch_sites) if support.intersection(patch)]
            if len(touched) > 1:
                raise ValueError(f"Term {label} spans two patches in layout {layout_id}")
            if not touched or any(label[i] in "XY" for i in separators):
                continue
            patch_id = touched[0]
            local = "".join(label[i] for i in layout.patches[patch_id])
            key = (layout_id, patch_id, local)
            sources.setdefault(key, []).append(index)
            if support <= patch_sites[patch_id]:
                exact_observations[index].append(key)

    order = sorted(range(len(labels)), key=lambda index: -len(supports[index]))
    rules = []
    for index in order:
        observations = exact_observations[index]
        if not observations:
            raise ValueError(f"No patch contains the support of {labels[index]}")
        equations = []
        for key in observations:
            donors = [other for other in sources[key] if other != index]
            if any(not supports[index] < supports[other] for other in donors):
                raise ValueError(f"Nontriangular boundary shift for {labels[index]}")
            equations.append((key, donors))
        rules.append((index, equations))
    return rules


def assemble(patch_estimates, rules):
    """Subtract boundary contributions before averaging repeated estimates."""
    corrected = np.empty(len(rules))
    uncorrected = np.empty(len(rules))
    for index, equations in rules:
        local_values = [patch_estimates[key] for key, _ in equations]
        corrected[index] = np.mean([
            value - sum(corrected[donor] for donor in donors)
            for value, (_, donors) in zip(local_values, equations)
        ])
        uncorrected[index] = np.mean(local_values)
    return corrected, uncorrected


def frame_signs(n_sites, layout):
    """Return computational-basis signs for the four toggling frames."""
    signs = []
    for sites in layout.separators:
        mask = sum(1 << (n_sites - 1 - site) for site in sites)
        signs.append(np.array([(-1) ** ((bits & mask).bit_count() % 2) for bits in range(2**n_sites)]))
    first, second = signs
    return np.ones(2**n_sites), first, first * second, second


def unitary_from_eigensystem(energies, vectors, time):
    """Construct the free-evolution unitary at the given time."""
    return (vectors * np.exp(-1j * energies * time)) @ vectors.conj().T


def four_frame_evolution(free, frames, cycles):
    # Physical pulse layers A, B, A, B visit I, A, AB, B, then reset to I.
    """Repeat the four-frame cycle and return its full-system unitary."""
    cycle = free
    for signs in frames[1:]:
        cycle = (signs[:, None] * free * signs[None, :]) @ cycle
    return np.linalg.matrix_power(cycle, cycles)


def exact_probabilities(evolved, effects):
    """Evaluate the patch effects on each reduced output state."""
    states = np.einsum("zie,zje->zij", evolved, evolved.conj())
    return np.real(np.einsum("eij,zji->ez", effects, states, optimize=True))


def ideal_patch_data(hamiltonian, n_sites, layout, frames, duration):
    # Compute the dense frame average independently of the coefficient rules.
    """Recover patch coefficients from the exact frame-averaged Hamiltonian."""
    average = sum(signs[:, None] * hamiltonian * signs[None, :] for signs in frames) / 4
    estimates = []
    phase_radii = []
    for patch in layout.patches:
        indices = [
            sum(bit << (n_sites - 1 - site) for bit, site in zip(bits, patch))
            for bits in product((0, 1), repeat=len(patch))
        ]
        local = average[np.ix_(indices, indices)]
        local -= np.eye(len(indices)) * np.trace(local) / len(indices)
        values, vectors = eigh(local)
        phase_radii.append(duration * np.max(np.abs(values)))
        unitary = unitary_from_eigensystem(values, vectors, duration)
        design = tomography_design(len(patch))
        outputs = (unitary @ design.states.T).T[:, :, None]
        estimates.append(recover_patch(design, exact_probabilities(outputs, design.effects), duration))
    return estimates, max(phase_radii)


def global_preparations(layout, n_sites, settings):
    """Prepare matching fiducial inputs on parallel patches and separators in zero."""
    position = {site: slot for patch in layout.patches for slot, site in enumerate(patch)}
    states = []
    for setting in settings:
        # All separators are in |0>; target patches use the same product design in parallel.
        factors = [FIDUCIALS[setting[position[i]]] if i in position else FIDUCIALS[0]
                   for i in range(n_sites)]
        states.append(kron_all(factors))
    return np.array(states).T


def sampled_probabilities(evolved, layout, n_sites, settings, shots, rng):
    """Marginalize joint multinomial shots onto the parallel patch events."""
    active = tuple(site for patch in layout.patches for site in patch)
    separators = tuple(site for color in layout.separators for site in color)
    position = {site: slot for patch in layout.patches for slot, site in enumerate(patch)}
    # Every physical outcome in a setting is shared by the simultaneously read out patches.
    outcomes = np.arange(2**len(active))
    zero_events = []
    for patch in layout.patches:
        mask = sum(1 << (len(active) - 1 - active.index(site)) for site in patch)
        zero_events.append((outcomes & mask) == 0)
    observations = np.empty((len(layout.patches), len(settings), len(settings)))
    for measurement_index, setting in enumerate(settings):
        rotated = evolved.reshape((2,) * n_sites + (len(settings),))
        for site in active:
            rows = np.moveaxis(rotated, site, 0)
            basis = MEASUREMENT_BASES[setting[position[site]]]
            rotated = np.moveaxis((basis @ rows.reshape(2, -1)).reshape(rows.shape), 0, site)
        rotated = rotated.transpose(active + separators + (n_sites,))
        amplitudes = rotated.reshape(len(outcomes), -1, len(settings))
        probabilities = np.sum(np.abs(amplitudes)**2, axis=1)
        probabilities /= probabilities.sum(axis=0)
        counts = np.array([rng.multinomial(shots, row) for row in probabilities.T]).T
        observations[:, measurement_index, :] = (np.array(zero_events) @ counts) / shots
    return observations


def finite_patch_data(evolved, layout, n_sites, settings, duration, shots, rng):
    """Recover exact-probability and sampled coefficients at finite kick count."""
    exact = []
    sampled = []
    sample = sampled_probabilities(evolved, layout, n_sites, settings, shots, rng) if shots else None
    for i, patch in enumerate(layout.patches):
        design = tomography_design(len(patch))
        free_sites = tuple(site for site in range(n_sites) if site not in patch)
        permuted = evolved.reshape((2,) * n_sites + (len(settings),)).transpose((n_sites,) + patch + free_sites)
        outputs = permuted.reshape(len(settings), design.states.shape[1], -1)
        probabilities = exact_probabilities(outputs, design.effects)
        num_local = len(design.states)
        # Smaller edge patches repeat their setting while the larger patches cycle through theirs.
        exact.append(recover_patch(design, probabilities.reshape(num_local, num_local, -1).mean(axis=2), duration))
        if sample is not None:
            shot_data = sample[i].reshape(num_local, len(settings) // num_local, num_local, -1)
            sampled.append(recover_patch(design, shot_data.mean(axis=(1, 3)), duration))
    return exact, sampled


def record_layout(patch_data, layout_id, estimates):
    """Index local coefficient estimates by layout, patch and Pauli label."""
    for i, patch in enumerate(estimates):
        patch_data.update({(layout_id, i, label): value for label, value in patch.items()})


def coefficient_error_metrics(differences):
    """Summarize coefficient errors using Euclidean and per-entry measures."""
    absolute = np.abs(differences)
    return {"l2": float(np.linalg.norm(differences)),
            "mean_abs": float(np.mean(absolute)), "max_abs": float(np.max(absolute)),
            "rmse": float(np.sqrt(np.mean(differences**2))), "count": len(differences)}


def error_summary(truth, estimates, group_labels):
    """Return overall and group-specific coefficient error metrics."""
    differences = estimates - truth
    group_array = np.asarray(group_labels)
    grouped = {group: coefficient_error_metrics(differences[group_array == group])
               for group in dict.fromkeys(group_labels)}
    return {**coefficient_error_metrics(differences), "by_group": grouped,
            "by_group_rmse": {group: metrics["rmse"] for group, metrics in grouped.items()}}


def run_instance(labels, group_labels, layouts, rules, seed, duration, kicks, display_kicks, shots, shot_seed):
    """Simulate one Hamiltonian, checking the ideal limit and finite-kick reconstructions."""
    n_sites = len(labels[0])
    truth = np.random.default_rng(seed).uniform(-1, 1, len(labels))
    hamiltonian = sum(c * kron_all(PAULI_BY_LABEL[p] for p in label) for label, c in zip(labels, truth))
    energies, vectors = eigh(hamiltonian)
    frames = [frame_signs(n_sites, layout) for layout in layouts]
    settings = [tuple(product(range(4), repeat=max(map(len, layout.patches)))) for layout in layouts]
    preparations = [global_preparations(layout, n_sites, rows) for layout, rows in zip(layouts, settings)]

    ideal_data = {}
    phase_radii = []
    for layout_id, layout in enumerate(layouts):
        estimates, radius = ideal_patch_data(hamiltonian, n_sites, layout, frames[layout_id], duration)
        record_layout(ideal_data, layout_id, estimates)
        phase_radii.append(radius)
    if max(phase_radii) >= 1:
        raise ValueError(f"Use a shorter time for seed {seed}; max T||H_patch|| = {max(phase_radii):.3f}")
    ideal, raw_ideal = assemble(ideal_data, rules)
    ideal_error = error_summary(truth, ideal, group_labels)
    if ideal_error["max_abs"] > 1e-9:
        raise RuntimeError(f"Ideal reconstruction failed for seed {seed}: {ideal_error['max_abs']:.3g}")

    sweep = []
    displayed = []
    shot_error = None
    raw_shot_error = None
    rng = np.random.default_rng(shot_seed)
    for r in kicks:
        free = unitary_from_eigensystem(energies, vectors, duration / r)
        exact_data = {}
        shot_data = {}
        for layout_id, layout in enumerate(layouts):
            evolution = four_frame_evolution(free, frames[layout_id], r // 4)
            run_shots = shots if r == display_kicks else 0
            exact, sampled = finite_patch_data(evolution @ preparations[layout_id], layout, n_sites,
                                               settings[layout_id], duration, run_shots, rng)
            record_layout(exact_data, layout_id, exact)
            record_layout(shot_data, layout_id, sampled)
        estimate, raw = assemble(exact_data, rules)
        sweep.append({"kicks": r, "corrected": error_summary(truth, estimate, group_labels),
                      "uncorrected": error_summary(truth, raw, group_labels)})
        if r == display_kicks:
            sampled, raw_sampled = assemble(shot_data, rules)
            shot_error = error_summary(truth, sampled, group_labels)
            raw_shot_error = error_summary(truth, raw_sampled, group_labels)
            for i, (label, group) in enumerate(zip(labels, group_labels)):
                displayed.append({"label": label, "group": group, "true": float(truth[i]),
                                  "estimate": float(sampled[i]), "exact_mean": float(estimate[i]),
                                  "uncorrected": float(raw[i]), "ideal_uncorrected": float(raw_ideal[i]),
                                  "sampled_uncorrected": float(raw_sampled[i])})
    return {"seed": seed, "shot_seed": shot_seed, "phase_radius": float(max(phase_radii)), "ideal": ideal_error,
            "ideal_uncorrected": error_summary(truth, raw_ideal, group_labels), "sweep": sweep,
            "sampled": shot_error, "sampled_uncorrected": raw_shot_error}, displayed, sum(len(rows)**2 for rows in settings)

N_SITES = 8
GROUP_NAMES = ("One-body", "Two-body", "Three-body")
SIDE_LENGTH = 3
SITE_COUNT = SIDE_LENGTH**2
AXES = "XYZ"

def chain_coefficients():
    """Label every Pauli term fitting within three consecutive chain sites."""
    groups = {}
    for weight, group in enumerate(GROUP_NAMES, start=1):
        for support in combinations(range(N_SITES), weight):
            if support[-1] - support[0] > 2:
                continue
            for factors in product("XYZ", repeat=weight):
                pauli = ["I"] * N_SITES
                for site, factor in zip(support, factors):
                    pauli[site] = factor
                groups["".join(pauli)] = group
    return groups

def chain_layout(shift):
    # Translate the infinite pattern, then clip it to the open chain.
    """Translate and clip the three-active, two-separator chain pattern."""
    phase = [(site - shift) % 5 for site in range(N_SITES)]
    patches = tuple(
        tuple(sites)
        for active, sites in groupby(range(N_SITES), key=lambda site: phase[site] < 3)
        if active
    )
    separators = tuple(
        tuple(site for site in range(N_SITES) if phase[site] == separator_phase)
        for separator_phase in (3, 4)
    )
    return Layout(patches, separators)

def site_index(row: int, col: int) -> int:
    """Map a grid coordinate to its row-major site index."""
    return SIDE_LENGTH * row + col

def lattice_coefficients() -> dict[str, str]:
    """Label all on-site and nearest-neighbor Pauli terms on the open grid."""
    groups = {}
    for row, col in product(range(SIDE_LENGTH), repeat=2):
        first = site_index(row, col)
        for axis in AXES:
            label = ["I"] * SITE_COUNT
            label[first] = axis
            groups["".join(label)] = "On-site"

        neighbors = (
            (row, col + 1, "Horizontal"),
            (row + 1, col, "Vertical"),
        )
        for next_row, next_col, group in neighbors:
            if next_row >= SIDE_LENGTH or next_col >= SIDE_LENGTH:
                continue
            second = site_index(next_row, next_col)
            for first_axis, second_axis in product(AXES, repeat=2):
                label = ["I"] * SITE_COUNT
                label[first] = first_axis
                label[second] = second_axis
                groups["".join(label)] = group
    return groups

def lattice_layouts() -> tuple[Layout, ...]:
    """Return all twelve schedules, including coincident open-boundary layouts."""
    layouts = []
    for rows_active, line_parity, offset in product((True, False), range(2), range(3)):
        patches = []
        for line in range(line_parity, SIDE_LENGTH, 2):
            # Each period begins with an active dimer. Clip dimers at open edges.
            for start in range(offset - 3, SIDE_LENGTH, 3):
                along_line = range(max(0, start), min(SIDE_LENGTH, start + 2))
                if along_line:
                    patch = tuple(
                        site_index(line, position) if rows_active else site_index(position, line)
                        for position in along_line
                    )
                    patches.append(patch)

        active = {site for patch in patches for site in patch}
        colors = ([], [])
        for row, col in product(range(SIDE_LENGTH), repeat=2):
            site = site_index(row, col)
            if site not in active:
                colors[(row + col) % 2].append(site)
        layouts.append(Layout(tuple(sorted(patches)), (tuple(colors[0]), tuple(colors[1]))))
    return tuple(layouts)

def run_study(name, groups, layouts, *, seeds=(43,), duration=0.01,
              kicks=(12,), display_kicks=12, copies=10_000_000_000,
              shot_seed=20261118):
    """Run the selected seeds and kick counts with a total-copy budget."""
    if not isinstance(copies, int) or isinstance(copies, bool) or copies <= 0:
        raise ValueError("copies must be a positive integer.")
    if not seeds or any(not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError("Use nonnegative integer Hamiltonian seeds.")
    if not isinstance(shot_seed, int) or shot_seed < 0:
        raise ValueError("shot_seed must be a nonnegative integer.")
    if any(not isinstance(r, int) for r in (*kicks, display_kicks)):
        raise ValueError("Use integer kick counts.")
    args = SimpleNamespace(seeds=list(seeds), duration=duration, kicks=list(kicks),
                           display_kicks=display_kicks, copies=copies, shots=1,
                           shot_seed=shot_seed)
    if args.copies is not None and args.copies <= 0:
        raise ValueError("--copies must be a positive integer.")
    if (not np.isfinite(args.duration) or args.duration <= 0 or (args.copies is None and args.shots <= 0)
            or any(seed < 0 for seed in (*args.seeds, args.shot_seed))):
        raise ValueError("Use a positive finite duration, positive shots and nonnegative seeds.")
    kicks = sorted(set(args.kicks))
    if args.display_kicks not in kicks or any(r <= 0 or r % 4 for r in kicks):
        raise ValueError("Use positive multiples of four and include the display kick count in --kicks.")

    labels = tuple(groups)
    rules = assembly_rules(labels, layouts)
    layout_metadata = []
    for index, layout in enumerate(layouts):
        patch_sizes = [len(patch) for patch in layout.patches]
        max_size = max(patch_sizes)
        layout_metadata.append({"index": index, "patches": layout.patches, "separators": layout.separators,
                                "patch_sizes": patch_sizes, "max_patch_size": max_size,
                                "settings": 4**(2 * max_size)})
    total_settings = sum(layout["settings"] for layout in layout_metadata)
    shots = args.shots if args.copies is None else (args.copies + total_settings - 1) // total_settings
    shot_seeds = [args.shot_seed + index for index in range(len(args.seeds))]
    result = {"name": name, "n_sites": len(labels[0]), "duration": args.duration,
              "display_kicks": args.display_kicks, "shots_per_setting": shots,
              "settings_per_instance": total_settings, "requested_copies": args.copies,
              "total_copies": total_settings * shots, "seeds": args.seeds, "shot_seeds": shot_seeds,
              "shot_seed": args.shot_seed, "layout_count": len(layouts),
              "distinct_layout_count": len(set(layouts)), "layouts": layout_metadata,
              "max_patch_size": max(layout["max_patch_size"] for layout in layout_metadata), "instances": []}
    print(f"## {name}", flush=True)
    print(f"\n{len(labels)} coefficients, {len(layouts)} schedules, T = {args.duration:g}\n", flush=True)
    print("| Seed | Ideal max error | Exact-probability RMSE at largest r | Sampled RMSE at display r |", flush=True)
    print("|---:|---:|---:|---:|", flush=True)
    for index, seed in enumerate(args.seeds):
        metrics, estimates, settings_count = run_instance(labels, tuple(groups.values()), layouts, rules, seed,
                                                          args.duration, kicks, args.display_kicks, shots,
                                                          shot_seeds[index])
        if settings_count != total_settings:
            raise RuntimeError("The physical tomography settings differ from the recorded layout settings.")
        result["instances"].append(metrics)
        print(f"| {seed} | {metrics['ideal']['max_abs']:.3e} | "
              f"{metrics['sweep'][-1]['corrected']['rmse']:.3e} | {metrics['sampled']['rmse']:.3e} |", flush=True)
        if index == 0:
            result["coefficients"] = estimates
    result["max_phase_radius"] = max(instance["phase_radius"] for instance in result["instances"])
    return result
