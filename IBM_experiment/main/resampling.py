"""Full-register resampling and numerical fits for the hardware analysis."""

import base64
import io
import json
from pathlib import Path
import zlib
import numpy as np
from scipy.linalg import logm
from scipy.optimize import minimize
from .tomography import Pauli_Transfer_Matrix


def read_results(path):
    """Read the recorded shot arrays without connecting to IBM Runtime."""
    saved = json.loads(Path(path).read_text(encoding="utf-8"))
    results = []
    for pub in saved["__value__"]["pub_results"]:
        bit = pub["__value__"]["data"]["__value__"]["fields"]["c"]["__value__"]
        array = np.load(io.BytesIO(zlib.decompress(base64.b64decode(bit["array"]["__value__"]))), allow_pickle=False)
        results.append({"array": array, "num_bits": bit["num_bits"]})
    return results


def prepare_resampling(results, learner):
    """Prepare patch outcomes and linear inversion for the recorded settings."""
    if len(results) != 972 or any(len(row["array"]) != 900 for row in results):
        raise ValueError("Expected 972 measurement settings with 900 shots each.")
    groups = learner.groups_qubits_tomo()
    packed = []
    for index, group in enumerate(groups):
        p = len(group)
        q = np.arange(2*p)
        array = np.empty((324, 900, p), np.uint8)
        for setting in range(324):
            raw = results[index*324+setting]["array"]
            bits = (raw[:, -1-q//8] >> (q%8)) & 1
            array[setting] = bits[:, 0::2] | (bits[:, 1::2] << 1)
        packed.append(array)
    states = np.array(learner._states)
    measures = np.array(learner._measures)
    A = np.linalg.pinv(measures.reshape(36, 16).conj())
    R = states.reshape(36, 16).T
    B = np.linalg.pinv(R.T.conj()).T.conj()
    probs = {"learner": learner, "packed": packed, "pairs": np.concatenate(groups), "A": A, "B": B}
    probs["raw_choi"] = construct(probs)
    return probs


def construct(probs, draws=None):
    """Build raw patch Choi matrices, sharing each register draw across patches."""
    matrices = []
    for index, array in enumerate(probs["packed"]):
        probabilities = np.empty((array.shape[2], 324, 4))
        for setting in range(324):
            sample = array[setting] if draws is None else array[setting, draws[index][setting]]
            probabilities[:, setting, :] = (sample[:, :, None] == np.arange(4)).mean(axis=0)
        p = probabilities.reshape(array.shape[2], 36, 9, 4).transpose(0, 2, 3, 1).reshape(array.shape[2], 36, 36)
        process = np.einsum('ab,nbc,cd->nad', probs["A"], p, probs["B"], optimize=True)
        matrices.extend(process.reshape(-1, 4, 4, 4, 4).transpose(0, 1, 3, 2, 4).reshape(-1, 16, 16))
    return np.asarray(matrices)


def reference_scores(probs, initial_fits):
    """Return raw overlap scores for the original-count fits."""
    return np.array([float(np.vdot(U.ravel(), J @ U.ravel()).real / 4)
                     for U, J in zip(initial_fits, probs["raw_choi"])])


def resample_patches(probs, rng, initial_fits, U, rep):
    """Fit one replica with the paper's fixed starts, checks and retry rule."""
    L = probs["learner"]
    draws = [rng.integers(0, 900, size=(324, 900)) for _ in range(3)]
    matrices = construct(probs, draws)
    init = np.r_[U.real.ravel(), U.imag.ravel()]
    Hs, scores = [], []
    diagnostics = {"statuses": [], "objective_checks": [], "phase_checks": [], "branch_diagnostics": []}
    for pi, J in enumerate(matrices):
        def fun(x):
            a = (x[:16]+1j*x[16:]).reshape(4, 4)
            v = L.complex2unitary(a).ravel()
            return -float(np.vdot(v, J @ v).real)
        wi = np.r_[initial_fits[pi].real.ravel(), initial_fits[pi].imag.ravel()]
        candidates = [minimize(fun, wi)]
        checked = (pi+rep)%20 == 0
        if checked or not candidates[0].success:
            candidates.append(minimize(fun, init))
        if all(not candidate.success for candidate in candidates):
            for seed in range(2):
                rg = np.random.default_rng(900000+1000*rep+10*pi+seed)
                perturb = initial_fits[pi]+.2*(rg.normal(size=(4, 4))+1j*rg.normal(size=(4, 4)))
                candidates.append(minimize(fun, np.r_[perturb.real.ravel(), perturb.imag.ravel()]))
        res = min(candidates, key=lambda candidate: candidate.fun)
        if not np.isfinite(res.x).all() or not np.isfinite(res.fun):
            raise ValueError("A resampled unitary fit returned non-finite values.")
        if checked:
            v0, v1 = -float(candidates[0].fun)/4, -float(candidates[1].fun)/4
            diagnostics["objective_checks"].append([rep, pi, v0, v1, abs(v0-v1), bool(candidates[0].success), bool(candidates[1].success)])
        diagnostics["statuses"].append([rep, pi, bool(res.success), len(candidates), float(res.fun)])
        a = (res.x[:16]+1j*res.x[16:]).reshape(4, 4)
        unitary = L.complex2unitary(a)
        aligned = unitary*np.exp(-1j*np.angle(np.trace(unitary)))
        diagnostics["branch_diagnostics"].append([rep, pi, float(abs(np.trace(unitary))), float(np.max(np.abs(np.angle(np.linalg.eigvals(aligned)))))])
        Hs.append(1j*logm(aligned))
        scores.append(-float(res.fun)/4)
        if checked:
            C = Pauli_Transfer_Matrix(Hs[-1], num_qubits=2)
            C0 = Pauli_Transfer_Matrix(1j*logm(unitary), num_qubits=2)
            diff = (C-C0).ravel()
            diff[0] = 0
            diagnostics["phase_checks"].append([rep, pi, float(np.linalg.norm(diff))])
    return Hs, scores, diagnostics
