"""Restart-safe compatibility checks for OpenJij calculation and lead caches.

Electrode self energies are reused ONLY after numerical H/S and structural
comparison with a snapshot made when the TBTGF was generated. No blind reuse
on a SHA-256 mismatch. A legacy TBTGF without a snapshot is regenerated.

Result config fingerprints are informational: reference/vertex matrices are
rebuilt for the current run, and the separate Static cache checks their values.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from scipy import sparse

# Values in SIESTA units (Hamiltonian eV, geometry Angstrom).
LEAD_H_ATOL = 1e-7
LEAD_S_ATOL = 1e-9
LEAD_XYZ_ATOL = 1e-7

# These SHA values are sensitive to sparse representation, floating-point
# arithmetic and device indices. They should not determine result-cache reuse.
_VOLATILE_RESULT_FINGERPRINTS = frozenset({
    "refs_sha256", "delta_i_sha256", "delta_j_sha256",
    "hydrogen_i_sha256", "hydrogen_j_sha256",
})
_LEAD_FINGERPRINTS = frozenset({"electrode_left_sha256", "electrode_right_sha256"})


def _write_json(path, content):
    path = Path(path)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        tmp.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _write_npz(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with open(tmp, "wb") as stream:
            np.savez_compressed(stream, **data)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _diff_keys(previous, current, ignore=()):
    return sorted(k for k in (previous.keys() | current.keys())
                  if k not in ignore and previous.get(k) != current.get(k))


def update_result_config(path, current):
    """Allow regenerated current inputs without rejecting bitwise hash changes.

    Unchanged science/numerical settings (energies, prefactor, orbitals, etc.)
    are still required. The current fingerprints are written for traceability.
    The results are recomputed each run, not loaded from this config.
    """
    path = Path(path)
    if path.exists():
        previous = json.loads(path.read_text())
        diff = _diff_keys(previous, current, _VOLATILE_RESULT_FINGERPRINTS)
        if diff:
            raise ValueError(f"Incompatible result configuration: {path}. "
                             f"Changed non-fingerprint parameters: {diff}. "
                             "Choose a new results_name or archive the previous run.")
    _write_json(path, current)


def _csr_data(prefix, matrix, output):
    m = sparse.csr_matrix(matrix, dtype=np.complex128)
    m.sum_duplicates()
    m.eliminate_zeros()
    m.sort_indices()
    output[f"{prefix}_shape"] = np.asarray(m.shape, dtype=np.int64)
    output[f"{prefix}_data"] = np.asarray(m.data)
    output[f"{prefix}_indices"] = np.asarray(m.indices, dtype=np.int64)
    output[f"{prefix}_indptr"] = np.asarray(m.indptr, dtype=np.int64)


def _unpack_csr(prefix, payload):
    return sparse.csr_matrix((payload[f"{prefix}_data"],
                              payload[f"{prefix}_indices"],
                              payload[f"{prefix}_indptr"]),
                             shape=tuple(payload[f"{prefix}_shape"]))


def _lead_snapshot(left, right):
    snapshot = {"snapshot_version": np.asarray(1, dtype=np.int64)}
    for name, H in (("left", left), ("right", right)):
        g = H.geometry
        snapshot[f"{name}_nsc"] = np.asarray(H.nsc, dtype=np.int64)
        snapshot[f"{name}_cell"] = np.asarray(H.cell, dtype=np.float64)
        snapshot[f"{name}_xyz"] = np.asarray(g.xyz, dtype=np.float64)
        snapshot[f"{name}_Z"] = np.asarray(g.atoms.Z, dtype=np.int64)
        snapshot[f"{name}_firsto"] = np.asarray(g.firsto, dtype=np.int64)
        snapshot[f"{name}_polarized"] = np.asarray(int(H.spin.is_polarized))
        for ik, k in enumerate(((0., 0., 0.), (.137, .271, .193))):
            for spin in ((0, 1) if H.spin.is_polarized else (0,)):
                M = H.Hk(k=k, spin=spin, format="csr") if H.spin.is_polarized \
                    else H.Hk(k=k, format="csr")
                _csr_data(f"{name}_k{ik}_spin{spin}_H", M, snapshot)
            _csr_data(f"{name}_k{ik}_S", H.Sk(k=k, format="csr"), snapshot)
    return snapshot


def _max_sparse_difference(a, b):
    d = (a - b).tocsr()
    d.eliminate_zeros()
    return float(np.max(np.abs(d.data))) if d.nnz else 0.0


def _compare_snapshots(previous, current):
    if previous.keys() != current.keys():
        return "electrode snapshot structure / spin channels changed"
    for name in ("left", "right"):
        for field in ("nsc", "Z", "firsto", "polarized"):
            key = f"{name}_{field}"
            if not np.array_equal(previous[key], current[key]):
                return f"{key} changed"
        for field in ("cell", "xyz"):
            key = f"{name}_{field}"
            a, b = previous[key], current[key]
            if a.shape != b.shape or not np.allclose(a, b, rtol=0, atol=LEAD_XYZ_ATOL):
                return f"{key} changed (beyond {LEAD_XYZ_ATOL:g} Angstrom)"
        for ik in range(2):
            for spin in ((0, 1) if current[f"{name}_polarized"].item() else (0,)):
                key = f"{name}_k{ik}_spin{spin}_H"
                diff = _max_sparse_difference(_unpack_csr(key, previous),
                                              _unpack_csr(key, current))
                if diff > LEAD_H_ATOL:
                    return f"{key} differs by {diff:.3g} eV"
            key = f"{name}_k{ik}_S"
            diff = _max_sparse_difference(_unpack_csr(key, previous),
                                          _unpack_csr(key, current))
            if diff > LEAD_S_ATOL:
                return f"{key} differs by {diff:.3g}"
    return None


def _read_snapshot(path):
    with np.load(path, allow_pickle=False) as stream:
        return {key: stream[key].copy() for key in stream.files}


def _lead_compatibility(folder, left, right, expected):
    folder = Path(folder)
    meta_file = folder / "config.json"
    snap_file = folder / "electrode_reference.npz"
    if not meta_file.is_file() or not snap_file.is_file():
        return "legacy or incomplete electrode cache (no numerical reference)"
    previous_meta = json.loads(meta_file.read_text())
    changed = _diff_keys(previous_meta, expected, _LEAD_FINGERPRINTS)
    if changed:
        return f"electrode calculation settings changed: {changed}"
    try:
        reference = _read_snapshot(snap_file)
        current = _lead_snapshot(left, right)
        return _compare_snapshots(reference, current)
    except Exception as exc:
        return f"cannot validate numerical electrode reference: {type(exc).__name__}: {exc}"


def prepare_lead_cache(folder, left, right, expected):
    """Called by rank 0 BEFORE the TBTGF 'already complete' shortcut.

    Existing valid TBTGFs are reused with no warning on SHA drift; otherwise
    all dependent self energies and scratch files are invalidated and rebuilt.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    has_existing = any((folder / f).exists() for f in
                       ("config.json", "electrode_reference.npz", "Left.TBTGF", "Right.TBTGF"))
    reason = _lead_compatibility(folder, left, right, expected) if has_existing else None
    if reason:
        print(f"[electrodes] {reason}; rebuilding cached self-energies", flush=True)
        for filename in ("Left.TBTGF", "Right.TBTGF", "Left.TBTGF.tmp",
                         "Right.TBTGF.tmp", "config.json", "electrode_reference.npz"):
            (folder / filename).unlink(missing_ok=True)
        scratch = folder / "TBTGF_tmp"
        if scratch.is_dir():
            for item in scratch.glob("*.npy"):
                item.unlink()
    # Safe to refresh metadata hashes after numerical equivalence confirmed.
    _write_json(folder / "config.json", expected)
    if not (folder / "electrode_reference.npz").is_file():
        _write_npz(folder / "electrode_reference.npz", _lead_snapshot(left, right))


def check_lead_cache(folder, left, right, expected):
    """Called on each rank when READING TBTGF: fail closed if incompatible."""
    reason = _lead_compatibility(folder, left, right, expected)
    if reason:
        raise ValueError(f"Incompatible TBTGF cache at {folder}: {reason}. "
                         "Rerun with self_energy_mode='compute+write' to regenerate.")
