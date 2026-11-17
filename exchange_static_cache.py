"""Disk cache for local exchange vertices and H-adatom downfolding blocks.

Designed for Exchange_Workflow.run_exchange and its existing Device / DysonSystem.
Only rank 0 writes. Other MPI ranks read the completed cache in 'read' mode.

Files live inside each Results_.../Static directory (not folder/Deltas).
Numerically compatible static matrices are reused. In auto mode, incompatible
matrices are regenerated from the CURRENT built DFT Hamiltonians rather than
blocking restart; in read mode incompatibilities fail (MPI safety).

The cached hydrogen matrices are *static* H_CH, H_HC, H_HH, S_CH, S_HC,
S_HH, plus the carbon->device orbital map. Sigma_H(z) remains evaluated
at every contour energy by the original calculation routines.

The parent builder still constructs the reference Hamiltonian and derives
its vertices. Cached matrices are checked against those currently computed
before they are used, preventing silent reuse after input changes.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from scipy import sparse
from exchange_mpi_io import load_local_delta, save_local_delta

SCHEMA = 2
_ATOL = 1.0e-7  # eV, local Hamiltonian/vertex compatibility
_S_ATOL = 1.0e-9  # dimensionless overlap compatibility


def _atomic_npz(path, **values):
    path = Path(path)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with open(tmp, "wb") as stream:
            np.savez_compressed(stream, **values)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _atomic_json(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _assert_array_matches(path, field, saved, current):
    saved, current = np.asarray(saved), np.asarray(current)
    if saved.shape != current.shape:
        raise ValueError(f"{path}: {field} shape changed: {saved.shape} != {current.shape}")
    if (np.issubdtype(saved.dtype, np.integer) or saved.dtype.kind == 'b'):
        if not np.array_equal(saved, current):
            raise ValueError(f"{path}: {field} changed (integer orbital mapping)")
    elif np.issubdtype(saved.dtype, np.number) and np.issubdtype(current.dtype, np.number):
        tol = _S_ATOL if field.startswith('S_') else _ATOL
        if not np.allclose(saved, current, rtol=1e-10, atol=tol, equal_nan=False):
            delta = np.max(np.abs(saved - current)) if saved.size else 0.0
            raise ValueError(f"{path}: cached {field} differs by {delta:.3g}")
    elif not np.array_equal(saved, current):
        raise ValueError(f"{path}: cached {field} changed; remove the cache or choose "
                         "another results_name")


def _delta_pair(obj, method):
    """Return selected static vertices and how to assign loaded versions."""
    if hasattr(obj, "vertex_own") and hasattr(obj, "vertex_total"):
        is_total = (method == "liechtenstein_sup"
                    and obj.info.get("vertices", "own") == "region")
        slot = "vertex_total" if is_total else "vertex_own"
        D = [sparse.csr_matrix(d) for d in getattr(obj, slot)]
        o = [np.asarray(x, dtype=np.int64) for x in obj.orbs]

        def assign(matrices, indices):
            setattr(obj, slot, [d.toarray() for d in matrices])
            # loc/R already used by Dyson propagators; indices must match.
        return D, o, assign

    D, o = obj.vertices(method)
    D = [sparse.csr_matrix(d) for d in D]
    o = [np.asarray(x, dtype=np.int64) for x in o]

    def assign(matrices, indices):
        obj.Deltas = matrices
        obj.orbs = indices
        if method in obj.vertex_sets:
            obj.vertex_sets[method] = (matrices, indices)
    return D, o, assign


def _load_or_write_delta(path, matrix, orbitals, mode):
    matrix = sparse.csr_matrix(matrix, dtype=np.complex128)
    orbitals = np.asarray(orbitals, dtype=np.int64)
    if path.is_file():
        try:
            saved, saved_orbs = load_local_delta(path)
            if not np.array_equal(saved_orbs, orbitals):
                raise ValueError("orbital indices changed")
            if saved.shape != matrix.shape:
                raise ValueError("delta shape changed")
            difference = (sparse.csr_matrix(saved) - matrix).tocsr()
            if difference.nnz and np.max(np.abs(difference.data)) > _ATOL:
                raise ValueError(f"delta changed by {np.max(np.abs(difference.data)):.3g} eV")
            return sparse.csr_matrix(saved), np.asarray(saved_orbs, dtype=np.int64)
        except (ValueError, OSError, KeyError) as exc:
            if mode == "read":
                raise ValueError(f"Incompatible local cache {path}: {exc}") from exc
            print(f"[static cache] refreshing {path.name}: {exc}", flush=True)
    elif mode == "read":
        raise FileNotFoundError(f"Missing static delta for MPI reader: {path}")
    save_local_delta(path, matrix, orbitals)
    return sparse.csr_matrix(matrix), orbitals


def _pristine_h_payload(data, mapped):
    values = {
        "kind": np.array("pristine_h"),
        "basis": np.array(str(data.get("basis", ""))),
        "c_orbs": np.asarray(data["c_orbs"], dtype=np.int64),
        "h_orbs": np.asarray(data["h_orbs"], dtype=np.int64),
        "mapped_c_orbs": np.asarray(mapped, dtype=np.int64),
    }
    for key in ("S_CH", "S_HC", "S_HH", "S_CC"):
        if key in data:
            values[key] = np.asarray(data[key])
    for spin, letter in ((0, "up"), (1, "dn")):
        for key in ("H_CH", "H_HC", "H_HH", "H_CC"):
            if key in data[spin]:
                values[f"{letter}_{key}"] = np.asarray(data[spin][key])
    return values


def _load_pristine_h(payload):
    data = {
        "basis": str(payload["basis"].item()),
        "c_orbs": payload["c_orbs"].copy(),
        "h_orbs": payload["h_orbs"].copy(),
    }
    for key in ("S_CH", "S_HC", "S_HH", "S_CC"):
        if key in payload:
            data[key] = payload[key].copy()
    for spin, letter in ((0, "up"), (1, "dn")):
        data[spin] = {}
        for key in ("H_CH", "H_HC", "H_HH", "H_CC"):
            k = f"{letter}_{key}"
            if k in payload:
                data[spin][key] = payload[k].copy()
    return data, payload["mapped_c_orbs"].copy()


def _dyson_h_payload(ad):
    values = {
        "kind": np.array("dyson_h"),
        "owner": np.array(ad.owner),
        "c_loc": np.asarray(ad.c_loc, dtype=np.int64),
    }
    for key in ("S_CH", "S_HC", "S_HH"):
        values[key] = np.asarray(getattr(ad, key))
    for key in ("H_CH", "H_HC", "H_HH"):
        values[f"up_{key}"] = np.asarray(getattr(ad, key)[0])
        values[f"dn_{key}"] = np.asarray(getattr(ad, key)[1])
    return values


def _load_or_write_h(path, current, mode):
    if path.is_file():
        try:
            with np.load(path, allow_pickle=False) as f:
                saved = {key: f[key].copy() for key in f.files}
            if saved.keys() != current.keys():
                raise ValueError("saved static-block schema changed")
            for name, value in current.items():
                _assert_array_matches(path, name, saved[name], value)
            return saved
        except (ValueError, OSError, KeyError) as exc:
            if mode == "read":
                raise ValueError(f"Incompatible H/static cache {path}: {exc}") from exc
            print(f"[static cache] refreshing {path.name}: {exc}", flush=True)
    elif mode == "read":
        raise FileNotFoundError(f"Missing static H blocks for MPI reader: {path}")
    _atomic_npz(path, **current)
    return current


def cache_exchange_static(obj, method, folder, results_name, *, basis,
                          defect_i, defect_j, n_dist, downfold, region, radius,
                          vertices, N_imp, N_buffer, axis, mode="auto", save_deltas=True):
    """Load cached static vertices and H blocks (or write them if absent).

    Parameters
    ----------
    mode : {'auto', 'read'}
        'auto' is rank 0 only. It creates files when absent, otherwise loads
        and validates. 'read' is for the other MPI ranks after rank 0 completed.

    Warning: the builder still reads the DFT inputs to create the device and
    cross-check caches. This is a validated cache, not a skip-build shortcut.
    """
    if mode not in ("auto", "read"):
        raise ValueError("mode must be 'auto' or 'read'")
    path = Path(folder) / str(results_name) / "Static"
    if mode == "auto":
        path.mkdir(parents=True, exist_ok=True)
    info = {
        "schema": SCHEMA, "method": str(method), "basis": str(basis),
        "defect_i": str(defect_i), "defect_j": str(defect_j),
        "n_dist": int(n_dist), "downfold": bool(downfold),
        "region": str(region), "radius": float(radius),
        "vertices": str(vertices),
        "N_imp": None if N_imp is None else int(N_imp),
        "N_buffer": int(N_buffer), "axis": int(axis),
        "save_deltas": bool(save_deltas),
    }
    config_path = path / "static_config.json"
    if config_path.is_file():
        previous = json.loads(config_path.read_text())
        if previous != info:
            if mode == "read":
                raise ValueError(f"{config_path}: static cache settings differ")
            print("[static cache] settings changed; refreshing Static/", flush=True)
            for old in path.glob('*.npz'):
                old.unlink()
            _atomic_json(config_path, info)
    elif mode == "read":
        raise FileNotFoundError(f"Missing static cache manifest: {config_path}")

    # Cache two compact local Delta matrices only when requested.
    loaded = None
    if save_deltas:
        matrices, indices, assign = _delta_pair(obj, method)
        if len(matrices) != 2 or len(indices) != 2:
            raise ValueError("Two vertices (i, j) are required")
        loaded = []
        for k, (D, orbitals) in enumerate(zip(matrices, indices)):
            filename = path / f"Delta_{'ij'[k]}.npz"
            loaded.append(_load_or_write_delta(filename, D, orbitals, mode))

    # For Dyson's G0-dressed reference, retain the static scattering matrices
    # used to assemble U(z), in addition to the spin vertices. This allows the
    # same static Dyson reference to be inspected or restored from disk.
    dyson_static = None
    if hasattr(obj, "dH_avg") and hasattr(obj, "dH_diff"):
        dyson_current = {
            "kind": np.array("dyson_reference"),
            "R": np.asarray(obj.R, dtype=np.int64),
            "dH_avg": np.asarray(obj.dH_avg),
            "dH_diff": np.asarray(obj.dH_diff),
            "dS_present": np.array(obj.dS is not None),
            "dS": np.asarray(obj.dS) if obj.dS is not None else np.empty((0, 0)),
        }
        dyson_static = _load_or_write_h(
            path / "Dyson_reference_static.npz", dyson_current, mode)

    # Keep separate unmodified static blocks; sigma_H(E) is NEVER cached at
    # just one energy, because it must be evaluated at each contour point.
    updates = []
    if downfold and getattr(obj, "downfold_kwargs", None):
        kw = obj.downfold_kwargs
        for letter in "ij":
            d = kw[f"hydrogen_{letter}"]
            mapped = kw[f"mapped_c_orbs_{letter}"]
            source = _pristine_h_payload(d, mapped)
            filename = path / f"Hydrogen_{letter}.npz"
            saved = _load_or_write_h(filename, source, mode)
            data, mapped = _load_pristine_h(saved)
            updates.append((letter, data, mapped))
    elif downfold and hasattr(obj, "adatoms"):
        for ia, ad in enumerate(obj.adatoms):
            letter = "ij"[ad.owner]
            filename = path / f"Hydrogen_{letter}_{ia}.npz"
            source = _dyson_h_payload(ad)
            saved = _load_or_write_h(filename, source, mode)
            updates.append((ad, saved))
    elif downfold:
        raise ValueError(f"downfold=True but {type(obj).__name__} lacks H blocks")

    # Changes are only applied after *all* cached files have passed checks.
    if loaded is not None:
        assign([x[0] for x in loaded], [x[1] for x in loaded])
    if dyson_static is not None:
        obj.dH_avg = dyson_static["dH_avg"].copy()
        obj.dH_diff = dyson_static["dH_diff"].copy()
        obj.dS = (dyson_static["dS"].copy()
                  if bool(dyson_static["dS_present"].item()) else None)
    if getattr(obj, "downfold_kwargs", None) and downfold:
        for letter, data, mapped in updates:
            obj.downfold_kwargs[f"hydrogen_{letter}"] = data
            obj.downfold_kwargs[f"mapped_c_orbs_{letter}"] = mapped
    elif downfold and hasattr(obj, "adatoms"):
        for ad, payload in updates:
            ad.owner = int(payload["owner"].item())
            ad.c_loc = payload["c_loc"].copy()
            for k in ("S_CH", "S_HC", "S_HH"):
                setattr(ad, k, payload[k].copy())
            for k in ("H_CH", "H_HC", "H_HH"):
                setattr(ad, k, [payload[f"up_{k}"].copy(), payload[f"dn_{k}"].copy()])

    if mode == "auto" and not config_path.exists():
        _atomic_json(config_path, info)
    return path
