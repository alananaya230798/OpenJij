"""Load-first cache of the built device / Dyson system for Exchange_Workflow.run_exchange.

Building a device is the slow, file-heavy part before the contour integral:
reading the SIESTA files, pruning, energy alignment, mapping, vertex (Delta)
extraction and the downfolding blocks. This cache stores the BUILT object
(Device or DysonSystem) together with the pruned pristine Hamiltonian, and on
the next run loads it INSTEAD of building:

The key contains every setting that changes the build, the fingerprints of the
input files (path, size, modification time) and the hash of the builder source
code. Re-running SIESTA, changing a setting or editing a builder module therefore
rebuilds automatically. If an input file is missing (e.g. results copied to
another machine) the cached build is used as it is, with a notice.

Only rank 0 reads or writes the cache; the other MPI ranks receive the object
by broadcast. The local Delta matrices are also written as Delta_i.npz /
Delta_j.npz (exchange_mpi_io.save_local_delta) for inspection.

The electrode self-energies have their own cache (TBTGF, exchange_cache_policy).
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import sparse

SCHEMA = 4
_SOURCES = ("Device_Builder", "Dyson_Exchange", "defects_dependencies", "J_dependencies", "regions")


def _atomic_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _atomic_json(path: Path, obj) -> None:
    _atomic_bytes(path, (json.dumps(obj, indent=2, sort_keys=True) + "\n").encode())


def file_fingerprint(path) -> dict:
    """Path, size and modification time of an input file (None values if missing)."""
    p = Path(path)
    try:
        st = p.stat()
        return {"path": str(p.resolve()), "size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}
    except OSError:
        return {"path": str(p.resolve()), "size": None, "mtime_ns": None}


def source_fingerprint() -> str:
    """SHA-256 of the source code of the modules that build the devices."""
    import importlib
    h = hashlib.sha256()
    for name in _SOURCES:
        mod = importlib.import_module(name)
        h.update(Path(mod.__file__).read_bytes())
    return h.hexdigest()


def build_key(settings: dict, input_files) -> dict:
    """Cache key: build settings, input-file fingerprints, builder source hash."""
    return {"schema": SCHEMA, "settings": json.loads(json.dumps(settings, default=str)),
            "files": [file_fingerprint(f) for f in input_files],
            "sources": source_fingerprint()}


def cache_dir(folder, results_name) -> Path:
    return Path(folder) / str(results_name) / "Static"


def _difference(previous: dict, key: dict) -> Optional[str]:
    """Why a stored key does not match (None: it matches, possibly with missing inputs)."""
    if previous.get("schema") != key["schema"]:
        return "cache format changed"
    if previous.get("sources") != key["sources"]:
        return "builder source code changed"
    changed = sorted(k for k in set(previous["settings"]) | set(key["settings"])
                     if previous["settings"].get(k) != key["settings"].get(k))
    if changed:
        return f"settings changed: {changed}"
    old = {f["path"]: f for f in previous["files"]}
    for f in key["files"]:
        o = old.get(f["path"])
        if o is None:
            return f"new input file {f['path']}"
        if f["size"] is None:          # input missing here: keep the cached build
            continue
        if (o["size"], o["mtime_ns"]) != (f["size"], f["mtime_ns"]):
            return f"input file changed: {f['path']}"
    return None


def load_build(path, key, verbose: bool = True) -> Optional[dict]:
    """The cached build if its key matches, else None (with the reason printed)."""
    path = Path(path)
    say = print if verbose else (lambda *a, **k: None)
    if not (path / "build_key.json").is_file() or not (path / "build.pkl").is_file():
        say(f"[build cache] nothing cached in {path}; building", flush=True)
        return None
    try:
        reason = _difference(json.loads((path / "build_key.json").read_text()), key)
    except Exception as exc:
        reason = f"unreadable key ({type(exc).__name__})"
    if reason:
        say(f"[build cache] {reason}; rebuilding", flush=True)
        return None
    try:
        with open(path / "build.pkl", "rb") as f:
            payload = pickle.load(f)
    except Exception as exc:
        say(f"[build cache] cannot load build.pkl ({type(exc).__name__}: {exc}); rebuilding", flush=True)
        return None
    missing = [f["path"] for f in key["files"] if f["size"] is None]
    if missing:
        say(f"[build cache] input files not found, using the cached build as it is: {missing}", flush=True)
    say(f"[build cache] loaded the built device from {path} (no DFT file read)", flush=True)
    return payload


def _vertices(obj, method):
    """Static vertices (sparse) and their device orbitals, for the Delta files."""
    if hasattr(obj, "vertex_own"):                       # DysonSystem
        total = method == "liechtenstein_sup" and obj.info.get("vertices", "own") == "region"
        Ds = obj.vertex_total if total else obj.vertex_own
        return [sparse.csr_matrix(D) for D in Ds], [np.asarray(o, dtype=np.int64) for o in obj.orbs]
    Ds, os_ = obj.vertices(method)
    return [sparse.csr_matrix(D) for D in Ds], [np.asarray(o, dtype=np.int64) for o in os_]


def save_build(path, key, payload: dict, method: str, save_deltas: bool = True) -> Path:
    """Write the build (pickle) and its key atomically; optionally the Delta files."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    _atomic_bytes(path / "build.pkl", pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    if save_deltas:
        from exchange_mpi_io import save_local_delta
        Ds, os_ = _vertices(payload["obj"], method)
        for k, (D, o) in enumerate(zip(Ds, os_)):
            save_local_delta(path / f"Delta_{'ij'[k]}.npz", D, o)
    _atomic_json(path / "build_key.json", key)          # last: marks the cache complete
    return path
