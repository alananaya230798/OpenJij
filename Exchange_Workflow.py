# -*- coding: utf-8 -*-
"""One-call exchange workflow and general visualizations.

    run = run_exchange(method="rkky", basis="pz", n_dist=2, N_imp=4, N_buffer=0,
                       adatom_species="H", defect_i="H0", defect_j="H0", downfold=True,
                       inputs="defect_files.py", cc_file="./CC_40eV.TSCCEQ-Elec")
    print(run.summary())
    plot_all(run)

run_exchange follows the procedure of the tutorials:

  1. read the complex contour (TranSiesta TSCCEQ file),
  2. read and prune the pristine Hamiltonian -> H_0 and the electrodes,
  3. build the defect specs from the single-defect files,
  4. choose the builder that belongs to the method,
       rkky                         PristineTiled       (single-defect files)
       rkky_sup, liechtenstein_sup  SuperpositionDyson  (single-defect files)
       rkky_nm, liechtenstein       DFTDevice           (two-defect DFT files)
  5. compute J and save everything in folder/results_name:
       ExchangeResult.npz, weighted_orbital_kernel.npy, ExchangeViz.npz,
       device.fdf and run_config.json (all inputs, for reproducibility).

The input file (a .py or .json file, or a dict) lists the DFT files:

    system_name   = "GNR"
    PRISTINE_FILE = "/home/cluster_mount/J_GNRs/Elec/siesta/siesta.fdf"
    DEFECT_FILES  = {"H0": ".../GNR_H0/siesta/siesta.fdf", ...}
    DEVICE_FILES  = {"FM": ".../J_{defect_i}/n{n_dist}/{defect_j}/FM/siesta_FM.fdf",
                     "NM": ".../J_{defect_i}/n{n_dist}/{defect_j}/NM/siesta_NM.fdf"}

DEVICE_FILES are templates ({defect_i}, {defect_j}, {n_dist}, {system_name});
they are only needed for rkky_nm and liechtenstein. A DEFECT_FILES entry can also
be a dict with explicit indices (defects_dependencies.load_defects), e.g.
    "2B-2": {"type": "heteroatom", "file": ".../siesta.fdf", "indices": [40, 51],
             "Z": "B", "anchor": 51}

MPI: run_exchange(..., comm=MPI.COMM_WORLD). Rank 0 reads the inputs and builds
the device (build_on_root broadcasts it and re-raises errors on every rank), the
contour energies are shared by all ranks, rank 0 writes the results and returns
the ExchangeRun; the other ranks return None. exchange_run.py is the driver.

The visualizations are in plot_dependencies.py:

    from plot_dependencies import *
    plot_all(run)

This module only computes and writes results (load_results reads them back).
"""
from __future__ import annotations

import datetime
import json
import runpy
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import scipy.sparse as sps

import J_dependencies as JD
from defects_dependencies import auto_spec, load_defects
from Device_Builder import PristineTiled, DFTDevice, build_on_root, canonical_method
from Dyson_Exchange import SuperpositionDyson, calculate_exchange_dyson
from exchange_static_cache import cache_exchange_static

__all__ = ["run_exchange", "ExchangeRun", "load_inputs", "load_results", "METHODS",
           "DEFAULT_PREFACTOR"]

METHODS = {"rkky": "PristineTiled", "rkky_sup": "SuperpositionDyson",
           "liechtenstein_sup": "SuperpositionDyson", "rkky_nm": "DFTDevice",
           "liechtenstein": "DFTDevice"}
# use one convention for everything. Here ....
DEFAULT_PREFACTOR = {"rkky": -1.0 / np.pi, "rkky_sup": -1.0 / np.pi,
                     "rkky_nm": -1.0 / np.pi, "liechtenstein": -1.0 /  np.pi,
                     "liechtenstein_sup": -1.0 / np.pi }
_DEFAULT_REGION = {"rkky": "all", "rkky_sup": "all", "liechtenstein_sup": "all",
                   "rkky_nm": "radius", "liechtenstein": "radius"}


# ===========================================================================
# Inputs
# ===========================================================================
def load_inputs(inputs) -> dict:
    """Read the file list from a .py / .json file or a dict.

    Keys: DEFECT_FILES (required), PRISTINE_FILE, DEVICE_FILES, system_name.
    """
    if isinstance(inputs, dict):
        data = dict(inputs)
    else:
        path = Path(inputs)
        if not path.is_file():
            raise FileNotFoundError(f"Input file not found: {path}")
        if path.suffix == ".json":
            data = json.loads(path.read_text())
        else:
            data = {k: v for k, v in runpy.run_path(str(path)).items() if not k.startswith("__")}
    if "DEFECT_FILES" not in data:
        raise ValueError("The input file must define DEFECT_FILES = {label: file, ...}")
    return data


def _read_contour(cc_file) -> dict:
    """Contour from a TSCCEQ file, or (energies, weights) arrays (tests)."""
    if isinstance(cc_file, (str, Path)):
        E_real, E_imag, W_real, W_imag, E, W, mu, T = JD.read_transiesta_contour(str(cc_file))
        return dict(En=np.asarray(E, dtype=complex), weights=np.asarray(W, dtype=complex),
                    mu=mu, T=T, E_real=E_real, E_imag=E_imag, W_real=W_real, W_imag=W_imag,
                    source=str(cc_file))
    En, W = (np.asarray(x, dtype=complex) for x in cc_file)
    return dict(En=En, weights=W, mu=None, T=None, E_real=En.real, E_imag=En.imag,
                W_real=W.real, W_imag=W.imag, source="arrays")


def _default_results_name(method, defect_i, defect_j, n_dist, downfold, region, radius,
                          vertices, vertex_source):
    name = f"Results_{defect_i}_{defect_j}_n{n_dist}"
    if region == "radius":
        name += f"_R{radius:.2f}"
    if method == "liechtenstein_sup":
        name += f"_{vertices}"
    if method == "rkky_nm" and vertex_source == "single":
        name += "_single"
    if downfold:
        name += "_downfold"
    return name


# ===========================================================================
# Run object
# ===========================================================================
@dataclass(eq=False)
class ExchangeRun:
    """Everything produced by run_exchange."""
    method: str
    config: dict
    result: "JD.ExchangeResult"
    results_dir: Path
    obj: object                 # Device (PristineTiled / DFTDevice) or DysonSystem
    builder: object
    H_0: object
    contour: dict
    specs: tuple = ()
    info: Dict = field(default_factory=dict)

    @property
    def J_meV(self) -> float:
        return float(self.result.J_meV)

    @property
    def H(self):
        """Device Hamiltonian whose geometry the orbitals refer to."""
        return self.obj.H

    def vertices(self):
        """(Deltas, orbs) used by this run (static part for the Dyson methods)."""
        if self.config["builder"] == "SuperpositionDyson":
            total = self.method == "liechtenstein_sup" and self.config["vertices"] == "region"
            static = self.obj.vertex_total if total else self.obj.vertex_own
            return [sps.csr_matrix(D) for D in static], list(self.obj.orbs)
        return self.obj.vertices(self.method)

    def summary(self) -> str:
        c = self.config
        lines = [f"method {self.method} ({c['builder']}): J = {self.J_meV:.6f} meV",
                 f"  defects {c['defect_i']} - {c['defect_j']}, n_dist {c['n_dist']}, "
                 f"distance {self.obj.anchor_distance:.3f} Ang (anchors)",
                 f"  basis {c['basis']}, region {c['region']}, downfold {c['downfold']}, "
                 f"prefactor {c['prefactor']:.6f}",
                 f"  results: {self.results_dir}"]
        return "\n".join(lines)


# ===========================================================================
# Workflow
# ===========================================================================
def run_exchange(method: str, basis: str = "pz", n_dist: int = 2,
                 defect_i: str = "H0", defect_j: str = "H0",
                 inputs="defect_files.py", cc_file="./CC_40eV.TSCCEQ-Elec",
                 folder=None, results_name: Optional[str] = None,
                 N_imp: Optional[int] = None, N_buffer: int = 1,
                 adatom_species: str = "H", defect_type: str = "adatom", Z=5,
                 downfold: bool = False, region: Optional[str] = None, radius: float = 7.5,
                 vertices: str = "own", vertex_source: Optional[str] = None,
                 vertex_downfold: str = "linear", prefactor: Optional[float] = None,
                 align: str = "vb", nk=(1, 0, 0), lead_eta: float = 1e-5,
                 self_energy_mode: str = "compute", orbital_resolved: bool = True,
                 write_results: bool = True, verbose: bool = True,
                 builder_kwargs: Optional[dict] = None, comm=None,
                 cache_static: bool = True,
                 save_deltas: bool = True) -> Optional[ExchangeRun]:
    """Compute J for one defect pair with any implemented method.

    method          : 'rkky', 'rkky_sup', 'liechtenstein_sup' ('lkag_sup'),
                      'rkky_nm', 'liechtenstein' ('lkag')
    basis           : host basis ('pz', 'pz1', 'pz+dz', ...)
    n_dist          : cells between the anchors (single-defect methods) / separation
                      label of the two-defect DFT files (DFT methods)
    defect_i/j      : labels of DEFECT_FILES
    inputs          : file (.py / .json) or dict with DEFECT_FILES, PRISTINE_FILE,
                      DEVICE_FILES
    cc_file         : TSCCEQ contour file
    folder          : results folder; default H_<system>_<basis>_<method>
    N_imp, N_buffer : device layout (rkky and the superposition methods); N_imp=None
                      sizes the device automatically
    adatom_species / defect_type / Z : how the defects are found in the files
    downfold        : fold the adatoms (rkky, rkky_sup, liechtenstein_sup only)
    region, radius  : vertex region; default 'all' for the single-defect methods,
                      'radius' for the DFT methods
    vertices        : liechtenstein_sup: 'own' (default) or 'region'
    vertex_source   : rkky_nm: 'fm' (default) or 'single'
    vertex_downfold : rkky_sup with downfold: 'linear' (default) or 'full'
    prefactor       : default from DEFAULT_PREFACTOR
    align           : band alignment of the pruned pristine Hamiltonian (prune_ham)
    self_energy_mode: 'compute' (default); 'compute+write' / 'read' cache the lead
                      self-energies in folder/Self_Energies (calculate_exchange only)
    save_deltas     : if False, cache H blocks and Dyson reference only.
    cache_static    : save/read static Delta and H-downfolding blocks in
                      folder/results_name/Static, independently of lead caching.
                      Validates against current builder data each run.
    builder_kwargs  : extra keyword arguments for the builder (e.g. tol, embed_radius)
    comm            : MPI communicator (e.g. MPI.COMM_WORLD) to share the contour
                      energies between ranks; rank 0 returns the ExchangeRun, the
                      other ranks None. With MPI, the DFT/PristineTiled methods let
                      calculate_exchange write the results (it needs the folder).
    """
    method = canonical_method(method)
    if method not in METHODS:
        raise ValueError(f"Unknown method '{method}'. Use one of {tuple(METHODS)}")
    builder_name = METHODS[method]
    if downfold and builder_name == "DFTDevice":
        raise ValueError(f"{method} keeps the adatoms explicit in the DFT device: "
                         "downfold must be False")
    rank = comm.Get_rank() if comm is not None else 0
    parallel = comm is not None and comm.Get_size() > 1
    say = print if (verbose and rank == 0) else (lambda *a, **k: None)
    region = _DEFAULT_REGION[method] if region is None else region
    prefactor = DEFAULT_PREFACTOR[method] if prefactor is None else float(prefactor)
    if vertex_source is None and method == "rkky_nm":
        vertex_source = "fm"

    def prepare():
        """Steps 1-4 (rank 0): inputs, contour, H_0, defect specs, builder."""
        data = load_inputs(inputs)
        system_name = data.get("system_name", "GNR")
        # 1. contour
        contour = _read_contour(cc_file)
        say(f"Contour: {len(contour['En'])} energies, mu = {contour['mu']} eV, T = {contour['T']} K")
        # 2. pristine Hamiltonian -> H_0 and electrodes
        if "PRISTINE_FILE" not in data:
            raise ValueError("The input file must define PRISTINE_FILE (pristine electrode)")
        H_0, H_0_DFT, eshift = JD.prune_ham(data["PRISTINE_FILE"], basis=basis, align=align)
        say(f"Pristine cell: {H_0.na} atoms, {H_0.no} orbitals (basis {basis}, align {align})")

        # 3. defect specs: a file (detected automatically) or explicit indices (dict)
        def spec(label):
            files = data["DEFECT_FILES"]
            if label not in files:
                raise KeyError(f"'{label}' not in DEFECT_FILES ({list(files)})")
            if isinstance(files[label], dict):
                return load_defects({label: files[label]})[label]
            if defect_type == "adatom":
                return auto_spec(label, files[label], "adatom", species=adatom_species)
            return auto_spec(label, files[label], "heteroatom", Z=Z)

        needs_single = builder_name != "DFTDevice" or vertex_source == "single"
        specs = (spec(defect_i), spec(defect_j)) if needs_single else ()

        # 4. builder
        extra = dict(builder_kwargs or {})
        extra.setdefault("verbose", verbose)
        if builder_name == "PristineTiled":
            builder = PristineTiled(H_0, specs, n_dist, basis=basis, N_imp=N_imp, N_buffer=N_buffer,
                                    region=region, radius=radius, downfold=downfold, **extra)
        elif builder_name == "SuperpositionDyson":
            builder = SuperpositionDyson(H_0, specs, n_dist, basis=basis, N_imp=N_imp,
                                         N_buffer=N_buffer, region=region, radius=radius,
                                         downfold=downfold, vertices=vertices, **extra)
        else:
            templates = data.get("DEVICE_FILES", {})
            fill = dict(defect_i=defect_i, defect_j=defect_j, n_dist=n_dist, system_name=system_name)
            fm = templates.get("FM", "").format(**fill) or None
            nm = templates.get("NM", "").format(**fill) or None
            if method == "liechtenstein" and fm is None:
                raise ValueError("liechtenstein needs DEVICE_FILES['FM'] in the input file")
            if method == "rkky_nm" and nm is None:
                raise ValueError("rkky_nm needs DEVICE_FILES['NM'] in the input file")
            if method == "rkky_nm" and vertex_source == "fm" and fm is None:
                raise ValueError("rkky_nm with vertex_source='fm' needs DEVICE_FILES['FM']")
            builder = DFTDevice(fm, H_0, basis=basis, defect_type=defect_type,
                                species=adatom_species, Z=Z,
                                nm_file=nm if method == "rkky_nm" else None,
                                vertex_source=vertex_source if method == "rkky_nm" else None,
                                single_defects=specs or None, region=region, radius=radius,
                                single_region=region if vertex_source == "single" else "all",
                                **extra)
        return dict(data=data, system_name=system_name, contour=contour, H_0=H_0,
                    eshift=eshift, specs=specs, builder=builder)

    # Steps 1-4 on rank 0; an error there is raised on every rank (no deadlock)
    if parallel:
        prep, error = None, None
        if rank == 0:
            try:
                prep = prepare()
            except Exception:
                error = traceback.format_exc()
        error = comm.bcast(error, root=0)
        if error:
            raise RuntimeError("run_exchange failed on rank 0:\n" + error)
        shared = comm.bcast(None if prep is None else
                            {k: v for k, v in prep.items() if k != "builder"}, root=0)
        builder = prep["builder"] if rank == 0 else None
        prep = dict(shared, builder=builder)
        obj = build_on_root(builder, method, comm)        # built on rank 0, broadcast
    else:
        prep = prepare()
        obj = prep["builder"].build(method)
    data, contour, H_0 = prep["data"], prep["contour"], prep["H_0"]

    folder = (Path(folder) if folder is not None
              else Path(f"H_{prep['system_name']}_{basis}_{method}"))
    results_name = results_name or _default_results_name(
        method, defect_i, defect_j, n_dist, downfold, region, radius, vertices, vertex_source)
    results_dir = folder / results_name
    # BEGIN RESTART-SAFE CACHE V2
    # BEGIN EXCHANGE STATIC CACHE
    if cache_static:
        cache_args = dict(
            basis=basis, defect_i=defect_i, defect_j=defect_j,
            n_dist=n_dist, downfold=downfold, region=region, radius=radius,
            vertices=vertices, N_imp=N_imp, N_buffer=N_buffer,
            axis=(builder_kwargs or {}).get("axis", 0),
            save_deltas=save_deltas,
        )
        cache_error = None
        if rank == 0:
            try:
                cache_path = cache_exchange_static(
                    obj, method, folder, results_name, mode="auto", **cache_args)
                say(f"Static vertices / H blocks: {cache_path}")
            except Exception:
                cache_error = traceback.format_exc()
        if parallel:
            cache_error = comm.bcast(cache_error, root=0)
        if cache_error:
            raise RuntimeError("Static exchange cache failed on rank 0:\n" + cache_error)
        if parallel:
            # Every rank must use the loaded static data; rank 0 alone writes.
            local_error = None
            if rank != 0:
                try:
                    cache_exchange_static(
                        obj, method, folder, results_name, mode="read", **cache_args)
                except Exception:
                    local_error = traceback.format_exc()
            errors = comm.allgather(local_error)
            if any(error is not None for error in errors):
                raise RuntimeError("Static exchange cache read failed:\n" +
                                   "\n".join(e for e in errors if e is not None))
    # END EXCHANGE STATIC CACHE
    if rank == 0:
        folder.mkdir(parents=True, exist_ok=True)
    if parallel:
        comm.Barrier()

    # 5. exchange (contour energies shared by all ranks)
    say(f"Computing {method} with {builder_name} on {comm.Get_size() if parallel else 1} rank(s) ...")
    # calculate_exchange writes the results itself when it caches self-energies or runs with MPI
    cached = builder_name != "SuperpositionDyson" and (self_energy_mode != "compute" or parallel)
    if builder_name == "SuperpositionDyson":
        result = calculate_exchange_dyson(
            obj, method, contour["En"], contour["weights"], prefactor=prefactor,
            lead_eta=lead_eta, vertex_downfold=vertex_downfold, vertices=vertices,
            orbital_resolved=orbital_resolved, verbose=verbose,
            comm=comm if parallel else None)
    else:
        result = JD.calculate_exchange(
            **obj.exchange_kwargs(method), prefactor=prefactor, energies=contour["En"],
            contour_weights=contour["weights"], nk=nk, lead_eta=lead_eta,
            orbital_resolved=orbital_resolved, verbose=False,
            self_energy_mode=self_energy_mode,
            calculation_folder=folder if cached else None,
            results_name=results_name if cached else None,
            write_results=cached, mpi=parallel)
    if rank != 0:
        return None
    say(f"J = {result.J_meV:.6f} meV")

    files = data["DEFECT_FILES"]
    config = dict(method=method, builder=builder_name, basis=basis, n_dist=n_dist,
                  defect_i=defect_i, defect_j=defect_j, defect_type=defect_type,
                  adatom_species=adatom_species, Z=Z, downfold=downfold, region=region,
                  radius=radius, vertices=vertices, vertex_source=vertex_source,
                  vertex_downfold=vertex_downfold, prefactor=prefactor, align=align,
                  N_imp=N_imp, N_buffer=N_buffer, nk=list(nk), lead_eta=lead_eta,
                  self_energy_mode=self_energy_mode, cache_static=bool(cache_static), contour=contour["source"],
                  mpi_ranks=comm.Get_size() if parallel else 1,
                  pristine_file=str(data["PRISTINE_FILE"]),
                  defect_files={k: (files[k] if isinstance(files[k], dict) else str(files[k]))
                                for k in (defect_i, defect_j) if k in files},
                  device_files={k: str(v) for k, v in data.get("DEVICE_FILES", {}).items()},
                  J_meV=float(result.J_meV), anchor_distance=float(obj.anchor_distance),
                  distance=float(obj.distance),
                  date=datetime.datetime.now().isoformat(timespec="seconds"))
    run = ExchangeRun(method, config, result, results_dir, obj, prep["builder"], H_0, contour,
                      prep["specs"], info=dict(pristine_eshift=prep["eshift"]))
    if write_results:
        _save(run, cached)
        say(f"Results written to {results_dir}")
    return run


def _save(run: ExchangeRun, cached: bool) -> None:
    """ExchangeResult, orbital kernel, ExchangeViz, device geometry and the run config."""
    d = run.results_dir
    d.mkdir(parents=True, exist_ok=True)
    res = run.result
    if not cached:
        if res.weighted_orbital_kernel_kE is not None:
            np.save(d / "weighted_orbital_kernel.npy", np.asarray(res.weighted_orbital_kernel_kE))
            res.weighted_orbital_kernel_kE = np.load(d / "weighted_orbital_kernel.npy", mmap_mode="r")
        res.save(str(d / "ExchangeResult.npz"))
    H = run.H
    geom = H.geometry
    orb_i = np.asarray(res.orb_i, dtype=int)
    orb_j = np.asarray(res.orb_j, dtype=int)
    adatoms = np.flatnonzero(geom.atoms.Z == 1) if run.config["defect_type"] == "adatom" else []
    # only out-of-plane H next to an anchor (not edge passivation)
    adatoms = np.array([a for a in adatoms
                        if min(np.linalg.norm(geom.xyz[a] - geom.xyz[list(run.obj.anchors)], axis=1)) < 1.6],
                       dtype=int)
    np.savez_compressed(
        d / "ExchangeViz.npz",
        xyz=geom.xyz, cell=geom.cell, Z=geom.atoms.Z,
        orb_i=orb_i, orb_j=orb_j,
        atoms_i=np.asarray(geom.o2a(orb_i), dtype=int),
        atoms_j=np.asarray(geom.o2a(orb_j), dtype=int),
        orbital_names_i=JD.get_orbital_names(H, orb_i),
        orbital_names_j=JD.get_orbital_names(H, orb_j),
        defect_atoms=np.array(run.obj.anchors), adatom_atoms=adatoms,
        method=run.method, basis=run.config["basis"])
    geom.write(str(d / "device.fdf"))
    (d / "run_config.json").write_text(json.dumps(run.config, indent=2))


def load_results(results_dir):
    """(ExchangeResult arrays, ExchangeViz arrays, run_config or None) of a results folder."""
    d = Path(results_dir)
    res = np.load(d / "ExchangeResult.npz", allow_pickle=False)
    geo = np.load(d / "ExchangeViz.npz", allow_pickle=False)
    cfg = json.loads((d / "run_config.json").read_text()) if (d / "run_config.json").is_file() else None
    return res, geo, cfg
