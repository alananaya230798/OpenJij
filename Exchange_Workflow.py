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

MPI: run_exchange(..., comm=MPI.COMM_WORLD). Rank 0 reads the inputs and loads
the built device from the cache (or builds and caches it); the other ranks receive
it by broadcast and read no file. Errors on rank 0 are raised on every rank. The
contour energies are shared by all ranks, BLAS threads are limited per rank
(blas_threads), rank 0 writes the results and returns the ExchangeRun; the other
ranks return None. exchange_run.py is the driver.

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
from Device_Builder import PristineTiled, DFTDevice, canonical_method
from Dyson_Exchange import SuperpositionDyson, calculate_exchange_dyson
from exchange_static_cache import build_key, cache_dir, load_build, save_build
from exchange_mpi_io import limit_blas_threads

__all__ = ["run_exchange", "ExchangeRun", "load_inputs", "load_results", "load_dos",
           "dos_curves", "METHODS",
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
    dos: Optional[dict] = None   # DOS curves computed by run_exchange(dos_energies=...)

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
                 cache_static: bool = True, save_deltas: bool = True,
                 force_rebuild: bool = False, solver: str = "sparse",
                 orbital_kernel: str = "classes",
                 blas_threads="auto", dos_energies=None, dos_eta: float = 0.01,
                 dos_orbitals="region") -> Optional[ExchangeRun]:
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
    cache_static    : load-first cache of the built device in folder/results_name/Static:
                      if the build key (settings, input-file fingerprints, builder
                      source) matches, the built device is loaded and no DFT file
                      is read; otherwise it is built and saved.
    save_deltas     : also write Delta_i.npz / Delta_j.npz to Static/ (inspection).
    force_rebuild   : ignore the cache once (it is rewritten).
    solver          : Dyson methods: 'sparse' (default) or 'dyson' (dense, reference).
    orbital_kernel  : per-energy orbital output with orbital_resolved: 'classes'
                      (default; kernel per pair of orbital classes, small, no files),
                      'full' (orbital-pair kernel per energy: n_i x n_j x nE, large;
                      with MPI one file per energy) or 'none'. J_orb is always kept.
    blas_threads    : with MPI, BLAS threads per rank: 'auto' (cores of the node /
                      ranks on the node), an integer, or None to leave them alone.
    dos_energies    : energies (eV) for the DOS of the reference (and its comparison
                      curve), computed with all ranks after J and saved as
                      results_name/dos.npz; plot_dos then plots them. None: no DOS.
    dos_eta         : broadening of the DOS (eV).
    dos_orbitals    : device builders: 'region' (defect regions, default), None
                      (total device DOS) or device orbital indices.
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
        """Steps 1-4 (rank 0): inputs and contour; the built device from the cache
        (no DFT file read) or from pruning, defect specs and the builder."""
        data = load_inputs(inputs)
        system_name = data.get("system_name", "GNR")
        # 1. contour
        contour = _read_contour(cc_file)
        say(f"Contour: {len(contour['En'])} energies, mu = {contour['mu']} eV, T = {contour['T']} K")
        if "PRISTINE_FILE" not in data:
            raise ValueError("The input file must define PRISTINE_FILE (pristine electrode)")
        folder_ = Path(folder) if folder is not None else Path(f"H_{system_name}_{basis}_{method}")
        name_ = results_name or _default_results_name(
            method, defect_i, defect_j, n_dist, downfold, region, radius, vertices, vertex_source)

        files = data["DEFECT_FILES"]
        needs_single = builder_name != "DFTDevice" or vertex_source == "single"
        for label in ((defect_i, defect_j) if needs_single else ()):
            if label not in files:
                raise KeyError(f"'{label}' not in DEFECT_FILES ({list(files)})")
        fm = nm = None
        if builder_name == "DFTDevice":
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
            if method != "rkky_nm":
                nm = None

        # Build key: every setting of the build, the input files, the builder source
        entry = lambda k: files[k] if isinstance(files[k], dict) else str(files[k])
        input_files = [data["PRISTINE_FILE"]]
        if needs_single:
            input_files += [e["file"] if isinstance(e, dict) else e
                            for e in (entry(defect_i), entry(defect_j))]
        input_files += [f for f in (fm, nm) if f]
        settings = dict(builder=builder_name, method=method, basis=basis, align=align,
                        n_dist=n_dist, defect_i=defect_i, defect_j=defect_j,
                        defect_type=defect_type, adatom_species=adatom_species, Z=Z,
                        downfold=downfold, region=region, radius=radius, vertices=vertices,
                        vertex_source=vertex_source, N_imp=N_imp, N_buffer=N_buffer,
                        builder_kwargs=sorted((k, repr(v)) for k, v in (builder_kwargs or {}).items()),
                        defect_entries=[entry(k) for k in ((defect_i, defect_j) if needs_single else ())])
        key = build_key(settings, input_files) if cache_static else None
        common = dict(data=data, system_name=system_name, contour=contour,
                      folder=str(folder_), results_name=name_)
        if cache_static and not force_rebuild:
            cached = load_build(cache_dir(folder_, name_), key, verbose=verbose)
            if cached is not None:
                return dict(common, obj=cached["obj"], H_0=cached["H_0"], eshift=cached["eshift"],
                            specs=cached["specs"], builder=None, from_cache=True)

        # 2. pristine Hamiltonian -> H_0 and electrodes
        H_0, H_0_DFT, eshift = JD.prune_ham(data["PRISTINE_FILE"], basis=basis, align=align)
        say(f"Pristine cell: {H_0.na} atoms, {H_0.no} orbitals (basis {basis}, align {align})")

        # 3. defect specs: a file (detected automatically) or explicit indices (dict)
        def spec(label):
            if isinstance(files[label], dict):
                return load_defects({label: files[label]})[label]
            if defect_type == "adatom":
                return auto_spec(label, files[label], "adatom", species=adatom_species)
            return auto_spec(label, files[label], "heteroatom", Z=Z)

        specs = (spec(defect_i), spec(defect_j)) if needs_single else ()

        # 4. builder and build
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
            builder = DFTDevice(fm, H_0, basis=basis, defect_type=defect_type,
                                species=adatom_species, Z=Z, nm_file=nm,
                                vertex_source=vertex_source if method == "rkky_nm" else None,
                                single_defects=specs or None, region=region, radius=radius,
                                single_region=region if vertex_source == "single" else "all",
                                **extra)
        obj = builder.build(method)
        if cache_static:
            path = save_build(cache_dir(folder_, name_), key,
                              dict(obj=obj, H_0=H_0, eshift=eshift, specs=specs), method, save_deltas)
            say(f"[build cache] built device saved to {path}", flush=True)
        return dict(common, obj=obj, H_0=H_0, eshift=eshift, specs=specs, builder=builder,
                    from_cache=False)

    # Steps 1-4 on rank 0 only; an error there is raised on every rank (no deadlock).
    # The other ranks receive the built device by broadcast and read no file.
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
        prep = dict(shared, builder=prep["builder"] if rank == 0 else None)
    else:
        prep = prepare()
    obj = prep["obj"]
    data, contour, H_0 = prep["data"], prep["contour"], prep["H_0"]
    folder, results_name = Path(prep["folder"]), prep["results_name"]
    results_dir = folder / results_name
    if rank == 0:
        folder.mkdir(parents=True, exist_ok=True)
    if parallel:
        comm.Barrier()
        threads = limit_blas_threads(comm, blas_threads)
        say(f"BLAS threads per rank: {threads}")

    # 5. exchange (contour energies shared by all ranks)
    say(f"Computing {method} with {builder_name} on {comm.Get_size() if parallel else 1} rank(s) ...")
    # calculate_exchange writes the results itself when it caches self-energies or runs with MPI
    cached = builder_name != "SuperpositionDyson" and (self_energy_mode != "compute" or parallel)
    if builder_name == "SuperpositionDyson":
        result = calculate_exchange_dyson(
            obj, method, contour["En"], contour["weights"], prefactor=prefactor,
            lead_eta=lead_eta, vertex_downfold=vertex_downfold, vertices=vertices,
            orbital_resolved=orbital_resolved, verbose=verbose,
            comm=comm if parallel else None, solver=solver, orbital_kernel=orbital_kernel)
    else:
        result = JD.calculate_exchange(
            **obj.exchange_kwargs(method), prefactor=prefactor, energies=contour["En"],
            contour_weights=contour["weights"], nk=nk, lead_eta=lead_eta,
            orbital_resolved=orbital_resolved, orbital_kernel=orbital_kernel, verbose=False,
            self_energy_mode=self_energy_mode,
            calculation_folder=folder if cached else None,
            results_name=results_name if cached else None,
            write_results=cached, mpi=parallel)
    curves = None
    if dos_energies is not None:            # DOS with all ranks (energies distributed)
        say(f"DOS: {len(np.atleast_1d(dos_energies))} energies on "
            f"{comm.Get_size() if parallel else 1} rank(s) ...")
        curves = dos_curves(obj, method, builder_name, dos_energies, eta=dos_eta,
                            orbitals=dos_orbitals, comm=comm if parallel else None, solver=solver)
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
                  self_energy_mode=self_energy_mode, cache_static=bool(cache_static),
                  build_from_cache=bool(prep.get("from_cache", False)), solver=solver,
                  orbital_kernel=orbital_kernel, contour=contour["source"],
                  mpi_ranks=comm.Get_size() if parallel else 1,
                  pristine_file=str(data["PRISTINE_FILE"]),
                  defect_files={k: (files[k] if isinstance(files[k], dict) else str(files[k]))
                                for k in (defect_i, defect_j) if k in files},
                  device_files={k: str(v) for k, v in data.get("DEVICE_FILES", {}).items()},
                  J_meV=float(result.J_meV), anchor_distance=float(obj.anchor_distance),
                  distance=float(obj.distance),
                  date=datetime.datetime.now().isoformat(timespec="seconds"))
    if curves is not None:
        config.update(dos_eta=dos_eta, dos_orbitals=(dos_orbitals if isinstance(dos_orbitals, (str, type(None)))
                                                     else "indices"),
                      dos_n_energies=len(np.atleast_1d(dos_energies)))
    run = ExchangeRun(method, config, result, results_dir, obj, prep["builder"], H_0, contour,
                      prep["specs"], info=dict(pristine_eshift=prep["eshift"]))
    run.dos = curves
    if write_results:
        _save(run, cached)
        if curves is not None:
            save_dos(results_dir / "dos.npz", curves,
                     meta=dict(method=method, eta=dos_eta, orbitals=config["dos_orbitals"]))
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


def dos_curves(obj, method: str, builder_name: str, energies, eta: float = 0.01,
               orbitals="region", comm=None, solver: str = "sparse") -> Optional[dict]:
    """DOS of the reference of `method` and its comparison curve (rank 0; others None).

    rkky                         pristine device
    rkky_sup, liechtenstein_sup  defect region R, compared with the pristine host on R
    rkky_nm                      NM device, compared with the FM device (if available)
    liechtenstein                FM device, spin up and down
    orbitals: device builders, None (total DOS), 'region' (defect regions) or indices.
    With comm the energies are distributed over the ranks.
    """
    from Device_Builder import compute_dos
    E = np.asarray(energies, dtype=float)
    kw = dict(eta=eta, comm=comm, solver=solver)
    if builder_name != "SuperpositionDyson":
        kw["orbitals"] = orbitals
    wanted = [(method, method)]
    if builder_name == "SuperpositionDyson":
        wanted.append(("pristine host (same region)", "rkky"))
    if method == "rkky_nm" and "liechtenstein" in getattr(obj, "methods", ()):
        wanted.append(("FM device", "liechtenstein"))
    curves = {label: compute_dos(obj, m, E, **kw) for label, m in wanted}
    if comm is not None and comm.Get_rank() != 0:
        return None
    return curves


def save_dos(path, curves: dict, meta: Optional[dict] = None) -> None:
    """Write DOS curves to an .npz file (no pickle)."""
    names = list(curves)
    data = {"names": np.asarray(names, dtype="U64"),
            "energies": np.asarray(next(iter(curves.values()))["energies"])}
    for k, name in enumerate(names):
        data[f"dos_{k}"] = np.asarray(curves[name]["dos"])
        data[f"spins_{k}"] = np.asarray(curves[name]["spins"], dtype="U8")
        data[f"n_orbitals_{k}"] = np.asarray(len(curves[name].get("orbitals", ())))
    if meta:
        data["meta"] = np.asarray(json.dumps(meta))
    np.savez_compressed(path, **data)


def load_dos(results_dir) -> Optional[dict]:
    """DOS curves saved by run_exchange(dos_energies=...), or None."""
    f = Path(results_dir) / "dos.npz"
    if not f.is_file():
        return None
    with np.load(f, allow_pickle=False) as z:
        E = z["energies"].copy()
        out = {}
        for k, name in enumerate(z["names"]):
            out[str(name)] = dict(energies=E, dos=z[f"dos_{k}"].copy(),
                                  spins=tuple(str(x) for x in z[f"spins_{k}"]))
        if "meta" in z.files:
            out["_meta"] = json.loads(str(z["meta"].item()))
    return out


def load_results(results_dir):
    """(ExchangeResult arrays, ExchangeViz arrays, run_config or None) of a results folder."""
    d = Path(results_dir)
    res = np.load(d / "ExchangeResult.npz", allow_pickle=False)
    geo = np.load(d / "ExchangeViz.npz", allow_pickle=False)
    cfg = json.loads((d / "run_config.json").read_text()) if (d / "run_config.json").is_file() else None
    return res, geo, cfg
