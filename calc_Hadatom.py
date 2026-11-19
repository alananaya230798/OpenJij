#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-method H-adatom exchange driver for OpenJij.

Requires Exchange_Workflow.py, plot_dependencies.py and their dependencies.
Select exactly ONE METHOD below, then run:
    mpirun -np 8 python H_adatom_exchange_run.py

The workflow performs hydrogen downfolding internally; there is no need to
prepare/load the old Hydrogen_downfold/*.npz files in this driver.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # save figures on clusters without a graphical display
from matplotlib import pyplot as plt
from mpi4py import MPI
from Exchange_Workflow import run_exchange
from plot_dependencies import (
    plot_setup,
    plot_vertices,
    plot_dos,
)
import sys
import numpy as np
# =====================================================================
# INPUTS -- edit these only
# =====================================================================

# General
system_name = "GNR"
HOME = Path("/home/cluster_mount/J_GNRs")
CC_FILE = "/home/Magnetic_Interactions/OpenJij/CC_40eV.TSCCEQ-Elec"
METHODS = {
    "rkky": "rkky",
    "rkky_sup": "rkky_sup",
    "lkag_sup": "liechtenstein_sup",
    "rkky_nm": "rkky_nm",
    "lkag": "liechtenstein",
}

# Method
METHOD = "rkky"              # rkky, rkky_sup, lkag_sup
BASIS = "pz+dz"             # pz1, pz, pz+dz, full
N_IMP = 5
N_BUFFER = 1
NK = (1, 0, 0)
ALIGN = "vb"
DOWNFOLD = True

# Vertex regions (all methods): disjoint, never overlapping
REGION = "disjoint"            # disjoint (default), radius, square, atom_mask ('all' removed)
REGION_ASSIGN = None           # None -> 'dominant' (rkky, rkky_sup, lkag_sup) / 'nearest' (DFT devices)
REGION_WEIGHT = "delta"        # 'delta' (||H_up-H_dn|| per atom, eV) or 'mulliken' (|m_a|, muB)
REGION_WMIN = 1e-4             # threshold on the weight (eV or muB)
REGION_RCUT = None             # Angstrom from the defect centre (None: no cutoff)
DFT_RADIUS = 8.0               # only for REGION = 'radius' / 'square'
SPARSITY_ATOL = 1e-6           # eliminate_zeros tolerance for every H/S read
RKKY_NM_VERTEX_SOURCE = "fm"   # "fm" or "single"

# Electrode
PRISTINE_FILE = HOME / "Elec/siesta/siesta.fdf"
TRANSPORT_AXIS = 0          # Documentation: direction A; controlled by builder
SEMI_L = "-A"               # Documentation: controlled by builder
SEMI_R = "+A"               # Documentation: controlled by builder
LEAD_ETA = 1e-5
SELF_ENERGY_MODE = "compute+write"

# Defects
DEFECT_I = "H0"
DEFECT_J = "H0"
N_DIST = 1
defect_type="adatom"
adatom_species="H"

defect_dict = {
    "H0": { "file": HOME / f"{system_name}_H0/siesta/siesta.fdf", "l": "H0"},
    "H1": { "file": HOME / f"{system_name}_H1/siesta/siesta.fdf", "l": "H1"},
    "H2": { "file": HOME / f"{system_name}_H2/siesta/siesta.fdf", "l": "H2"},
    "H3": {"file": HOME / f"{system_name}_H3/siesta/siesta.fdf", "l": "H3"},
}

# Two-defect DFT Hamiltonian templates. Change these to your actual directories.
# Supported placeholders: {defect_i}, {defect_j}, {n_dist}, {system_name}
# FM: spin-polarized FM two-H device; NM: nonmagnetic two-H device.
DEVICE_FILES = {
    "FM": str(HOME / "J_{defect_i}/n{n_dist}/{defect_j}/FM/siesta_FM.fdf"),
    "NM": str(HOME / "J_{defect_i}/n{n_dist}/{defect_j}/NM/siesta_NM.fdf"),
}

# Optional command-line arguments
if len(sys.argv) == 6:
    DEFECT_I, DEFECT_J, n_dist_arg, BASIS, METHOD = sys.argv[1:]
    try:
        N_DIST = int(n_dist_arg)
    except ValueError as exc:
        raise SystemExit(f"Invalid N_DIST: {n_dist_arg}") from exc

elif len(sys.argv) != 1:
    raise SystemExit( "Usage: python H_adatom_exchange_run.py [DEFECT_I DEFECT_J N_DIST BASIS]")

# Output
IS_DFT_DEVICE = METHOD in ("rkky_nm", "lkag")
EFFECTIVE_DOWNFOLD = DOWNFOLD and not IS_DFT_DEVICE

folder = Path("Examples/HydrogenAdatomInGNR") / f"{BASIS}_{METHOD}"
results_name = f"Results_{DEFECT_I}_{DEFECT_J}_n{N_DIST}"
results_name += f"_{REGION}"
if REGION in ("radius", "square"):
    results_name += f"{DFT_RADIUS:.2f}"
if IS_DFT_DEVICE:
    if METHOD == "rkky_nm":
        results_name += f"_{RKKY_NM_VERTEX_SOURCE}"
if EFFECTIVE_DOWNFOLD:
    results_name += "_downfold"

SAVE_DOS = False                      # DOS computed with all MPI ranks inside run_exchange
DOS_ENERGIES = np.linspace(-0.25, 0.25, 51)   # eV around E_F (was -0.25..25 eV)
SAVE_PDF = False
PLOT_DPI = 200

# =====================================================================
# End of inputs
# =====================================================================

def save_fig(fig, name, plot_dir):
    fig.savefig(plot_dir / f"{name}.png", dpi=PLOT_DPI, bbox_inches="tight")
    if SAVE_PDF:
        fig.savefig(plot_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if METHOD not in METHODS:
        raise ValueError(f"METHOD must be one of {list(METHODS)}")
    if DEFECT_I not in defect_dict or DEFECT_J not in defect_dict:
        raise ValueError(f"Unknown defect; choose from {list(defect_dict)}")
    if N_DIST < 0:
        raise ValueError("N_DIST must be nonnegative")

    method = METHODS[METHOD]

    # Resolve only the two-defect files needed by the selected method.
    required = []
    if METHOD == "lkag":
        required = ["FM"]
    elif METHOD == "rkky_nm":
        required = ["NM"] + (["FM"] if RKKY_NM_VERTEX_SOURCE == "fm" else [])

    device_files = {}
    for state in required:
        file = Path(DEVICE_FILES[state].format(
            defect_i=DEFECT_I, defect_j=DEFECT_J,
            n_dist=N_DIST, system_name=system_name,
        ))
        if not file.is_file():
            raise FileNotFoundError(f"Missing {state} two-defect device for {METHOD}: {file}. Update DEVICE_FILES to match your SIESTA directories.")
        device_files[state] = str(file)

    inputs = {
        "system_name": system_name,
        "PRISTINE_FILE": str(PRISTINE_FILE),
        "DEVICE_FILES": device_files,
        "DEFECT_FILES": { label: str(data["file"]) for label, data in defect_dict.items()},}

    if rank == 0:
        print(f"Method: {METHOD}; defects: {DEFECT_I}/{DEFECT_J}; "
              f"separation: {N_DIST}; basis: {BASIS}; downfold: {EFFECTIVE_DOWNFOLD}", flush=True)

    run = run_exchange(
        method=method,
        basis=BASIS,
        n_dist=N_DIST,
        defect_i=DEFECT_I,
        defect_j=DEFECT_J,
        inputs=inputs,
        cc_file=CC_FILE,
        folder=folder,
        results_name=results_name,
        N_imp=N_IMP,
        N_buffer=N_BUFFER,
        defect_type=defect_type,
        adatom_species=adatom_species,
        downfold=EFFECTIVE_DOWNFOLD,
        region=REGION,
        # for dft device
        radius=DFT_RADIUS,
        region_assign=REGION_ASSIGN,
        region_weight=REGION_WEIGHT,
        region_wmin=REGION_WMIN,
        region_rcut=REGION_RCUT,
        sparsity_atol=SPARSITY_ATOL,
        # for rkky+nm 
        vertex_source=RKKY_NM_VERTEX_SOURCE if METHOD == "rkky_nm" else None,
        builder_kwargs={"axis": TRANSPORT_AXIS},
        align=ALIGN,
        nk=NK,
        lead_eta=LEAD_ETA,
        self_energy_mode=SELF_ENERGY_MODE,
        orbital_resolved=True,
        dos_energies=DOS_ENERGIES if SAVE_DOS else None,   # -> results_name/dos.npz
        write_results=True,
        verbose=True,
        comm=comm,
    )

    # Nonroot MPI ranks return None and never generate duplicate figures.
    if run is None:
        return
    # Save only THIS method's plots next to THIS method's numerical results.
    plot_dir = run.results_dir / "Plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    save_fig(plot_setup(run), "01_device_setup", plot_dir)
    for k, fig in enumerate(plot_vertices(run)):
        save_fig(fig, f"02_delta_{'ij'[k]}", plot_dir)

    # This function returns a data dict but creates a Matplotlib figure.
    #plot_orbital_contributions(run.results_dir)
    #save_fig(plt.gcf(), "03_orbital_contributions", plot_dir)
    #save_fig(plot_exchange_kernel(run.results_dir), "04_exchange_kernel", plot_dir)
    #save_fig(plot_atomic_contributions(run.results_dir), "05_atomic_contributions", plot_dir)
    
    if SAVE_DOS:
        fig, _curves = plot_dos(run,energies=DOS_ENERGIES )          # plots the curves computed in parallel
        save_fig(fig, "06_reference_DOS", plot_dir)

    (plot_dir / "summary.txt").write_text(run.summary() + "\n")
    print(run.summary(), flush=True)
    print(f"Figures saved in {plot_dir}", flush=True)


if __name__ == "__main__":
    main()
