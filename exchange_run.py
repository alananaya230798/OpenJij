#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MPI driver for exchange calculations (replaces RKKY_calc_2B.py and Liecht_calc_Hadatom.py).

    mpirun -np 8 python exchange_run.py METHOD DEFECT_I DEFECT_J N_DIST [options]

Examples
    mpirun -np 8 python exchange_run.py rkky H0 H0 2 --downfold --N-imp 4 --N-buffer 0
    mpirun -np 8 python exchange_run.py rkky_sup H0 H2 3 --downfold
    mpirun -np 8 python exchange_run.py liechtenstein_sup H0 H0 2 --vertices own
    mpirun -np 8 python exchange_run.py rkky_nm H0 H2 2 --radius 7.5
    mpirun -np 8 python exchange_run.py liechtenstein H0 H2 2 --radius 7.5 --basis pz+dz

METHOD   rkky | rkky_sup | liechtenstein_sup (lkag_sup) | rkky_nm | liechtenstein (lkag)
The DFT files are listed in the input file (--inputs, default defect_files.py):
PRISTINE_FILE, DEFECT_FILES and, for rkky_nm / liechtenstein, DEVICE_FILES.

Rank 0 reads the inputs and builds the device, the contour energies are shared
between the ranks, rank 0 writes folder/results_name (ExchangeResult.npz,
weighted_orbital_kernel.npy, ExchangeViz.npz, device.fdf, run_config.json).
Plots: load the results folder in a notebook with plot_dependencies.
"""
import argparse
import datetime
import sys

from Exchange_Workflow import METHODS, run_exchange


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("method", help=f"one of {', '.join(METHODS)} (aliases lkag, lkag_sup)")
    p.add_argument("defect_i", help="label of defect i (key of DEFECT_FILES)")
    p.add_argument("defect_j", help="label of defect j")
    p.add_argument("n_dist", type=int, help="cells between the anchors / DFT separation label")
    p.add_argument("--basis", default="pz")
    p.add_argument("--inputs", default="defect_files.py", help="input file (.py or .json)")
    p.add_argument("--cc-file", default="./CC_40eV.TSCCEQ-Elec", help="TSCCEQ contour file")
    p.add_argument("--folder", default=None, help="results folder (default H_<system>_<basis>_<method>)")
    p.add_argument("--results-name", default=None)
    p.add_argument("--N-imp", type=int, default=None, help="device layout; default automatic")
    p.add_argument("--N-buffer", type=int, default=1)
    p.add_argument("--adatom-species", default="H")
    p.add_argument("--defect-type", default="adatom", choices=("adatom", "heteroatom"))
    p.add_argument("--Z", default="5", help="heteroatom species (Z or symbol)")
    p.add_argument("--downfold", action="store_true", help="fold the adatoms (rkky, rkky_sup, liechtenstein_sup)")
    p.add_argument("--region", default=None, choices=("all", "radius"))
    p.add_argument("--radius", type=float, default=7.5)
    p.add_argument("--vertices", default="own", choices=("own", "region"))
    p.add_argument("--vertex-source", default=None, choices=("fm", "single"))
    p.add_argument("--vertex-downfold", default="linear", choices=("linear", "full"))
    p.add_argument("--prefactor", type=float, default=None)
    p.add_argument("--self-energy-mode", default="compute", choices=("compute", "compute+write", "read"))
    p.add_argument("--serial", action="store_true", help="do not use MPI")
    return p.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    comm = None
    if not a.serial:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
    rank = comm.Get_rank() if comm is not None else 0
    if rank == 0:
        n = comm.Get_size() if comm is not None else 1
        print(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}  {a.method} {a.defect_i} {a.defect_j} "
              f"n{a.n_dist} on {n} rank(s)", flush=True)
    Z = int(a.Z) if a.Z.isdigit() else a.Z
    run = run_exchange(a.method, basis=a.basis, n_dist=a.n_dist, defect_i=a.defect_i,
                       defect_j=a.defect_j, inputs=a.inputs, cc_file=a.cc_file, folder=a.folder,
                       results_name=a.results_name, N_imp=a.N_imp, N_buffer=a.N_buffer,
                       adatom_species=a.adatom_species, defect_type=a.defect_type, Z=Z,
                       downfold=a.downfold, region=a.region, radius=a.radius,
                       vertices=a.vertices, vertex_source=a.vertex_source,
                       vertex_downfold=a.vertex_downfold, prefactor=a.prefactor,
                       self_energy_mode=a.self_energy_mode, comm=comm)
    if run is not None:
        print(run.summary(), flush=True)
        print(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}  done", flush=True)
    return run


if __name__ == "__main__":
    main(sys.argv[1:])
