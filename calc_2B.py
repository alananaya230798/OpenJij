#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pairwise RKKY exchange for magnetic sites of a 2B-7AGNR.

Based on RKKY_2B-7AGNRs.ipynb and RKKY_calc_Hadatom.py.

Usage (same positional ordering as the H-adatom script)::

    mpirun -np 8 python RKKY_calc_2B.py 2B-1 2B-2 3 pz
    mpirun -np 8 python RKKY_calc_2B.py 2B-2 2B-1 5 'pz+dz'

The labels select which boron-centred magnetic site is embedded. n_dist
counts pristine 7AGNR cells between the chosen host cell positions, as
in the notebook; the geometric separation is calculated from the anchors.

Assumes J_dependencies.py and exchange_mpi_io.py are importable, as in the
original notebook. Each MPI job has its own folder (including self-energies).
"""

import datetime 
import json
import math
import os
from pathlib import Path
import traceback
import numpy as np
import sisl
from mpi4py import MPI
import socket
import sys

from J_dependencies import *
from exchange_mpi_io import *

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def root_print(*values):
    if rank == 0:
        print(*values, flush=True)

if rank == 0:
    print(f"Running RKKY calculation with {size} MPI processes", flush=True)
print(f"MPI rank {rank}/{size} running on {socket.gethostname()}",flush=True)

# =====================================================================
# 1. User configuration
# =====================================================================
# system
system_name="7AGNR"
write_results=True

# Home directory
HOME='/home/Magnetic_Interactions/'

# Pristine Hamiltonian file
H_PRISTINE_FILE=HOME+'7AGNR/Siesta_SP/siesta.fdf'

# Transport directions
semi_L = "-A"
semi_R = "+A"
self_energy_mode="compute+write"
nk=(1, 0, 0)  
n_tile=2 # For tiling electrode for self energy calculation
tile_ax=0 # tiling axis for self energy calculation 

# Method
method='rkky'

# Basis
default_basis= 'pz'

# Band alignement
align='vb'

# Defect configurations
perturbation="2B"
tol=0.25  # tolerance to match atoms in A
pert_type='heteroatom'
pert_Z = 5

defect_dict= {"2B-1": {"idx": 15,
                        "file": Path("/home/Magnetic_Interactions/2B-7AGNR_15/Siesta_SP/siesta.fdf"),
                        "boron_ordinal": 0},
               "2B-2": {"idx": 26,
                        "file": Path("/home/Magnetic_Interactions/2B-7AGNR_15/Siesta_SP/siesta.fdf"), 
                        "boron_ordinal": 1}}

# Choose the two defect types
default_defect_i = "2B-2"
default_defect_j = "2B-1"

# Distance between defects
default_n_dist=1

# Device size 2*(N_imp+N_buferr)+ n_dist + 1
N_imp = 8 # cell in the perturbation calculation where the perturbation is placed STARTING FROM 0,1,...N_imp
N_buffer=1

# complex contour
cc_file = "/home/Magnetic_Interactions/CC_40eV.TSCCEQ-Elec"

# =====================================================================
# End of user configuration
# =====================================================================

# Retrieve defects
defect_i = sys.argv[1] if len(sys.argv) > 1 else default_defect_i
defect_j = sys.argv[2] if len(sys.argv) > 2 else default_defect_j
defects_lables=[defect_i,defect_j]
selected_defects = [defect_dict[defect_i],defect_dict[defect_j]]
D_idx = [d["idx"] for d in selected_defects]
D1, D2 = D_idx
root_print("Defect labels D1,D2:",D1,D2)

# Defect Hamiltonian files
H_files = [d["file"] for d in selected_defects]

# Retrive n_dist
n_dist= int(sys.argv[3]) if len(sys.argv) > 1 else default_n_dist

# Retrive basis
basis= sys.argv[4] if len(sys.argv) > 4 else default_basis

# Folders
folder=Path(f'{perturbation}_{system_name}_{basis}_{method}')
results_name=f"Results_{defect_i}_{defect_j}_n{n_dist}"

input_data = None
if rank == 0:
    folder.mkdir(exist_ok=True)
    root_print(folder)
    # ===============================================================
    # 2.0 Read Contour
    # ===============================================================
    _, _, _, _, E, W, mu, T = read_transiesta_contour(cc_file)
    
    En = np.asarray(E, dtype=complex)
    weights = np.asarray(W, dtype=complex)
    root_print("Number of contour energies:", len(En))
    root_print("Chemical potential:", mu, "eV") 
    root_print("Electronic temperature:", T, "K") 

    H_0_DFT=sisl.get_sile(H_PRISTINE_FILE).read_hamiltonian()
    H_0, _,_ = prune_ham(H_PRISTINE_FILE,basis=basis, align= align)    
    input_data = (En, weights, mu, T, H_0)

# All MPI ranks receive the data
En, weights, mu, T, H_0 = comm.bcast(input_data, root=0)

#Tile 
HamL=H_0.tile(n_tile,tile_ax).copy()
HamR=H_0.tile(n_tile,tile_ax).copy()

# ===============================================================
# 3.0 Perturbations 
# ===============================================================
N  = 2*(N_imp+N_buffer) + 1 + n_dist
root_print("N:",N)

H=H_0.tile(N,0)    # transport direction in x
H.set_nsc([1,1,1]) # remove pbc in transport direction x
na=H.na    
no=H.no
root_print(f"Number of atoms in device:{na}")
root_print(f"Number of orbitals in device:{no}")
if rank==0:
    H.geometry.write(f"{folder}/{results_name}/device.fdf")

# Pristine indexes to match
Pristine_idx=[(N_imp+N_buffer)*H_0.na + D1 ,(N_imp+N_buffer+n_dist)*H_0.na + D2]
root_print("Pristine indices:",Pristine_idx)
# Distance in the pristine host
distance=np.linalg.norm(H.geometry.Rij(Pristine_idx[0],Pristine_idx[1]))
root_print(f"Distance {distance:.3f}Å")

Delta_dir = folder / "Deltas"
if rank == 0:
    Delta_dir.mkdir(parents=True, exist_ok=True)
Delta_files = [Delta_dir / (f"Delta_{defects_lables[i]}_{Pristine_idx[i]}.npz") for i in range(2)]

Deltas=[None,None]
orbs=[None,None]

for i,delta_file in enumerate(Delta_files):
    delta_data=None
    if rank == 0:
        if delta_file.is_file():
            root_print(f"Reading Delta {i}: {delta_file}")
            Delta, orb = load_local_delta(delta_file)
            delta_data = (Delta, np.asarray(orb, dtype=int))
        else: 
            H_imp=sisl.get_sile(H_files[i]).read_hamiltonian()   
            if pert_type=='heteroatom':
            # Find Boron atom to anchor,  Ordinal relates to their relative numbering
                p_idx=np.where(H_imp.geometry.atoms.Z==pert_Z)[0][selected_defects[i]["boron_ordinal"]]
                root_print(f"Boron atom with {p_idx} index")
                H_idx=()
                allow_species_mismatch=True,
            if pert_type=='adatom':
                # Future Change to a more general code with adatoms
                p_idx=find_C_with_outofplane_H(H_imp.geometry)[0]
                H_idx=H_imp.geometry.close(C_idx,1.2)[1]
                root_print(f"Carbon {C_idx} has a H adatom with atom index {H_idx} ")
            
            Delta, orb, _ = extract_single_Delta(
            H_host=H, # Tiled host Hamiltonian from tiled Hamiltonian
            H_polarized_file=H_files[i], # File for spin polarized perturbation
            host_anchor=Pristine_idx[i], # C Host anchor before prunning 
            defect_anchor=p_idx, # C perturbation anchor 
            basis=basis, # which orbitals to keep        
            tol=tol,  # tolerance for matching atoms
            #needs to be generalized to None or empty list
            adatom_atoms=[H_idx], # to remove Adatoms 
            allow_species_mismatch=allow_species_mismatch)
            
            root_print(f"Saving Delta {i}: {delta_file}")
            save_local_delta(delta_file, Delta, orb)
            delta_data = (Delta, np.asarray(orb, dtype=int))
        # Broadcast the Delta matrix and mapped orbital indices
    Delta, orb = comm.bcast(delta_data, root=0)
    Deltas[i] = Delta
    orbs[i] = np.asarray(orb, dtype=int)
    
root_print("Device:")
root_print("Device nsc:", H.nsc)

root_print("Defect i:")
root_print("Delta shape:", Deltas[0].shape)
root_print("Mapped orbitals:", len(orbs[0]))
root_print("Orbital range:", np.min(orbs[0]), np.max(orbs[0]))
root_print("Max |Delta_i|:", np.max(np.abs(Deltas[0].data)), "eV")

root_print("Defect j:")
root_print("Delta shape:", Deltas[1].shape)
root_print("Mapped orbitals:", len(orbs[1]))
root_print("Orbital range:", np.min(orbs[1]), np.max(orbs[1]))
root_print("Max |Delta_j|:", np.max(np.abs(Deltas[1].data)), "eV")

root_print("\nOverlapping magnetic orbitals:")
root_print(np.intersect1d(orbs[0], orbs[1]))

root_print("Device nsc:", H.nsc)
root_print("Left electrode nsc:", HamL.nsc)
root_print("Right electrode nsc:", HamR.nsc)

root_print("========================================")
root_print(f"Jk_calc.py run started at {datetime.datetime.now()}")
root_print("========================================\n")

result = calculate_exchange(
    method=method,
    H_ref=H,
    # Magnetic perturbation i
    Delta_i=Deltas[0],
    orb_i=orbs[0],
    # Magnetic perturbation j
    Delta_j=Deltas[1],
    orb_j=orbs[1],
    # Equilibrium complex contour
    energies=En,
    contour_weights=weights,
    # Gamma point only
    nk=nk,     
    # Pristine GNR electrodes
    HamL=HamL,
    semi_L=semi_L,
    HamR=HamR,
    semi_R=semi_R,
    lead_eta=1e-5,
    # Numerical settings
    orbital_resolved=True,
    verbose=False,
    self_energy_mode=self_energy_mode,
    calculation_folder=folder,
    write_results=write_results,
    results_name =results_name,
    mpi=True,
    distance=distance)
comm.Barrier()

root_print("========================================")
root_print(f"Jk_calc.py run ended at {datetime.datetime.now()}")
root_print("========================================\n")
if rank == 0:
    result = ExchangeResult.load(f"{folder}/{results_name}/ExchangeResult.npz")
    root_print(f"J = {result.J_meV:.3f} meV")