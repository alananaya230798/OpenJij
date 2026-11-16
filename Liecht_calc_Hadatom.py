
# -*- coding: utf-8 -*-
"""

----------------
Author: Alan Anaya
"""
import datetime
import sys
from pathlib import Path
import numpy as np
import sisl
from mpi4py import MPI
from J_dependencies import *
from exchange_mpi_io import *

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def root_print(*args, **kwargs):
    if rank == 0:
        print(*args, **kwargs, flush=True)

# =====================================================================
# 1. User configuration
# =====================================================================

# system
system_name="GNR"
write_results=True

# Home directory
HOME='/home/cluster_mount/J_GNRs/'

# Pristine Hamiltonian file
H_PRISTINE_FILE=HOME+'Elec/siesta/siesta.fdf'

# Transport directions
semi_L = "-A"
semi_R = "+A"
self_energy_mode="compute+write"
nk=(1, 0, 0)  
n_tile=1  #For tiling electrode for self energy calculation
tile_ax=0 # tiling axis for self energy calculation 

# Method
method='liechtenstein'
prefactor = -1.0 / (4.0 * np.pi) # ferro is negative anti is posiitive 
# Basis
default_basis= 'pz'

# Band alignement
align='vb'

# Defect configurations
perturbation="H"
downfold=False  
region='radius'
radius=7.5 # check 
tol=0.1

# Choose the two defect types
default_defect_i = "H0"
default_defect_j = "H0"

# Distance between defects in unit cells
default_n_dist=2

# Device
# if using pristine device: Device size 2*(N_imp+N_buferr)+ n_dist + 1
#  N_imp is the cell in the perturbation calculation where the perturbation is placed
N_imp = 4
N_buffer=0

# complex contour
cc_file = "/home/Magnetic_Interactions/CC_40eV.TSCCEQ-Elec"

# =====================================================================
# End of user configuration
# =====================================================================

# Retrieve defects
defect_i = sys.argv[1] if len(sys.argv) > 1 else default_defect_i
defect_j = sys.argv[2] if len(sys.argv) > 2 else default_defect_j
defects_lables=[defect_i,defect_j]

# Retrive n_dist
n_dist= int(sys.argv[3]) if len(sys.argv) > 1 else default_n_dist

# Retrive basis
basis= sys.argv[4] if len(sys.argv) > 4 else default_basis

# Folders
results_name=f"Results_{defect_i}_{defect_j}_n{n_dist}_R{radius:.2f}"
if downfold==True:
    folder=Path(f'{perturbation}_{system_name}_{basis}_{method}_downfold')
else:
    folder=Path(f'{perturbation}_{system_name}_{basis}_{method}')

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
my_energies = np.arange(rank, len(En), size)
print(f"Rank {rank}/{size}: received {len(En)} energies, calculates {len(my_energies)} energies", flush=True)

#Tile electrodes for satisfying nsc if necessary 
HamL=H_0.tile(n_tile,tile_ax).copy()
HamR=H_0.tile(n_tile,tile_ax).copy()

# ===============================================================
# 3.0 Device 
# ===============================================================
device_file=HOME+f"J_{defect_i}/n{n_dist}/{defect_j}/FM/siesta_FM.fdf" # Ferromagnetic as reference state
root_print(device_file)
H_dev=sisl.get_sile(device_file).read_hamiltonian()
# Original SIESTA atom indices
C_idx_original = find_C_with_outofplane_H(H_dev.geometry)
C_idx1, C_idx2 = C_idx_original

H_idx1 = H_dev.geometry.close(C_idx1, 1.2)[1]
H_idx2 = H_dev.geometry.close(C_idx2, 1.2)[1]
root_print(f"Carbon {C_idx1} has a H adatom with atom index {H_idx1} ")
root_print(f"Carbon {C_idx2} has a H adatom with atom index {H_idx2} ")

keep_orbitals={H_idx1:'all', H_idx2:'all'}
# Prune hamiltonian of the 2-Defects
H_dev, original_atoms = prepare_spin_basis(H_dev, basis='pz',keep_orbitals=keep_orbitals)
# Get pristnine carbon indices to match after prunning and H orbitals to keep
[C_idx1,C_idx2]=find_C_with_outofplane_H(H_dev.geometry)
# Pristine indexes to match
Pristine_idx = [int(np.flatnonzero(original_atoms == C_idx)[0])for C_idx in C_idx_original]

# allign device to electrode 
H, dev_eshift ,info = align_defect_to_electrodes(H_dev,H_0,axis=0)
root_print(dev_eshift)
H.set_nsc([1,1,1]) # remove pbc in transport direction x
na=H.na    
no=H.no

root_print(f"Number of atoms in device:{na}")
root_print(f"Number of orbitals in device:{no}")

# ===============================================================
# 4.0 Perturbations 
# ===============================================================
# Distance in the pristine host
distance=np.linalg.norm(H.geometry.Rij(Pristine_idx[0],Pristine_idx[1]))
root_print(f"Distance {distance:.3f} Å")

Delta_dir = folder / "Deltas"
if rank == 0:
    Delta_dir.mkdir(parents=True, exist_ok=True)
Delta_files = [Delta_dir / (f"Delta_{Pristine_idx[i]}_{defects_lables[0]}_{defects_lables[1]}_n{n_dist}__R{radius:.2f}.npz") for i in range(2)]

Deltas=[None,None]
orbs=[None,None]
mapped_c_orbs=[None,None]
hydrogens_data=[None,None]

for i,delta_file in enumerate(Delta_files):
    delta_data=None
    if rank == 0:
        if delta_file.is_file():
            root_print(f"Reading Delta {i}: {delta_file}")
            Delta, orb = load_local_delta(delta_file)
        else:            
            Delta, orb, _ = extract_single_Delta(
            H_host=H, # Tiled host Hamiltonian from tiled Hamiltonian
            H_polarized_file=device_file, # File for spin polarized perturbation
            host_anchor=Pristine_idx[i],     #  Host anchor atom after pruning host
            defect_anchor=C_idx_original[i], # C perturbation anchor before pruning
            basis=basis, # which orbitals to keep        
            tol=tol,  # tolerance for matching atoms
            keep_orbitals= keep_orbitals,      
            region =region,
            radius = radius)    
            root_print(f"Saving Delta {i}: {delta_file}")
            save_local_delta(delta_file, Delta, orb)
        # Broadcast the Delta matrix and mapped orbital indices
        root_print("Continuing w/o downfold")
        hydrogen_data = None
        mapped_c_orb = None
        delta_data = (Delta, np.asarray(orb, dtype=int),hydrogen_data,mapped_c_orb)
    Delta, orb,hydrogen_data,mapped_c_orb = comm.bcast(delta_data, root=0)
    Deltas[i] = Delta
    orbs[i] = np.asarray(orb, dtype=int)
    hydrogens_data[i]=hydrogen_data
    mapped_c_orbs[i]=mapped_c_orb
################################################################################################
# 2. Load the hydrogen information

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
root_print(f" Run started at {datetime.datetime.now()}")
root_print("========================================\n")

result = calculate_exchange(
    method=method,
    prefactor=prefactor,
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
    distance=distance,
    downfold=downfold,
    hydrogen_i=hydrogens_data[0],
    hydrogen_j=hydrogens_data[1],
    mapped_c_orbs_i=mapped_c_orbs[0],
    mapped_c_orbs_j=mapped_c_orbs[1],
)
comm.Barrier()

root_print("========================================")
root_print(f"Jk_calc.py run ended at {datetime.datetime.now()}")
root_print("========================================\n")
if rank == 0:
    result = ExchangeResult.load(f"{folder}/{results_name}/ExchangeResult.npz")
    root_print(f"J = {result.J_meV:.3f} meV")
    H.geometry.write(f"{folder}/{results_name}/device.fdf")
    geom = H.geometry
    geom.write(f"{folder}/{results_name}/device.fdf")
    orb_i = np.asarray(result.orb_i, dtype=int)
    orb_j = np.asarray(result.orb_j, dtype=int)

    # Orbital classification
    orbital_names_i = get_orbital_names(H, result.orb_i)
    orbital_names_j = get_orbital_names(H, result.orb_j)

    atoms_i = np.asarray(geom.o2a(orb_i), dtype=int)
    atoms_j = np.asarray(geom.o2a(orb_j), dtype=int)
    
    np.savez_compressed(
        folder / results_name/"ExchangeViz.npz",
        # Device geometry
        xyz=geom.xyz,
        cell=geom.cell,
        Z=geom.atoms.Z,
        # Magnetic regions
        orb_i=orb_i,
        orb_j=orb_j,
        atoms_i=atoms_i,
        atoms_j=atoms_j,
        # Orbital classification
        orbital_names_i=orbital_names_i,
        orbital_names_j=orbital_names_j,
        # Defect positions (atom indices)
        defect_atoms=np.array(Pristine_idx),
        # Metadata
        method=method,
        basis=basis,
    )



