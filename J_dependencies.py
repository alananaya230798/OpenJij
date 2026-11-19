import warnings
import numpy as np
import scipy as sp
from scipy.sparse import eye
import os
import os.path as osp
import sisl
from scipy.sparse.linalg import inv
from scipy.linalg import solve
import concurrent.futures
import re
from collections import defaultdict
from collections import defaultdict
from scipy.spatial import cKDTree
from dataclasses import dataclass
from typing import Any, Callable, Optional
from scipy.sparse.linalg import splu
from dataclasses import dataclass
from typing import Optional
from scipy import sparse
from scipy.sparse.linalg import splu

# ---------------------------------------------------------------------------
# Reading DFT Hamiltonians (one place, so every matrix gets the same sparsity)
# ---------------------------------------------------------------------------
SPARSITY_ATOL = 1e-6   # elements with |H_up|, |H_dn| (eV) and |S| all <= atol are removed


def set_sparsity_atol(atol):
    """Set the default tolerance of read_hamiltonian.
    0 removes only exact zeros; None removes nothing."""
    global SPARSITY_ATOL
    SPARSITY_ATOL = None if atol is None else float(atol)
    return SPARSITY_ATOL


def read_hamiltonian(file, atol="default"):
    """Read a SIESTA/TranSiesta Hamiltonian and drop negligible stored elements.

    sisl's eliminate_zeros removes a stored (i, j) element only if ALL components
    (every spin of H, in eV, and S) are <= atol. The same tolerance is used for every
    file (pristine, single-defect and two-defect), so differences such as
    V = H_def - H_0 contain no artificial entries larger than atol.
    atol: 'default' -> SPARSITY_ATOL, None -> keep the file as read.
    """
    atol = SPARSITY_ATOL if atol == "default" else atol
    H = sisl.get_sile(str(file)).read_hamiltonian()
    if atol is not None:
        H.eliminate_zeros(atol=float(atol))
    return H


def find_C_with_outofplane_H(struct, cutoff=1.2, z_thresh=0.3):
    """Find carbons bonded to H atoms that are displaced in z."""
    defect_idx = []
    for i, atom in enumerate(struct.atoms):
        if atom.symbol == "C":
            neighbors = struct.close(i, R=cutoff)
            for j in neighbors:
                if struct.atoms[j].symbol == "H":
                    dz = struct.xyz[j, 2] - struct.xyz[i, 2]
                    if abs(dz) > z_thresh:  # out of plane
                        defect_idx.append(i)
                        break
    return sorted(set(defect_idx))


def get_Delta_orbital_atoms(
    H,
    Delta,
    orbital="pz",
    orb_host=None,
    zeta=None,
):
    """
    Calculate orbital-resolved diagonal spin splitting per atom.

    Parameters
    ----------
    H : sisl.Hamiltonian
        Host Hamiltonian.

    Delta : scipy sparse matrix or ndarray
        Full host-space Delta, or compact Delta.

    orbital : str
        'pz1'    : first pz zeta
        'pz2'    : second pz zeta
        'pz'     : all pz
        'dxy'    : dxy
        'dyz'    : dyz
        'dz2'    : dz2
        'dxz'    : dxz
        'dx2-y2' : dx2-y2
        'dz'     : dyz + dz2 + dxz
        'd'      : all five d orbitals
        'pz+dz'  : pz + dyz + dz2 + dxz

    orb_host : ndarray, optional
        Host orbital indices corresponding to compact Delta.

    zeta : int, optional
        Select a particular zeta for orbital types other
        than pz1/pz2. None includes all zeta functions.

    Returns
    -------
    values : ndarray, shape (H.na,)
        Orbital-resolved diagonal spin splitting per atom.
    """

    orbital_types = {
        "s": {(0, 0)},
        "px+py": {(1, 1),(1,-1)},
        "pz": {(1, 0)},
        "dxy": {(2, -2)},
        "dyz": {(2, -1)},
        "dz2": {(2, 0)},
        "dxz": {(2, 1)},
        "dx2-y2": {(2, 2)},
        "dz": {(2, -1),(2,0), (2, 1)}, # dz2 important for adatoms
        "d": {(2, m) for m in range(-2, 3)},
    }
    diag = np.asarray(Delta.diagonal()).real
    if orb_host is not None:
        orb_host = np.asarray(orb_host, dtype=int)
        if len(diag) != len(orb_host):
            raise ValueError("Delta and orb_host mismatch")
        diag_host = np.zeros(H.no, dtype=float)
        diag_host[orb_host] = diag
    else:
        if len(diag) != H.no:
            raise ValueError("For compact Delta, provide orb_host")
        diag_host = diag
    tokens = orbital.lower().replace(" ", "").split("+")
    selection = []
    for token in tokens:
        if token in ("pz1", "pz2"):
            # pz with specified zeta
            selection.append((orbital_types["pz"], int(token[-1])))
        elif token in orbital_types:
            selection.append((orbital_types[token], zeta))
        else:
            raise ValueError(f"Unknown orbital: {token}")
    # 3. Calculate contribution per atom
    values = np.zeros(H.na)
    for ia in range(H.na):
        atom = H.atoms[ia]
        global_orbs = np.asarray( H.a2o(ia, all=True), dtype=int)
        selected = set()
        for symmetries, selected_zeta in selection:
            for io, orb in enumerate(atom.orbitals):
                if (orb.l, orb.m) not in symmetries:
                    continue
                orb_zeta = getattr(orb, "zeta",getattr(orb, "Z", None))
                if (selected_zeta is not None and orb_zeta != selected_zeta):
                    continue
                selected.add(io)
        if selected:
            local_idx = np.array(sorted(selected), dtype=int)
            values[ia] = np.sum(diag_host[global_orbs[local_idx]])
    return values

def prepare_spin_basis(H, basis="pz", remove_atoms=(), keep_orbitals=(), extra_basis="s"):
    """
    Prepare a spin-polarized Hamiltonian with the same basis convention as prune_ham.

    Parameters
    ----------
    H : sisl.Hamiltonian
        Spin-polarized Hamiltonian from SIESTA/TranSiesta.
    basis : str
        'pz1', 'pz', 'pz+dz', or 'full'.
    remove_atoms : sequence of int
        Original atom indices to remove (e.g., H atoms for RKKY/downfolding).
    keep_orbitals : sequence of int or dict
        Additional original atom indices to retain. A dict maps each index to
        an orbital label ('s', 'p', 'pz', 's+pz', 'all'); a sequence uses extra_basis.
    extra_basis : str
        Orbital label for extra atoms supplied as a sequence.

    Returns
    -------
    H_new : sisl.Hamiltonian
    original_indices : ndarray
        Original atom index for each atom in H_new (in the same order).
    """
    basis = basis.lower().replace(" ", "")
    valid = ("pz1", "pz", "pz+dz", "full")
    if basis not in valid:
        raise ValueError(f"Invalid basis: {basis}")

    # Original atoms to remove, including adatoms explicitly downfolded elsewhere.
    removed = np.asarray(list(remove_atoms), dtype=int)
    if np.any((removed < 0) | (removed >= H.na)):
        raise IndexError(f"Invalid atom index in remove_atoms. Valid range: 0 to {H.na - 1}")
    removed = set(removed.tolist())

    # NEW: optional extra atoms, indexed in the ORIGINAL SP Hamiltonian.
    if keep_orbitals is None:
        extra = {}
    elif isinstance(keep_orbitals, dict):
        extra = {int(ia): label.lower().replace(" ", "")
                 for ia, label in keep_orbitals.items()}
    else:
        indices = [int(ia) for ia in keep_orbitals]
        if len(indices) != len(set(indices)):
            raise ValueError("Duplicate atom indices in keep_orbitals")
        extra = {ia: extra_basis.lower().replace(" ", "") for ia in indices}

    valid_extra = ("s", "p", "pz", "s+pz", "all")
    base_atoms = set(np.flatnonzero(np.isin(H.atoms.Z, (5, 6, 7, 8, 9))).tolist())
    if basis == "full" and extra:
        # 'full' keeps every atom and orbital, adatoms included: keep_orbitals only
        # identifies atoms that are kept anyway. A partial label cannot be honoured.
        for ia, label in extra.items():
            if not 0 <= ia < H.na:
                raise IndexError(f"Invalid atom index in keep_orbitals: {ia}")
            if label not in valid_extra:
                raise ValueError(f"Invalid orbital selection '{label}' for atom {ia}")
            if ia in removed:
                raise ValueError(f"Atom {ia} is in both remove_atoms and keep_orbitals")
        partial = {ia: lab for ia, lab in extra.items() if lab != "all"}
        if partial:
            warnings.warn(f"basis='full' keeps all orbitals; keep_orbitals labels {partial} "
                          "are ignored")
        extra = {}
    for ia, label in extra.items():
        if not 0 <= ia < H.na:
            raise IndexError(f"Invalid atom index in keep_orbitals: {ia}")
        if label not in valid_extra:
            raise ValueError(f"Invalid orbital selection '{label}' for atom {ia}")
        if ia in removed:
            raise ValueError(f"Atom {ia} is in both remove_atoms and keep_orbitals")
        if basis == "full" or ia in base_atoms:
            raise ValueError(f"Atom {ia} is already selected by basis='{basis}'")

    # NEW: preserve the original atom ordering; never append extra atoms at the end.
    original_indices = np.array([
        ia for ia in range(H.na)
        if ia not in removed and
        (basis == "full" or ia in base_atoms or ia in extra)
    ], dtype=int)
    if original_indices.size == 0:
        raise ValueError("No atoms remain after selection")

    if basis == "full":
        H_new = H.copy() if not removed else H.sub(original_indices)
        return H_new, original_indices

    H_new = H.sub(original_indices)
    groups = defaultdict(list)
    expected_no = 0
    for ia in range(H_new.na):
        atom = H_new.atoms[ia]
        original_ia = int(original_indices[ia])
        pz = [io for io, orb in enumerate(atom.orbitals)
              if orb.l == 1 and orb.m == 0]
        dz = [io for io, orb in enumerate(atom.orbitals) # dz2 important for adatoms
              if orb.l == 2 and orb.m in (-1,0, 1)]

        # NEW: select orbitals on explicitly retained adatoms by label.
        if original_ia in extra:
            label = extra[original_ia]
            selected = [
                io for io, orb in enumerate(atom.orbitals)
                if label == "all"
                or (label == "s" and orb.l == 0)
                or (label == "p" and orb.l == 1)
                or (label == "pz" and orb.l == 1 and orb.m == 0)
                or (label == "s+pz" and
                    (orb.l == 0 or (orb.l == 1 and orb.m == 0)))
            ]
        elif basis == "pz1":
            first = [io for io in pz if getattr(
                atom.orbitals[io], "zeta", getattr(atom.orbitals[io], "Z", 1)
            ) == 1]
            selected = first[:1]
        elif basis == "pz":
            selected = pz
        elif basis == "pz+dz":
            selected = sorted(pz + dz)

        if not selected:
            raise ValueError(f"No matching orbitals on original atom {original_ia}")
        groups[tuple(selected)].append(ia)
        expected_no += len(selected)

    for orbital_indices, atoms in groups.items():
        H_new = H_new.sub_orbital(atoms=atoms, orbitals=list(orbital_indices))
    H_new.reduce()
    if H_new.no != expected_no:
        raise RuntimeError("Orbital selection failed")
    return H_new, original_indices

def map_spin_to_host(
    H_host,
    H_defect,
    host_anchor,
    defect_anchor,
    tol=0.1,
    allow_species_mismatch=False,):
    """
    Match defect atoms and orbitals to the host.

    Alignment is a translation in XY using the two anchors.
    Orbital order and orbital labels must match.

    Parameters
    ----------
    H_host : sisl.Hamiltonian
        Hamiltonian from SIESTA/TranSiesta.
    H_defect : sisl.Hamiltonian
        Spin-polarized Hamiltonian from SIESTA/TranSiesta.
    host_anchor : int
        Atomic index of host atom to be anchor
    defect_anchor: int
         Atomic index of defect atom to be anchor

    Returns
    -------
    atom_mapping : dict
        {defect_atom_index: pristine_atom_index}
    orb_mapping : dict
        {defect_orbital_index: pristine_orbital_index}
    """

    xyz_h = H_host.geometry.xyz
    xyz_d = H_defect.geometry.xyz

    if not (0 <= host_anchor < H_host.na):
        raise IndexError("Invalid host_anchor")
    if not (0 <= defect_anchor < H_defect.na):
        raise IndexError("Invalid defect_anchor")

    shift = xyz_h[host_anchor, :2] - xyz_d[defect_anchor, :2]
    tree = cKDTree(xyz_h[:, :2])
    atom_map = {}
    orbital_map = {}
    used_host = set()

    # Anchor first, then all remaining atoms
    order = [defect_anchor] + [ ia for ia in range(H_defect.na) if ia != defect_anchor ]
    for d in order:
        if d == defect_anchor:
            h = host_anchor
        else:
            pos = xyz_d[d, :2] + shift
            candidates = tree.query_ball_point(pos, tol)
            candidates = [ h for h in candidates if h not in used_host]
            if not allow_species_mismatch:
                candidates = [  h for h in candidates if H_host.atoms.Z[h] == H_defect.atoms.Z[d]]
            if not candidates:
                raise ValueError( f"Could not map defect atom {d}. Check geometry, species, tolerance or removed adatoms.")
            h = min(candidates,key=lambda j: np.linalg.norm( xyz_h[j, :2] - pos))
        if h in used_host:
            raise ValueError(f"Host atom {h} mapped twice")
        if (not allow_species_mismatch and H_host.atoms.Z[h] != H_defect.atoms.Z[d]):
            raise ValueError(f"Species mismatch: defect atom {d} Z={H_defect.atoms.Z[d]} host atom {h} Z={H_host.atoms.Z[h]}")
        # Compare orbital labels, not just orbital counts
        def_names = [ orb.name()  for orb in H_defect.atoms[d].orbitals]
        host_names = [orb.name()  for orb in H_host.atoms[h].orbitals]
        if def_names != host_names:
            raise ValueError( f"Orbital basis mismatch at defect {d},host {h}:\n{def_names}\n{host_names}" )
        atom_map[d] = h
        used_host.add(h)
        od = np.asarray( H_defect.a2o(d, all=True), dtype=int)
        oh = np.asarray( H_host.a2o(h, all=True), dtype=int)
        for a, b in zip(od, oh):
            orbital_map[int(a)] = int(b)
    return atom_map, orbital_map

def _square_region_atoms(xyz, anchor, half_side):
    """Atom indices inside an axis-aligned xy square centred on anchor.

    ``half_side`` is the distance from the centre to each edge, in angstrom.
    All z coordinates are ignored (works for planar sheets with adatoms).
    """
    if half_side is None or half_side <= 0:
        raise ValueError("radius (square half-side) must be positive in Angstrom")
    xyz = np.asarray(xyz)
    delta_xy = np.abs(xyz[:, :2] - xyz[int(anchor), :2])
    return np.flatnonzero(np.all(delta_xy <= half_side, axis=1))


def extract_single_Delta(
    H_host,
    H_polarized_file,
    host_anchor,
    defect_anchor,
    basis="pz",
    region="all",
    tol=0.1,
    adatom_atoms=(),
    allow_species_mismatch=False,
    Delta_file=None,
    return_full=False,
    calculation_folder=None,
    keep_orbitals=(),        # NEW: retained adatom atoms (original indices)
    extra_basis="s",         # NEW: orbital label for listed adatoms
    radius=3.5,             # Angstrom; radius or square half-side
    atom_mask=None,         # NEW: original SP atom indices for region='atom_mask'
):
    """
    Extract Delta = H_up - H_down and map it to host orbitals.

    The structural mapping and tolerance checks are unchanged.
    Magnetic region options:
        'all'       : all orbitals of the prepared SP Hamiltonian
        'radius'    : orbitals on atoms within radius of defect_anchor
        'square'    : atoms inside xy square centred on defect_anchor
        'atom_mask' : orbitals on original SP atoms listed in atom_mask

    keep_orbitals and adatom_atoms use ORIGINAL SP atom indices.
    The radius is centered on defect_anchor, which is usually the carbon
    bonded to H. The mask also uses ORIGINAL SP atom indices.
    """
    region = region.lower()
    if region not in ("all", "radius", "square", "atom_mask"):
        raise ValueError("region must be 'all', 'radius', 'square', or 'atom_mask'")

    # 1. Read the spin-polarized Hamiltonian
    H_sp = read_hamiltonian(H_polarized_file)
    if not H_sp.spin.is_polarized:
        raise ValueError("A collinear spin-polarized Hamiltonian is required")

    # 2. Apply basis selection (NEW: optionally retain adatom orbitals)
    H_def, original_atoms = prepare_spin_basis(
        H_sp, basis=basis, remove_atoms=adatom_atoms,
        keep_orbitals=keep_orbitals, extra_basis=extra_basis,
    )
    positions = np.flatnonzero(original_atoms == defect_anchor)
    if len(positions) != 1:
        raise ValueError("defect_anchor was removed by basis selection")
    defect_anchor_new = int(positions[0])

    # 3. Map ALL retained atoms/orbitals to the host as before.
    atom_map, orbital_map = map_spin_to_host(
        H_host, H_def,
        host_anchor=host_anchor,
        defect_anchor=defect_anchor_new,
        tol=tol,
        allow_species_mismatch=allow_species_mismatch,
    )

    # 4. Compute real-space, central-cell Delta (unchanged)
    H_local = H_def.copy()
    H_local.set_nsc([1, 1, 1])
    H_up = H_local.Hk(spin=0, format="csr")
    H_down = H_local.Hk(spin=1, format="csr")
    D = (H_up - H_down).tocsr()
    D.eliminate_zeros()

    # 5. Per-atom magnetic strength (unchanged; retained for diagnostics)
    scores = np.zeros(H_def.na)
    for ia in range(H_def.na):
        orb = np.asarray(H_def.a2o(ia, all=True), dtype=int)
        block = D[orb, :][:, orb].toarray()
        scores[ia] = np.linalg.norm(block, ord="fro")

    # 6. NEW: choose magnetic region without changing matrix/orbital ordering.
    if region == "all":
        selected_atoms = np.arange(H_def.na)
    elif region == "radius":
        if radius is None or radius <= 0:
            raise ValueError("radius must be a positive distance in Angstrom")
        xyz = np.asarray(H_def.geometry.xyz)
        dist = np.linalg.norm(xyz - xyz[defect_anchor_new], axis=1)
        selected_atoms = np.flatnonzero(dist <= radius)
    elif region == "square":
        selected_atoms = _square_region_atoms(H_def.geometry.xyz, defect_anchor_new, radius)
    else:  # region == 'atom_mask'
        if atom_mask is None:
            raise ValueError("atom_mask must be provided for region='atom_mask'")
        mask = np.asarray(atom_mask, dtype=int).ravel()
        if np.any((mask < 0) | (mask >= H_sp.na)):
            raise IndexError("atom_mask contains indices outside the original SP Hamiltonian")
        missing = np.setdiff1d(mask, original_atoms)
        if missing.size:
            raise ValueError(
                f"atom_mask includes atoms removed by basis selection: {missing.tolist()}"
            )
        selected_atoms = np.flatnonzero(np.isin(original_atoms, mask))

    if len(selected_atoms) == 0:
        raise ValueError("No magnetic atoms selected")

    # 7. Orbital selection and mapping (unchanged)
    orb_def = np.concatenate([
        np.asarray(H_def.a2o(int(a), all=True), dtype=int)
        for a in selected_atoms
    ])
    orb_host = np.array([orbital_map[int(o)] for o in orb_def], dtype=int)
    Delta = D[orb_def, :][:, orb_def].tocsr()

    # 8. Sparse embedding into the complete host space (unchanged)
    Delta_full = None
    if return_full:
        D_coo = Delta.tocoo()
        Delta_full = sparse.coo_matrix(
            (D_coo.data, (orb_host[D_coo.row], orb_host[D_coo.col])),
            shape=(H_host.no, H_host.no),
        ).tocsr()

    # 9. Save compact result (unchanged)
    if calculation_folder is not None and Delta_file is None:
        Delta_file = os.path.join(  calculation_folder, "Deltas", f"Delta.npz" )
    if Delta_file is not None:
        from exchange_mpi_io import save_local_delta
        save_local_delta(
            Delta_file, Delta, orb_host,
            atom_idx=np.array([atom_map[int(a)] for a in selected_atoms], dtype=int),
            defect_atom_idx=original_atoms[selected_atoms],
            scores=scores, basis=basis, region=region)

    # 10. Report
    print(f"\nBasis: {basis}")
    print(f"Region: {region}")
    if region == "radius":
        print(f"Radius: {radius:.3f} Angstrom")
    elif region == "square":
        print(f"Square half-side: {radius:.3f} Angstrom")
    print(f"Total defect atoms: {H_def.na}")
    print(f"Selected magnetic atoms: {len(selected_atoms)}")
    print(f"Selected orbitals: {len(orb_host)}")
    print(f"Delta shape: {Delta.shape}")
    print(f"Max |Delta|: {np.max(np.abs(Delta)):.6f} eV")
    
    return Delta, orb_host, Delta_full


def read_transiesta_contour(cc_file="TSCCEQ"):
    """
    Reads TranSiesta equilibrium contour file (TSCCEQ).

    Returns
    -------
    E_real, E_imag, W_real, W_imag : arrays
    E, W : complex arrays
    mu : float
        Chemical potential [eV].
    T : float
        Electronic temperature [K].
    """

    E_real, E_imag, W_real, W_imag = [], [], [], []
    mu, T = None, None

    read_mu, read_T = False, False

    with open(cc_file, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            # Chemical potential label
            if line.startswith("#") and "Chemical potential:" in line:
                read_mu = True
                read_T = False
                continue

            # Electronic temperature label
            if line.startswith("#") and "Electronic temperature:" in line:
                read_T = True
                read_mu = False
                continue

            # Read chemical potential from the next commented line
            if read_mu:
                if not line.startswith("#"):
                    raise ValueError("Missing chemical potential in TSCCEQ")

                parts = line.lstrip("#").split()
                mu = float("".join(parts[:-1]))  # handles '+  0.000'
                read_mu = False
                continue

            # Read temperature from the next commented line
            if read_T:
                if not line.startswith("#"):
                    raise ValueError("Missing electronic temperature in TSCCEQ")

                parts = line.lstrip("#").split()
                T = float(parts[0])
                read_T = False
                continue

            # Skip remaining comment lines
            if line.startswith("#"):
                continue

            # Read contour energies and weights
            parts = line.split()

            if len(parts) >= 4:
                E_real.append(float(parts[0]))
                E_imag.append(float(parts[1]))
                W_real.append(float(parts[2]))
                W_imag.append(float(parts[3]))

    # Build arrays
    E_real = np.asarray(E_real)
    E_imag = np.asarray(E_imag)
    W_real = np.asarray(W_real)
    W_imag = np.asarray(W_imag)

    E = E_real + 1j * E_imag
    W = W_real + 1j * W_imag

    return E_real, E_imag, W_real, W_imag, E, W, mu, T


# updated version of prune hamiltonian

def prune_ham(
    H_PRISTINE_FILE,
    basis="pz",
    align="none",
    shift_eV=0.0,
    verbose=True,
    extra_atoms=None,      
    extra_basis="s",       #
):
    """
    Read and project a DFT Hamiltonian.

    Parameters
    ----------
    basis : str
        'pz1'    : first pz orbital (zeta=1)
        'pz'     : all pz orbitals
        'pz+dz'  : all pz, dxz, dyz, dz2 orbitals
        'full'   : retain all atoms and orbitals

    align : str
        'none'   : no energy shift
        'manual' : apply shift_eV
        'VB'     : align the valence band
        'CB'     : align the conduction band

    shift_eV : float
        Energy shift for align='manual'.

    extra_atoms : list, dict or None
        Atom indices of additional atoms to retain.
        A list uses extra_basis for all atoms.
        A dict maps each atom index to its orbital selection.
        Ex: extra_atoms={1: "s", 2: "s+pz"}

    extra_basis : str
        's'    : all s orbitals
        'p'    : all px, py, pz orbitals
        'pz'   : all pz orbitals
        's+pz' : all s and pz orbitals
        'all'  : all orbitals

    Returns
    -------
    H_0 : sisl.Hamiltonian
        Selected and optionally shifted Hamiltonian.
    H_0_DFT : sisl.Hamiltonian
        Original Hamiltonian.
    eshift : float
        Energy shift applied to H_0, in eV.
    """

    # 1. Read Hamiltonian
    H_0_DFT = read_hamiltonian(H_PRISTINE_FILE)
    basis = basis.lower().replace(" ", "")
    align = align.lower()
    valid_basis = ("pz1", "pz", "pz+dz", "full")
    valid_align = ("none", "manual", "vb", "cb")
    if basis not in valid_basis:
        raise ValueError(f"basis must be one of {valid_basis}")
    if align not in valid_align:
        raise ValueError(f"align must be one of {valid_align}")
    # 2. Select atoms and orbitals
    elements = {5: "B", 6: "C", 7: "N", 8: "O", 9: "F"}

    if extra_atoms is None:
        extra_selection = {}
    elif isinstance(extra_atoms, dict):
        extra_selection = {int(ia): label.lower().replace(" ", "") for ia, label in extra_atoms.items()}
    else:
        indices = np.atleast_1d(extra_atoms).tolist()
        # Reject repeated indices
        if len(indices) != len(set(indices)):
            raise ValueError("extra_atoms contains duplicate atom indices.")

        extra_selection = { int(ia): extra_basis.lower().replace(" ", "")for ia in indices}

    # Validate additional atoms and orbital labels
    valid_extra_basis = ("s", "p", "pz", "s+pz", "all")

    for ia, label in extra_selection.items():
        if ia < 0 or ia >= H_0_DFT.na:
            raise ValueError(f"Invalid extra atom index: {ia}")
        if label not in valid_extra_basis:
            raise ValueError(f"Invalid orbital selection '{label}' for atom {ia}. Choose from {valid_extra_basis}")
    if basis == "full":
        # No atom or orbital pruning
        H_0 = H_0_DFT.copy()
        #  all atoms are already included
        if extra_selection:
            raise ValueError("extra_atoms cannot be used with basis='full' all atoms and orbitals are already retained.")
    else:
        keep_atoms = np.flatnonzero( np.isin(H_0_DFT.atoms.Z, list(elements)))

        # NEW: reject extra atoms already retained by the base selection
        overlap = np.intersect1d(keep_atoms, list(extra_selection))
        if len(overlap):
            raise ValueError( f"Additional atom indices already included in the base selection: {overlap.tolist()}")
        # reserve the original atomic ordering
        keep_atoms = np.sort(np.concatenate((keep_atoms, np.array(list(extra_selection), dtype=int))))

        if len(keep_atoms) == 0:
            raise ValueError("No atoms selected.")
        H_0 = H_0_DFT.sub(keep_atoms)
        # Group atoms with identical selected local orbital indices
        orbital_groups = defaultdict(list)
        expected_no = 0
        for ia in range(H_0.na):
            atom = H_0.atoms[ia]
            # Retrieve the original atom index
            original_ia = int(keep_atoms[ia])
            # pz : l=1, m=0
            pz_idx = [io for io, orb in enumerate(atom.orbitals)if orb.l == 1 and orb.m == 0]
            # dz : dxz, dyz, dz2
            # l=2, m=-1, 0, +1
            dz_idx = [io for io, orb in enumerate(atom.orbitals)if orb.l == 2 and orb.m in (-1,0, 1)] # dz2 important for adatoms
            # Additional atoms use their own orbital selection
            if original_ia in extra_selection:
                label = extra_selection[original_ia]
                selected = [
                    io for io, orb in enumerate(atom.orbitals)
                    if (
                        label == "all"
                        or (label == "s" and orb.l == 0)
                        or (label == "p" and orb.l == 1)
                        or (label == "pz" and orb.l == 1 and orb.m == 0)
                        or (
                            label == "s+pz"
                            and (
                                orb.l == 0
                                or (orb.l == 1 and orb.m == 0)
                            )
                        )
                    )
                ]
            elif basis == "pz1":
                first_pz = [io for io in pz_idx if getattr(atom.orbitals[io], "zeta",getattr(atom.orbitals[io], "Z", None)) == 1]
                if not first_pz:
                    raise ValueError( f"No zeta=1 pz orbital for atom {ia}")
                # Prefer the non-polarization orbital
                first_pz.sort( key=lambda io: bool(getattr(atom.orbitals[io], "P", False)))
                selected = [first_pz[0]]
            elif basis == "pz":
                selected = pz_idx
            elif basis == "pz+dz":
                selected = sorted(pz_idx + dz_idx)
            if not selected:
                raise ValueError(f"No matching orbitals for atom {original_ia} Z={atom.Z}")
            orbital_groups[tuple(selected)].append(ia)
            expected_no += len(selected)
        # Retain orbitals with the same local indices in one operation
        for orb_indices, atom_indices in orbital_groups.items():
            H_0 = H_0.sub_orbital(atoms=atom_indices,orbitals=list(orb_indices),)
        H_0.reduce()
        if H_0.no != expected_no:
            raise RuntimeError(f"Expected {expected_no} orbitals, got {H_0.no}")
    # 3. Energy alignment (unchanged)
    eshift = 0.0
    if align == "none":
        print("No shift applied")
    elif align == "manual":
        eshift = float(shift_eV)
    elif align == "vb":
        tmp = H_0_DFT.eigh()
        en_dft = tmp[tmp < 0].max()
        tmp = H_0.eigh()
        en_tb = tmp[tmp < 0].max()
        eshift = en_dft - en_tb
    elif align == "cb":
        tmp = H_0_DFT.eigh()
        en_dft = tmp[tmp > 0].min()
        tmp = H_0.eigh()
        en_tb = tmp[tmp > 0].min()
        eshift = en_dft - en_tb
    # Apply shift correctly, including overlap matrix
    H_0.shift(eshift)

    # 4. Report
    if verbose:
        print(f"\nBasis selection: {basis}")
        print(f"Alignment: {align}")
        print(f"Energy shift: {eshift:.6f} eV")
        # Show additional atom selections
        if extra_selection:
            print(f"Additional atoms (original indices): {extra_selection}")
        # Report elements of additional atoms
        report_elements = {
            **elements,
            **{
                int(Z): str(H_0.atoms[np.where(H_0.atoms.Z == Z)[0][0]].Z)
                for Z in np.unique(H_0.atoms.Z)
                if int(Z) not in elements
            }
        }

        for Z, symbol in report_elements.items():
            idx = np.where(H_0.atoms.Z == Z)[0]
            if len(idx) == 0:
                continue
            atom = H_0.atoms[int(idx[0])]
            print(f"\n{symbol}: {len(idx)} atoms")
            print("Orbitals on representative atom:")
            for io, orb in enumerate(atom.orbitals):
                print(f"  {io}: {orb.name()}")
        print("\nFinal Hamiltonian")
        print(f"Atoms: {H_0.na}")
        print(f"Orbitals: {H_0.no}")
    return H_0, H_0_DFT, eshift

ORBITAL_CLASSES = ("C-pz", "H", "d", "other")


def orbital_class_labels(geometry, orbitals):
    """Class index of every orbital: 0 C-pz (pz of a heavy atom), 1 H (any orbital
    of an H atom), 2 d, 3 other. Returns (labels, names of the classes present)."""
    orbitals = np.asarray(orbitals, dtype=int)
    atoms = geometry.o2a(orbitals)
    labels = np.full(len(orbitals), 3, dtype=int)
    for k, (o, a) in enumerate(zip(orbitals, atoms)):
        a = int(a)
        orb = geometry.atoms[a].orbitals[int(o - geometry.firsto[a])]
        name = orb.name()
        if geometry.atoms.Z[a] == 1:
            labels[k] = 1
        elif "pz" in name:
            labels[k] = 0
        elif getattr(orb, "l", -1) == 2 or name.lstrip("0123456789").startswith("d"):
            labels[k] = 2
    present = [c for c in range(len(ORBITAL_CLASSES)) if np.any(labels == c)]
    remap = {c: i for i, c in enumerate(present)}
    return np.array([remap[c] for c in labels], dtype=int), tuple(ORBITAL_CLASSES[c] for c in present)


def class_sums(contribution, labels_i, labels_j, n_i, n_j):
    """Sum an orbital-pair matrix into orbital-class blocks."""
    out = np.zeros((n_i, n_j), dtype=complex)
    for a in range(n_i):
        rows = contribution[labels_i == a]
        for b in range(n_j):
            out[a, b] = rows[:, labels_j == b].sum()
    return out


@dataclass
class ExchangeResult:
    method: str
    J_meV: float
    J_k_meV: np.ndarray
    kpoints: np.ndarray
    k_weights: np.ndarray
    energies: np.ndarray
    contour_weights: np.ndarray
    weighted_kernel_kE: np.ndarray  # prefactor * W(E) * trace, not yet Im or x 1000
    J_orb_meV: Optional[np.ndarray]  # shape (len(orb_i), len(orb_j)); sums to J_meV
    prefactor: complex
    spin_average: bool  # Legacy output flag: new Liechtenstein uses one up/down kernel
    orb_i: np.ndarray
    orb_j: np.ndarray
    weighted_orbital_kernel_kE: Optional[np.ndarray] = None
    distance: Optional[float] = None
    # Per-energy kernel summed over pairs of orbital classes (C-pz, H, d, other):
    # shape (nk, nE, n_classes_i, n_classes_j); names in class_names_i / _j.
    weighted_class_kernel_kE: Optional[np.ndarray] = None
    class_names_i: Optional[tuple] = None
    class_names_j: Optional[tuple] = None

    def save(self, filename):
        """Store numeric results without pickled objects."""
        data = {
            "method": self.method,
            "J_meV": self.J_meV,
            "J_k_meV": self.J_k_meV,
            "kpoints": self.kpoints,
            "k_weights": self.k_weights,
            "energies": self.energies,
            "contour_weights": self.contour_weights,
            "weighted_kernel_kE": self.weighted_kernel_kE,
            "prefactor": self.prefactor,
            "spin_average": self.spin_average,
            "orb_i": self.orb_i,
            "orb_j": self.orb_j,
        }
        if self.J_orb_meV is not None:
            data["J_orb_meV"] = self.J_orb_meV
        if isinstance(self.weighted_orbital_kernel_kE, np.memmap):
            # Keep the large orbital kernel as a separate memory-mapped .npy file.
            data["weighted_orbital_kernel_external"] = os.path.basename(
                self.weighted_orbital_kernel_kE.filename)
        elif self.weighted_orbital_kernel_kE is not None:
            data["weighted_orbital_kernel_kE"] = self.weighted_orbital_kernel_kE
        if self.distance is not None:
            data["distance"] = float(self.distance)
        if self.weighted_class_kernel_kE is not None:
            data["weighted_class_kernel_kE"] = self.weighted_class_kernel_kE
            data["class_names_i"] = np.asarray(self.class_names_i, dtype="U16")
            data["class_names_j"] = np.asarray(self.class_names_j, dtype="U16")
        np.savez_compressed(filename, **data)

    @classmethod
    def load(cls, filename):
        """Read an exchange result saved in the compressed NumPy format."""
        with np.load(filename, allow_pickle=False) as f:
            orbital = f["J_orb_meV"].copy() if "J_orb_meV" in f.files else None
            orb_kernels = (f["weighted_orbital_kernel_kE"].copy() if "weighted_orbital_kernel_kE" in f.files  else None)
            if "weighted_orbital_kernel_external" in f.files:
                orb_path = os.path.join(os.path.dirname(filename),
                                        str(f["weighted_orbital_kernel_external"].item()))
                orb_kernels = np.load(orb_path, mmap_mode="r", allow_pickle=False)
            return cls(
                method=str(f["method"].item()),
                J_meV=float(f["J_meV"].item()),
                J_k_meV=f["J_k_meV"].copy(),
                kpoints=f["kpoints"].copy(),
                k_weights=f["k_weights"].copy(),
                energies=f["energies"].copy(),
                contour_weights=f["contour_weights"].copy(),
                weighted_kernel_kE=f["weighted_kernel_kE"].copy(),
                weighted_orbital_kernel_kE=orb_kernels,
                J_orb_meV=orbital,
                prefactor=complex(f["prefactor"].item()),
                spin_average=bool(f["spin_average"].item()),
                orb_i=f["orb_i"].copy(),
                orb_j=f["orb_j"].copy(),
                distance=(float(f["distance"].item())
                          if "distance" in f.files and f["distance"].shape == ()
                          and f["distance"].dtype.kind in "fi" else None),
                weighted_class_kernel_kE=(f["weighted_class_kernel_kE"].copy()
                                          if "weighted_class_kernel_kE" in f.files else None),
                class_names_i=(tuple(str(x) for x in f["class_names_i"])
                               if "class_names_i" in f.files else None),
                class_names_j=(tuple(str(x) for x in f["class_names_j"])
                               if "class_names_j" in f.files else None),
            )


def monkhorst_pack_grid(H_ref, nk=(1, 1, 1), axes=("x", "y", "z"), trs=False):
    """Construct an sisl Monkhorst-Pack grid and normalized integration weights.
    Parameters
    ----------
    H_ref : sisl.Hamiltonian
        Parent for the Brillouin-zone sampling.
    nk : sequence of integers
        Grid counts. A zero means no sampling on that axis (Gamma only),
        and is converted to 1 for sisl. E.g. (15, 9, 0) -> (15, 9, 1).
    axes : sequence of strings
        Labels assigned to the entries of nk. For example, axes=('y','x','z')
        swaps the first two entries. Default is Cartesian/reduced x,y,z order.
    trs : bool
        Time-reversal symmetry reduction. False is the safe default for
        arbitrary spin-polarized or symmetry-broken calculations.

    Returns
    -------
    kpoints : ndarray, shape (N, 3)
        Reduced reciprocal coordinates used by sisl.Hk/Sk.
    weights : ndarray, shape (N,)
        Normalized Brillouin-zone quadrature weights.

    Notes
    -----
    A zero in nk is NOT a zero k-point; it requests Gamma-only sampling.
    """
    nk = tuple(int(v) for v in nk)
    axes = tuple(str(v).lower() for v in axes)
    if len(nk) != 3 or len(axes) != 3 or set(axes) != {'x', 'y', 'z'}:
        raise ValueError("nk and axes must have 3 entries; axes must be a permutation of x,y,z")
    if any(v < 0 for v in nk):
        raise ValueError("nk entries must be nonnegative")
    counts = [1, 1, 1]
    for count, axis in zip(nk, axes):
        counts['xyz'.index(axis)] = max(1, count)
    bz = sisl.MonkhorstPack(H_ref, counts, trs=trs)
    kpoints = np.asarray(bz.k, dtype=float)
    weights = np.asarray(bz.weight, dtype=float)
    if kpoints.ndim != 2 or kpoints.shape[1] != 3 or weights.shape != (len(kpoints),):
        raise ValueError("Unexpected MonkhorstPack k/weight shapes")
    if not np.all(np.isfinite(weights)) or weights.sum() <= 0:
        raise ValueError("Invalid MonkhorstPack weights")
    return kpoints, weights / weights.sum()


def recursive_leads_from_hamiltonians(
    H_device, 
    HamL,
    semi_L,
    HamR,
    semi_R,
    left_orbs=None,
    right_orbs=None,
    eta=1e-5,
    polarized_leads=False,
):
    """
    Construct left/right RecursiveSI self-energies.
    Parameters
    ----------
    H_device : sisl.Hamiltonian
        Device Hamiltonian.
    HamL, HamR : sisl.Hamiltonian
        Left and right electrode Hamiltonians.
    semi_L, semi_R : str
        Electrode semi-infinite directions, e.g. "-B", "+B".
    left_orbs, right_orbs : array-like or None
        Device orbitals coupled to each electrode.
        If None, use the original convention:
            left  -> first nl device orbitals
            right -> last nr device orbitals
        where nl, nr are the dimensions of the corresponding
        RecursiveSI self-energy matrices.
    eta : float
        Broadening parameter for RecursiveSI.
    polarized_leads : bool
        Whether electrodes are spin polarized.
    Returns
    -------
    provider : callable(z, k, spin)
        Returns [(left_indices, SigmaL), (right_indices, SigmaR)].
    """
    # Construct RecursiveSI only once
    RSI_L = sisl.RecursiveSI(HamL, semi_L, eta=eta)
    RSI_R = sisl.RecursiveSI(HamR, semi_R, eta=eta)
    # Total number of device orbitals
    no = H_device.no
    def provider(z, k, spin=None):
        # Spin is only needed for magnetic electrodes
        # NEW: only pass a spin index to electrodes that have two channels.
        kwL = {"spin": (0 if spin is None else int(spin))} if polarized_leads and HamL.spin.is_polarized else {}
        kwR = {"spin": (0 if spin is None else int(spin))} if polarized_leads and HamR.spin.is_polarized else {}
        # Calculate electrode self-energies at energy z and k-point
        SigmaL = RSI_L.self_energy(z, k=k, **kwL)
        SigmaR = RSI_R.self_energy(z, k=k, **kwR)
        # Number of orbitals in each electrode coupling block
        nl = SigmaL.shape[0]
        nr = SigmaR.shape[0]
        #  First/last block convention by default
        il = (np.arange(nl) if left_orbs is None else np.asarray(left_orbs, dtype=int))
        ir = ( np.arange(no - nr, no) if right_orbs is None else np.asarray(right_orbs, dtype=int))
        # Ensure that the embedding dimensions agree
        if len(il) != nl or len(ir) != nr:
            raise ValueError("Electrode orbital indices do not match self-energy dimensions")
        return [(il, SigmaL), (ir, SigmaR)]
    return provider

def check_electrode_interfaces(H_device, HamL, HamR, left_orbs=None,
                               right_orbs=None, tol=0.2):
    """Check that the device terminal orbitals match the two electrode cells.

    The orbital arrays must be in electrode order. No atoms are reordered or
    appended to the device; a missing/incompatible terminal cell raises an error.
    Returns (left_orbs, right_orbs) in device indexing.
    """
    nl, nr = HamL.no, HamR.no
    li = (np.arange(nl, dtype=int) if left_orbs is None
          else np.asarray(left_orbs, dtype=int))
    ri = (np.arange(H_device.no - nr, H_device.no, dtype=int)
          if right_orbs is None else np.asarray(right_orbs, dtype=int))

    if li.shape != (nl,) or ri.shape != (nr,):
        raise ValueError("Electrode interfaces must have the electrode orbital counts")
    if (np.any(li < 0) or np.any(ri < 0)
            or np.any(li >= H_device.no) or np.any(ri >= H_device.no)
            or len(np.unique(li)) != len(li) or len(np.unique(ri)) != len(ri)
            or np.intersect1d(li, ri).size):
        raise ValueError("Invalid or overlapping left/right electrode orbitals")

    def orbital_identity(H, io):
        ia = int(H.geometry.o2a(int(io)))
        local = int(io) - int(H.geometry.a2o(ia))
        atom = H.atoms[ia]
        return int(atom.Z), atom.orbitals[local].name(), H.geometry.xyz[ia]

    for side, elec, device_orbs in (("left", HamL, li), ("right", HamR, ri)):
        xyz_dev, xyz_elec = [], []
        for io, jo in enumerate(device_orbs):
            z_ref, name_ref, r_ref = orbital_identity(elec, io)
            z_dev, name_dev, r_dev = orbital_identity(H_device, jo)
            if z_ref != z_dev or name_ref != name_dev:
                raise ValueError(
                    f"{side} interface orbital {io}: electrode ({z_ref}, {name_ref}) "
                    f"!= device ({z_dev}, {name_dev}) at device orbital {jo}"
                )
            xyz_elec.append(r_ref)
            xyz_dev.append(r_dev)

        xyz_elec, xyz_dev = np.asarray(xyz_elec), np.asarray(xyz_dev)
        # Allow a rigid translation, but not reordered/distorted terminal atoms.
        delta = xyz_dev - xyz_elec
        mismatch = np.linalg.norm(delta - delta.mean(axis=0), axis=1).max()
        if mismatch > tol:
            raise ValueError(
                f"{side} electrode/device positions differ by {mismatch:.4f} Ang "
                f"after translation (tolerance {tol:.4f} Ang)"
            )
    return li, ri


def _lead_embedding(n, terms):
    """Sparse device-sized sum of lead self-energies, from orbital-index blocks."""
    if not terms:
        return sparse.csc_matrix((n, n), dtype=complex)
    rows, cols, vals = [], [], []
    for idx, sigma in terms:
        idx = np.asarray(idx, dtype=int)
        if idx.ndim != 1 or len(set(idx.tolist())) != len(idx):
            raise ValueError("Lead orbital indices must be unique 1D indices")
        if idx.size and (idx.min() < 0 or idx.max() >= n):
            raise IndexError("Lead orbital indices outside the device")
        if sparse.issparse(sigma):
            part = sigma.tocoo()
        else:
            part = sparse.coo_matrix(np.asarray(sigma, dtype=complex))
        if part.shape != (len(idx), len(idx)):
            raise ValueError("Lead self-energy shape does not match orbital indices")
        rows.append(idx[part.row])
        cols.append(idx[part.col])
        vals.append(part.data)
    return sparse.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n), dtype=complex
    ).tocsc()


def _selected_product(lu, target, source, local_delta, rhs_batch=32):
    """Compute [A^{-1}]_(target,source) @ local_delta without full inverse.
    Uses only n-by-batch dense RHS and one small dense output. Note SuperLU
    factors themselves may fill in and can still consume substantial memory.
    """
    n = lu.shape[0]
    nt, ns = len(target), len(source)
    delta = local_delta.tocsc()
    out = np.empty((nt, ns), dtype=np.complex128)
    for first in range(0, ns, rhs_batch):
        last = min(ns, first + rhs_batch)
        rhs = np.zeros((n, last - first), dtype=np.complex128)
        rhs[source, :] = delta[:, first:last].toarray()
        out[:, first:last] = lu.solve(rhs)[target, :]
    return out

def _validate_vertex(Delta, orbitals, n, label):
    orb = np.asarray(orbitals, dtype=int)
    if orb.ndim != 1 or not len(orb):
        raise ValueError(f"{label}: selected orbitals must be a nonempty 1D array")
    if np.any(orb < 0) or np.any(orb >= n) or len(np.unique(orb)) != len(orb):
        raise ValueError(f"{label}: duplicate or invalid orbital indices")
    D = sparse.csr_matrix(Delta, dtype=np.complex128)
    if D.shape != (len(orb), len(orb)):
        raise ValueError(f"{label}: Delta {D.shape} incompatible with {len(orb)} orbitals")
    return D, orb


def prepare_exchange_vertex(Delta, orb, downfold=None, mapped_c_orbs=None):
    """Embed static perturbation and optionally attach H downfolding data.

    The saved downfolding data use ORIGINAL impurity orbital indices.
    mapped_c_orbs must be the matching DEVICE indices in the same order.
    """
    orb = np.asarray(orb, dtype=int)
    D = sparse.csr_matrix(Delta, dtype=np.complex128)
    if D.shape != (len(orb), len(orb)):
        raise ValueError("Static Delta shape mismatch")
    if len(np.unique(orb)) != len(orb):
        raise ValueError("Duplicate static orbital indices")

    if downfold is None:
        if mapped_c_orbs is not None:
            raise ValueError("mapped_c_orbs requires downfold data")
        return {"Delta": D.copy(), "orb": orb.copy(),
                "downfold": None, "idx_H": None}

    c = np.asarray(mapped_c_orbs, dtype=int)
    if c.ndim != 1 or len(c) != len(downfold["c_orbs"]):
        raise ValueError("Hydrogen orbital mapping size mismatch")
    if len(np.unique(c)) != len(c):
        raise ValueError("Duplicate mapped hydrogen orbitals")

    orb_eff = np.union1d(orb, c)
    idx_static = np.searchsorted(orb_eff, orb)
    idx_H = np.searchsorted(orb_eff, c)

    # Sparse embedding of static Delta (do not densify the entire region).
    coo = D.tocoo()
    Delta_base = sparse.coo_matrix(
        (coo.data, (idx_static[coo.row], idx_static[coo.col])),
        shape=(len(orb_eff), len(orb_eff)),
        dtype=np.complex128,
    ).tocsr()
    return {"Delta": Delta_base, "orb": orb_eff,
            "downfold": downfold, "idx_H": idx_H}


def evaluate_exchange_vertex(z, vertex):
    """Return CSR Delta_eff(z), compatible with _selected_product."""
    D = vertex["Delta"]
    if vertex["downfold"] is None:
        return D

    data = vertex["downfold"]
    Delta_H = (hydrogen_self_energy(z, data, 0)
               - hydrogen_self_energy(z, data, 1))
    idx = vertex["idx_H"]
    m = len(idx)
    if Delta_H.shape != (m, m):
        raise ValueError("Hydrogen correction shape mismatch")

    correction = sparse.coo_matrix(
        (Delta_H.ravel(), (np.repeat(idx, m), np.tile(idx, m))),
        shape=D.shape,
    ).tocsr()
    return D + correction


def calculate_exchange(
    method,
    H_ref,
    Delta_i, orb_i,
    Delta_j, orb_j,
    energies, contour_weights,
    spin_average=False,
    orbital_resolved=False,
    orbital_kernel="full",
    rhs_batch=32,
    spin_independent_leads=True, # Share spin-0 leads for a physically nonmagnetic electrode
    verbose=False,
    nk=(1, 1, 1),
    axes=("x", "y", "z"),
    trs=False,
    HamL=None,
    semi_L=None,
    HamR=None,
    semi_R=None,
    left_orbs=None,
    right_orbs=None,
    lead_eta=1e-5,
    prefactor = -1.0 /(2*np.pi),
    self_energy_mode="compute",
    calculation_folder=None,
    write_results=False,
    results_name="Results",
    mpi=False,
    lead_mu=0.0,
    distance=None,
    downfold=False,
    hydrogen_i=None,
    hydrogen_j=None,
    mapped_c_orbs_i=None,
    mapped_c_orbs_j=None,
): 
    """Compute exchange from a selected reference using sparse LU.

    Before calling, align device and electrode potentials and ensure that
    the magnetic vertices use the same orbital ordering as H_ref.

    Algorithm for every k-point and complex contour energy:
      1. Build sparse A_sigma = z*S(k) - H_sigma(k) - Sigma_elec(z,k).
      2. Sparse-LU factorize A_sigma; never form the full inverse G.
      3. Solve A_sigma X = B for columns of each local Delta, in batches.
      4. Form local G_ij*Delta_j and G_ji*Delta_i blocks.
      5. Sum trace contributions, contour weights, and k-point weights.

    H_ref : sisl.Hamiltonian
        Pristine host for 'rkky', two-defect NM host for 'rkky_nm', or one
        spin-polarized two-defect host for 'liechtenstein'. Magnetic vertices
        must use the orbital indices of this Hamiltonian.
    energies, contour_weights : complex arrays
        E + 1j*eta and complex quadrature weights from TSCCEQ.
    HamL, HamR : sisl Hamiltonians, optional
        Left/right electrode Hamiltonians (nonmagnetic physically, but they
        may come from a spin-polarized DFT run). When both supplied,
        RecursiveSI is constructed INSIDE this function, once per calculation.
    semi_L, semi_R : str, required with HamL/HamR 
        Semi-infinite directions exactly as in the old code, e.g. '-B', '+B'.
    left_orbs, right_orbs : array-like of int, optional
        Global device orbital indices on which to place the lead self-energies.
        Omit to reproduce first/last contiguous-block assumption, but
        note this is only safe when the device interface orbital ordering
        matches the electrode's surface orbital ordering exactly.
    prefactor : float or complex
        User-defined multiplier of the contour-weighted kernel (default -1/2pi).
        Its sign and normalization depend on the Heisenberg convention,
        Delta=H_up-H_down and the contour weights; benchmark these factors.
    lead_eta : float
        RecursiveSI eta. For complex contour energies z, sisl uses Im(z).
    orbital_resolved : bool
        Integrate the contributions for each pair of local basis orbitals.
    orbital_kernel : {"full", "classes", "none"}
        What is kept per contour energy when orbital_resolved is True.
        "full": the whole orbital-pair kernel (n_i x n_j per energy; with MPI
        written as one file per energy and gathered by rank 0, which is slow
        for large vertices). "classes": the kernel summed over pairs of orbital
        classes (C-pz, H, d, other), reduced in memory, no files. "none": only
        the energy-integrated J_orb.
    nk : tuple of 3 integers, optional
        Monkhorst-Pack counts, e.g. (21, 5, 0) for x/y sampling and
        Gamma-only z. This is the only k-point input.
    axes : tuple of str
        Axis labels corresponding to nk entries (default x,y,z).
    trs : bool
        Enable sisl time-reversal reduction; disabled by default.
    spin_independent_leads : bool
        True reuses electrode spin 0 for both device spins. This is intended
        for physically nonmagnetic electrodes, even if their DFT file has
        two spin channels. False uses the respective electrode spin channels.
    self_energy_mode : str
        'compute' uses RecursiveSI, 
        'read' loads Left/Right.TBTGF,
        'compute+write' writes both spin channels then reads the files.
    calculation_folder : str, optional
        Calculation directory containing Deltas, Self_Energies and Results.
    write_results : bool
        Write results in calculation_folder
    mpi : bool
        Divide contour energies between mpi4py ranks; rank 0 returns result.
    lead_mu : float
        Chemical potential written in TBTGF header (eV).
    downfold : bool
        Supported only for 'rkky' (hydrogen correction to magnetic vertices).
        If True, requires hydrogen_i, hydrogen_j, mapped_c_orbs_i and
        mapped_c_orbs_j. Both 'rkky_nm' and 'liechtenstein' keep adatoms explicit.

    Exchange kernels (the user-selected prefactor is applied afterwards):
        rkky / rkky_nm: Tr[Delta_i G0_ij Delta_j G0_ji]
        liechtenstein:  Tr[Delta_i Gup_ij Delta_j Gdn_ji]
        The result uses 1000 * Im(sum_kE prefactor * weight * kernel) in meV.
    """    
    method = method.lower().strip()

    # Downfolding is opt-in and requires both impurities and their mappings.
    if not isinstance(downfold, (bool, np.bool_)):
        raise TypeError("downfold must be True or False")
    hydrogen_args = {
        "hydrogen_i": hydrogen_i,
        "hydrogen_j": hydrogen_j,
        "mapped_c_orbs_i": mapped_c_orbs_i,
        "mapped_c_orbs_j": mapped_c_orbs_j,
    }
    if downfold:
        missing = [name for name, value in hydrogen_args.items() if value is None]
        if missing:
            raise ValueError( "downfold=True requires hydrogen_i, hydrogen_j, mapped_c_orbs_i and mapped_c_orbs_j. "
                "Missing: " + ", ".join(missing))
        for label in ("hydrogen_i", "hydrogen_j"):
            if not isinstance(hydrogen_args[label], dict):
                raise TypeError(f"{label} must be a loaded downfolding dictionary")
    else:
        unused = [name for name, value in hydrogen_args.items() if value is not None]
        if unused:
            raise ValueError("downfold=False but hydrogen arguments were supplied: "+ ", ".join(unused))

    from exchange_mpi_io import mpi_comm
    
    comm, rank, size = mpi_comm(mpi)
    self_energy_mode = self_energy_mode.lower().strip()
    if self_energy_mode not in ("compute", "read", "compute+write"):
        raise ValueError("self_energy_mode must be compute, read or compute+write")
    if (mpi or self_energy_mode != "compute") and calculation_folder is None:
        raise ValueError("calculation_folder required for MPI or TBTGF I/O")
    # Sanity checks
    if method not in {"rkky", "rkky_nm", "liechtenstein"}:
        raise ValueError("method must be 'rkky', 'rkky_nm' or 'liechtenstein'")
    if rhs_batch < 1:
        raise ValueError("rhs_batch must be >=1")
    # NEW: NM and Liechtenstein use one common two-defect reference.
    if isinstance(H_ref, (list, tuple)):
        raise ValueError(f"{method} requires a single reference Hamiltonian")
    if method in ("rkky_nm", "liechtenstein") and downfold:
        raise ValueError(f"{method} keeps adatom orbitals explicitly; downfold=False required")
    if method == "liechtenstein" and not H_ref.spin.is_polarized:
        raise ValueError("Liechtenstein requires one spin-polarized two-defect H_ref")
    H_refs = [H_ref]
    H_base = H_refs[0]
    n = H_base.no
    if method == "rkky_nm" and H_base.spin.is_polarized:
        # NEW: spin-polarized file format is allowed for a physically NM host.
        # Do not silently use one channel of a genuinely magnetic reference.
        dH = (H_base.Hk(k=[0, 0, 0], spin=0, format="csr")
              - H_base.Hk(k=[0, 0, 0], spin=1, format="csr")).tocsr()
        if dH.nnz and np.max(np.abs(dH.data)) > 1e-3:
            raise ValueError("rkky_nm reference has >1 meV spin splitting at Gamma")
    # 1. Self Energy
    direct_leads = any(v is not None for v in (HamL, semi_L, HamR, semi_R))
    self_energy = None
    if direct_leads and any(v is None for v in (HamL, semi_L, HamR, semi_R)):
        raise ValueError("Supply all of HamL, semi_L, HamR, semi_R together")
    if not direct_leads and (left_orbs is not None or right_orbs is not None):
        raise ValueError("left_orbs/right_orbs only apply with HamL/HamR")
    # A semi-infinite direction MUST NOT ALSO BE INTEGRATED with Bloch k.
    # In your original -B/+B GNR setup, k_y is not a periodic direction.
    if direct_leads:
        if semi_L not in ("+A", "-A", "+B", "-B", "+C", "-C") or \
           semi_R not in ("+A", "-A", "+B", "-B", "+C", "-C"):
            raise ValueError("Electrode directions must be +/-A, +/-B or +/-C")
        # Identify transport direction: A=0, B=1, C=2
        open_axis = "ABC".index(semi_L[1])
        print("Transport axis:",open_axis)
        counts = {axis.lower(): max(1, int(count))     for axis, count in zip(axes, nk) }
        # Do not sample k along the transport direction
        if counts["xyz"[open_axis]] > 1:
            raise ValueError(    f"Cannot sample k along open electrode direction {semi_L[1]}" )
        for side, electrode in (("left", HamL), ("right", HamR)):
            if electrode.nsc[open_axis] != 3:
                raise ValueError(f"{side} electrode requires nsc[{open_axis}]=3 along the transport direction")
        # An electrode must contain +/- nearest-cell couplings on
        # every Bloch-sampled axis (nsc=3 on that axis).
        for axis, axis_name in enumerate("xyz"):
            if counts[axis_name] > 1:
                for side, electrode in (("left", HamL), ("right", HamR)):
                    if electrode.nsc[axis] != 3:
                        raise ValueError(f"{side} electrode requires nsc[{axis_name}]=3 for k sampling, got {electrode.nsc[axis]}")
        # Remove periodic couplings of the device along transport direction
        for ih, h in enumerate(H_refs):
            if h.nsc[open_axis] != 1:
                h = h.copy()
                h.set_nsc([None if i != open_axis else 1 for i in range(3)])
                H_refs[ih] = h
        H_base = H_refs[0]
        # Electrode provider is selected after the k-point grid is known.
    # 1. Reading Perturbations
    print("Reading Perturbations")
    Di, oi = _validate_vertex(Delta_i, orb_i, n, "i")
    Dj, oj = _validate_vertex(Delta_j, orb_j, n, "j")

    # NEW: local spin rotations must have disjoint orbital regions.
    if method == "liechtenstein" and np.intersect1d(oi, oj).size:
        raise ValueError("Liechtenstein magnetic regions overlap; use disjoint radius/masks")
    if direct_leads and method in ("rkky_nm", "liechtenstein"):
        li, ri = check_electrode_interfaces(H_base, HamL, HamR,
                                             left_orbs=left_orbs, right_orbs=right_orbs)
        interfaces = np.union1d(li, ri)
        if np.intersect1d(np.union1d(oi, oj), interfaces).size:
            raise ValueError("Magnetic region overlaps a device/electrode interface")

    # vertex with downfoalding
    vertex_i = prepare_exchange_vertex(
        Di, oi,
        downfold=hydrogen_i if downfold else None,
        mapped_c_orbs=mapped_c_orbs_i if downfold else None,
    )
    vertex_j = prepare_exchange_vertex(
        Dj, oj,
        downfold=hydrogen_j if downfold else None,
        mapped_c_orbs=mapped_c_orbs_j if downfold else None,
    )
    orb_i_eff = vertex_i["orb"]
    orb_j_eff = vertex_j["orb"]
    # Check mapped indices against device size before the expensive LU loop.
    _validate_vertex(vertex_i["Delta"], orb_i_eff, n, "i effective")
    _validate_vertex(vertex_j["Delta"], orb_j_eff, n, "j effective")
    
    # 2. Complex Contour    
    print("Reading CC")
    z_array = np.asarray(energies, dtype=np.complex128)
    w_array = np.asarray(contour_weights, dtype=np.complex128)

    # 3. K-points 
    print("Reading kpts")
    kvals, kw = monkhorst_pack_grid(H_base, nk=nk, axes=axes, trs=trs)

    # Save input perturbations and configuration once, on rank 0.
    config_error = None
    if calculation_folder is not None:
        from exchange_mpi_io import prepare_calculation_folder
        try:
            prepare_calculation_folder(
                calculation_folder, H_refs, method, Di, oi, Dj, oj, z_array, w_array,
                nk, axes, trs, spin_independent_leads, orbital_resolved, prefactor,
                left_orbs, right_orbs, lead_eta, semi_L, semi_R,
                rank=rank, results_name=results_name,
                downfold=downfold,
                hydrogen_i=hydrogen_i if downfold else None,
                hydrogen_j=hydrogen_j if downfold else None,
                mapped_c_orbs_i=mapped_c_orbs_i if downfold else None,
                mapped_c_orbs_j=mapped_c_orbs_j if downfold else None,
            )
        except Exception as exc:
            config_error = f'{type(exc).__name__}: {exc}'
    if comm is not None:
        config_error = comm.bcast(config_error if rank == 0 else None, root=0)
    if config_error:
        raise RuntimeError('Calculation configuration failed: ' + config_error)

    if direct_leads:
        if self_energy_mode in ("read", "compute+write"):
            from exchange_mpi_io import TBTGFProvider, write_electrode_tbtgf
            print("Processing self energy:")
            se_folder = os.path.join(calculation_folder, "Self_Energies")
            if self_energy_mode == "compute+write":
                print("writting self energy")
                write_electrode_tbtgf(
                    se_folder, H_base, HamL, semi_L, HamR, semi_R,
                    z_array, nk=nk, axes=axes, trs=trs,
                    left_orbs=left_orbs, right_orbs=right_orbs,
                    lead_eta=lead_eta, mu=lead_mu, mpi=mpi)
            print("Reading Self energy")
            self_energy = TBTGFProvider(
                se_folder, H_base, HamL, HamR, semi_L, semi_R,
                z_array, kvals, kw, left_orbs=left_orbs,
                right_orbs=right_orbs, lead_eta=lead_eta, mu=lead_mu)
        else:
            print("computing self energies")
            self_energy = recursive_leads_from_hamiltonians(
                H_device=H_base,
                HamL=HamL, semi_L=semi_L,
                HamR=HamR, semi_R=semi_R,
                left_orbs=left_orbs, right_orbs=right_orbs,
                eta=lead_eta,
                polarized_leads=(HamL.spin.is_polarized or HamR.spin.is_polarized))
    elif self_energy_mode != "compute":
        raise ValueError("TBTGF modes require electrodes")

    # 4. Kernels
    print("Generating kernel")
    kernels = np.zeros((len(kvals), len(z_array)), dtype=np.complex128)  # CHANGE: nk is a 3-tuple
    orb_integrated = np.zeros((len(orb_i_eff), len(orb_j_eff)), dtype=complex) if orbital_resolved else None
    if orbital_kernel not in ("full", "classes", "none"):
        raise ValueError("orbital_kernel must be 'full', 'classes' or 'none'")
    keep_full = orbital_resolved and orbital_kernel == "full"
    class_kernels = None
    if orbital_resolved and orbital_kernel == "classes":
        lab_i, names_i = orbital_class_labels(H_base.geometry, orb_i_eff)
        lab_j, names_j = orbital_class_labels(H_base.geometry, orb_j_eff)
        class_kernels = np.zeros((len(kvals), len(z_array), len(names_i), len(names_j)), dtype=complex)

    # orbital contribution for each k-point and contour energy
    if keep_full and size == 1:
        if calculation_folder is None:
            orb_kernels = np.zeros((len(kvals), len(z_array), len(orb_i_eff), len(orb_j_eff)), dtype=complex)
        else:
            orb_kernels = np.lib.format.open_memmap(
                os.path.join(calculation_folder, results_name, "weighted_orbital_kernel.npy"),
                mode="w+", dtype=np.complex128,
                shape=(len(kvals), len(z_array), len(orb_i_eff), len(orb_j_eff)))
    else:
        orb_kernels = None
    p = complex(prefactor)

    #5. Loop over kpts
    print("Looping over kpts")
    print("Method:" ,method)
    for ik, kvec in enumerate(kvals):
        # 6. Get Hk,Sk
        Sk = sparse.csc_matrix(H_base.Sk(k=kvec, format="csr"), dtype=complex)
        if method in ("rkky", "rkky_nm"):
            # NEW: rkky_nm uses the two-defect NM propagator; same kernel as rkky.
            Hk0 = H_base.Hk(k=kvec, spin=0, format="csr") if H_base.spin.is_polarized \
                else H_base.Hk(k=kvec, format="csr")
            Hk0 = sparse.csc_matrix(Hk0, dtype=complex)
        else:  # liechtenstein: both spins of the SAME two-defect Hamiltonian
            Hkup = sparse.csc_matrix(H_base.Hk(k=kvec, spin=0, format="csr"), dtype=complex)
            Hkdn = sparse.csc_matrix(H_base.Hk(k=kvec, spin=1, format="csr"), dtype=complex)
        
        for ie in range(rank, len(z_array), size):
            z = z_array[ie]
            # Delta + Delta(z)
            Delta_i_z = evaluate_exchange_vertex(z, vertex_i)
            Delta_j_z = evaluate_exchange_vertex(z, vertex_j)
            
            # At this (k,E), cache embedding by spin to avoid repeated TBTGF reads.
            sigma_by_spin = {}
            def embedded_sigma(spin):
                key = 0 if spin_independent_leads else spin
                if key not in sigma_by_spin:
                    if self_energy is None:
                        terms = None
                    elif self_energy_mode == "compute":
                        terms = self_energy(z, kvec, key)
                    else:
                        terms = self_energy(z, kvec, key, ie=ie, ik=ik)
                    sigma_by_spin[key] = _lead_embedding(n, terms)
                return sigma_by_spin[key]
            shared_sigma = None
            if method == "liechtenstein" and spin_independent_leads:
                # NEW: physically NM leads use the same self-energy for up/down.
                shared_sigma = embedded_sigma(0)
            
            def factor(hmat, spin, overlap=None):
                """
                Factorize the inverse retarded Green's function:   
                G(z,k) = [z S(k) - H(k) - Sigma(z,k)]^(-1)
                    
                Parameters
                ----------
                hmat : sparse matrix
                    Spin-dependent or nonmagnetic Hamiltonian H(k).
                spin : int
                    spin index 
                Returns
                -------
                scipy.sparse.linalg.SuperLU
                LU factorization of the inverse Green's function.
                """
                if shared_sigma is not None:
                    sigma = shared_sigma
                else:
                    sigma = embedded_sigma(spin)
                A = (z * (Sk if overlap is None else overlap) - hmat - sigma).tocsc()
                return splu(A)
            if method in ("rkky", "rkky_nm"):
                # x=(G0ijDelta_j) y=(G0_jiDelta_ij)
                lu = factor(Hk0, None) 
                x = _selected_product(lu, orb_i_eff, orb_j_eff, Delta_j_z, rhs_batch)
                y = _selected_product(lu, orb_j_eff, orb_i_eff, Delta_i_z, rhs_batch)
                del lu
                contribution = x * y.T
            if method == "liechtenstein":
                # NEW: standard common-reference kernel Gup_ij Dj Gdn_ji Di.
                # Factorize one spin at a time to limit LU memory usage.
                lu_up = factor(Hkup, 0)
                x_up = _selected_product(lu_up, orb_i_eff, orb_j_eff, Delta_j_z, rhs_batch)
                del lu_up
                lu_dn = factor(Hkdn, 1)
                y_dn = _selected_product(lu_dn, orb_j_eff, orb_i_eff, Delta_i_z, rhs_batch)
                del lu_dn
                contribution = x_up * y_dn.T
            weighted = p * w_array[ie]
            # integrate kernenls
            kernels[ik, ie] = weighted * contribution.sum()
            if orbital_resolved:
                orb_integrated += kw[ik] * weighted * contribution
                if class_kernels is not None:
                    class_kernels[ik, ie] = weighted * class_sums(contribution, lab_i, lab_j,
                                                                  len(names_i), len(names_j))
                if not keep_full:
                    pass
                elif size == 1:
                    orb_kernels[ik, ie] = weighted * contribution
                else:
                    from exchange_mpi_io import _atomic_npy
                    temp_file = os.path.join(
                        calculation_folder, results_name, "tmp", f"Orb_E{ie:08d}_K{ik:04d}.npy")
                    _atomic_npy(temp_file, weighted * contribution)
            if verbose:
                print(f"k={ik+1}/{len(kvals)} E={ie+1}/{len(z_array)} K={kernels[ik,ie]:.6g}")
    if comm is not None:
        from mpi4py import MPI
        total_kernels = np.zeros_like(kernels) if rank == 0 else None
        comm.Reduce(kernels, total_kernels, op=MPI.SUM, root=0)
        kernels = total_kernels
        if orbital_resolved:
            total_orb = np.zeros_like(orb_integrated) if rank == 0 else None
            comm.Reduce(orb_integrated, total_orb, op=MPI.SUM, root=0)
            orb_integrated = total_orb
        if class_kernels is not None:
            total_cls = np.zeros_like(class_kernels) if rank == 0 else None
            comm.Reduce(class_kernels, total_cls, op=MPI.SUM, root=0)
            class_kernels = total_cls
        comm.Barrier()
        if rank != 0:
            if self_energy_mode in ("read", "compute+write") and self_energy is not None:
                self_energy.close()
            return None
        if keep_full and size > 1:
            from pathlib import Path
            orb_kernels = np.lib.format.open_memmap(
                os.path.join(calculation_folder, results_name, "weighted_orbital_kernel.npy"),
                mode="w+", dtype=np.complex128,
                shape=(len(kvals), len(z_array), len(orb_i_eff), len(orb_j_eff)))
            for ie in range(len(z_array)):
                for ik in range(len(kvals)):
                    f = Path(calculation_folder) / results_name / "tmp" / f"Orb_E{ie:08d}_K{ik:04d}.npy"
                    block = np.load(f, allow_pickle=False, mmap_mode="r")
                    orb_kernels[ik, ie] = block
                    del block
                    f.unlink()
            orb_kernels.flush()
    if self_energy_mode in ("read", "compute+write") and self_energy is not None:
        self_energy.close()
    J_k = 1000.0 * np.imag(kernels.sum(axis=1)) # in
    J = float(np.dot(kw, J_k))
    J_orb = 1000.0 * orb_integrated.imag if orbital_resolved else None
    result = ExchangeResult(
        method=method, J_meV=J, J_k_meV=J_k, kpoints=kvals,
        k_weights=kw, energies=z_array, contour_weights=w_array,
        weighted_kernel_kE=kernels, J_orb_meV=J_orb,
        prefactor=p, spin_average=False, orb_i=orb_i_eff.copy(), orb_j=orb_j_eff.copy(),weighted_orbital_kernel_kE=orb_kernels,distance=distance,
        weighted_class_kernel_kE=class_kernels,
        class_names_i=names_i if class_kernels is not None else None,
        class_names_j=names_j if class_kernels is not None else None)
    if (calculation_folder is not None) and (write_results ==True) :
        result.save(os.path.join(calculation_folder, results_name, "ExchangeResult.npz"))
    return result

def get_R0_block(H, dim):
    """Return R=(0,0,0) block of a sisl Hamiltonian component."""
    no = H.no
    isc0 = H.geometry.sc_index([0, 0, 0])
    M = H.tocsr(dim=dim)
    return M[:, isc0 * no:(isc0 + 1) * no].tocsr()

def select_retained_orbitals(H, H_idx, basis="pz"):
    """
    Select retained orbitals in the ORIGINAL full DFT indexing.
    pz1  : first-zeta pz
    pz   : all pz
    pz+dz: all pz + d orbitals with m=-1,0,1
    all  : all orbitals except those of H_idx
    """
    basis = basis.lower()
    if basis == "full":
        basis = "all"
    if basis not in ("pz1", "pz", "pz+dz", "all"):
        raise ValueError(f"Unknown basis: {basis}")
    g = H.geometry
    retained_orbs = []

    for ia in range(g.na):
        if ia == H_idx:
            continue

        atom = g.atoms[ia]
        atom_orbs = g.a2o(ia, all=True)

        for io, orb in zip(atom_orbs, atom.orbitals):

            pz = (orb.l == 1 and orb.m == 0)
            dz = (orb.l == 2 and orb.m in (-1,0, 1)) # dz2 important for adatoms

            if basis == "all":
                keep = True
            elif basis == "pz1":
                keep = pz and orb.zeta == 1
            elif basis == "pz":
                keep = pz
            else:
                keep = pz or dz

            if keep:
                retained_orbs.append(int(io))

    return np.asarray(retained_orbs, dtype=int)

def prepare_hydrogen_downfold(H, H_idx, Z=1, basis="pz"):
    """
    Prepare local H-downfolding matrices.
    
    H     : full spin-polarized sisl Hamiltonian
    H_idx : index of hydrogen atom to eliminate
    basis : pz1, pz, pz+dz, all
    
    Returns local blocks and original orbital indices.
    """
    print(f"Preparing Adadtom Downfolding",flush=True)

    if not H.spin.is_polarized:
        raise ValueError("A collinear spin-polarized H is required.")

    if H.geometry.atoms[H_idx].Z != Z:
        raise ValueError(F"idx must identify Z={Z}.")
    h = np.asarray(H.geometry.a2o(H_idx, all=True), dtype=int)
    orb_all = select_retained_orbitals(H, H_idx, basis)

    # R=0 Hamiltonian and overlap
    Hu = get_R0_block(H, 0) #up
    Hd = get_R0_block(H, 1) #dn

    if H.non_orthogonal:
        print("0verlap matrix was found")
        S = get_R0_block(H, H.dim - 1)
    else:
        print("No overlap matrix was found")
        S = eye(H.no, format="csr")

    def block(M, rows, cols):
        return M[rows, :][:, cols].toarray()

    # Keep only orbitals directly coupled to this hydrogen.
    # This makes Sigma_H small, even for a large ribbon.
    active = np.zeros(len(orb_all), dtype=bool)

    for M in (Hu, Hd, S):
        active |= np.any(block(M, orb_all, h) != 0, axis=1)
        active |= np.any(block(M, h, orb_all) != 0, axis=0)

    orb = orb_all[active]

    if len(orb) == 0:
        raise ValueError("No retained orbitals couple to hydrogen.")
    
    
    print(f"Total retained orbitals: {len(orb_all)}")
    print(f"Orbitals coupled to adatom: {len(orb)}")
    print(f"Hydrogen orbitals: {len(h)}",flush=True)
    
    data = {
        "basis": basis,
        "c_orbs": orb,
        "h_orbs": h,
        "S_CC": block(S, orb, orb),
        "S_CH": block(S, orb, h),
        "S_HC": block(S, h, orb),
        "S_HH": block(S, h, h),
    }

    for spin, Hs in enumerate((Hu, Hd)):
        data[spin] = {
            "H_CC": block(Hs, orb, orb),
            "H_CH": block(Hs, orb, h),
            "H_HC": block(Hs, h, orb),
            "H_HH": block(Hs, h, h),
        }
    return data

def hydrogen_self_energy(z, data, spin):
    """Hydrogen self-energy on retained orbitals."""
    d = data[spin]
    A = d["H_CH"] - z * data["S_CH"]
    B = z * data["S_HH"] - d["H_HH"]
    C = d["H_HC"] - z * data["S_HC"]
    # Gcc=(zScc - Hcc -Simga_h)
    #sigma_h=(Hch-zSch)(zS_hh-H_hh)-1(Hhc-zHhc)
    # A (B-1 C)
    return A @ np.linalg.solve(B, C)

def effective_delta(z, Delta_CC, data):
    """Spin splitting including hydrogen downfolding."""
    Sigma_up = hydrogen_self_energy(z, data, 0)
    Sigma_dn = hydrogen_self_energy(z, data, 1)
    return Delta_CC + (Sigma_up - Sigma_dn), (Sigma_up - Sigma_dn)

def test_downfold(data, z=0.2 + 0.05j):
    nc = len(data["c_orbs"])
    for spin in (0, 1):
        d = data[spin]
        A = z * data["S_CC"] - d["H_CC"]
        B = z * data["S_CH"] - d["H_CH"]
        C = z * data["S_HC"] - d["H_HC"]
        D = z * data["S_HH"] - d["H_HH"]
        # Green function before eliminating H
        M = np.block([[A, B], [C, D]])
        # Gcc Gch    
        # Ghc Ghc
        G_full = np.linalg.inv(M)[:nc, :nc]
        # Green function after eliminating H
        Sigma = hydrogen_self_energy(z, data, spin)
        G_eff = np.linalg.inv(A - Sigma)
        error = np.linalg.norm(G_full - G_eff) / np.linalg.norm(G_full)
        print(f"Spin {spin}: relative error = {error:.3e}")
        assert error < 1e-10
    print("Downfolding test passed")

def check_hydrogen_supercell_blocks(H, H_idx, basis="pz"):
    g = H.geometry
    no = H.no

    h = np.asarray(g.a2o(H_idx, all=True), dtype=int)
    c = select_retained_orbitals(H, H_idx, basis)

    Hu = H.tocsr(dim=0)
    Hd = H.tocsr(dim=1)
    S = H.tocsr(dim=H.dim - 1) if H.non_orthogonal else None

    for R in g.sc_off:
        isc = int(g.sc_index(R))
        hc = isc * no + c
        hh = isc * no + h
        print(f"\nR = {tuple(R)}")
        for name, M in (("Hu", Hu), ("Hd", Hd), ("S", S)):
            if M is None:
                continue
            H_HC = M[h, :][:, hc].toarray()
            H_HH = M[h, :][:, hh].toarray()
            print(f"{name}: "f"||HC|| = {np.linalg.norm(H_HC):.6f}, "  f"||HH|| = {np.linalg.norm(H_HH):.6f}" )


import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def align_defect_to_electrodes(
    H_def,
    H_0,
    axis=0,
    tol=0.2,
    max_offset_difference=0.01,
    verbose=True,
):
    """
    Align a pruned defect Hamiltonian to a pristine electrode.

    Works with nonmagnetic or collinear spin-polarized Hamiltonians.
    Extra adatom orbitals may be retained in H_def.

    Parameters
    ----------
    H_def : sisl.Hamiltonian
        Pruned device Hamiltonian, optionally containing adatoms.
        May be nonmagnetic or spin-polarized.
    H_0 : sisl.Hamiltonian
        Pruned, already aligned electrode Hamiltonian.
        May be nonmagnetic or spin-polarized.
        Assumes one electrode unit cell.
    axis : int
        Transport lattice-vector index.
    tol : float
        Maximum geometrical mismatch in Angstrom.
    max_offset_difference : float
        Maximum allowed difference between left and right
        fitted energy offsets, in eV.
    verbose : bool
        Print alignment diagnostics.

    Returns
    -------
    H_aligned : sisl.Hamiltonian
        Aligned copy of H_def.
    dev_eshift : float
        Common energy shift applied, in eV.
    info : dict
        Alignment diagnostics and electrode-interface indices.
    """

    # 1. Identify atoms compatible with the pristine electrode.
    # Do not assume that adatoms are absent or stored
    # in any particular position in the atomic ordering.
    def orbital_signature(atom):
        return (
            int(atom.Z),
            tuple((orb.l, orb.m) for orb in atom.orbitals)
        )

    ref_keys = [
        orbital_signature(H_0.atoms[ia])
        for ia in range(H_0.na)
    ]

    ref_key_set = set(ref_keys)

    candidate_atoms = np.array([
        ia for ia in range(H_def.na)
        if orbital_signature(H_def.atoms[ia]) in ref_key_set
    ], dtype=int)

    if len(candidate_atoms) < 2 * H_0.na:
        raise ValueError(
            "Not enough electrode-compatible atoms to "
            "identify both terminal regions."
        )

    # 2. Find the leftmost and rightmost electrode-sized regions.
    # Select spatially, not by the original atom indices.
    direction = np.asarray(H_0.cell[axis], dtype=float)
    direction /= np.linalg.norm(direction)

    xyz_def = np.asarray(H_def.geometry.xyz)
    xyz_ref = np.asarray(H_0.geometry.xyz)

    projection = xyz_def @ direction
    ordered = candidate_atoms[
        np.argsort(projection[candidate_atoms], kind="stable")
    ]

    left_candidates = ordered[:H_0.na]
    right_candidates = ordered[-H_0.na:]

    # 3. Match each terminal atom with its corresponding
    # electrode atom, preserving the electrode orbital ordering.
    def match_region(candidates, side):
        xyz = xyz_def[candidates]

        # A rigid translation does not affect matching.
        ref_centered = xyz_ref - xyz_ref.mean(axis=0)
        def_centered = xyz - xyz.mean(axis=0)

        cost = cdist(ref_centered, def_centered)

        for i, key in enumerate(ref_keys):
            for j, ia in enumerate(candidates):
                if orbital_signature(H_def.atoms[int(ia)]) != key:
                    cost[i, j] = 1e6

        rows, cols = linear_sum_assignment(cost)

        mismatch = cost[rows, cols].max()

        if mismatch > tol:
            raise ValueError(
                f"{side} electrode geometry mismatch: "
                f"{mismatch:.4f} Ang > tolerance {tol:.4f} Ang. "
                "Check terminal cell selection or relaxation."
            )

        # Atoms are now ordered like the pristine electrode.
        atoms = candidates[cols]

        ref_orbs = np.concatenate([
            np.asarray(H_0.a2o(int(ia), all=True), dtype=int)
            for ia in rows
        ])

        def_orbs = np.concatenate([
            np.asarray(H_def.a2o(int(ia), all=True), dtype=int)
            for ia in atoms
        ])

        if len(ref_orbs) != len(def_orbs):
            raise ValueError(f"{side} electrode orbital mismatch.")

        return atoms, ref_orbs, def_orbs, mismatch

    left_atoms, ref_L, orb_L, geom_L = match_region(
        left_candidates, "Left"
    )

    right_atoms, ref_R, orb_R, geom_R = match_region(
        right_candidates, "Right"
    )

    # 4. Get central-cell Hamiltonian matrices.
    # Keep only R=0 couplings for the alignment comparison.
    H_ref_local = H_0.copy()
    H_ref_local.set_nsc([1, 1, 1])

    H_def_local = H_def.copy()
    H_def_local.set_nsc([1, 1, 1])

    # NEW: allow both reference and device to be spin-polarized.
    polarized_ref = H_0.spin.is_polarized
    polarized = H_def.spin.is_polarized

    # unpolarized matrices take no spin argument in recent sisl
    H_ref_up = (H_ref_local.Hk(spin=0, format="csr") if polarized_ref
                else H_ref_local.Hk(format="csr"))
    H_up = H_def_local.Hk(spin=0, format="csr") if polarized else H_def_local.Hk(format="csr")

    if polarized_ref:
        H_ref_dn = H_ref_local.Hk(spin=1, format="csr")
    else:
        H_ref_dn = H_ref_up

    if polarized:
        H_dn = H_def_local.Hk(spin=1, format="csr")
    else:
        H_dn = H_up

    # NEW: compare the spin averages for energy alignment.
    H_ref_mat = 0.5 * (H_ref_up + H_ref_dn)
    H_avg = 0.5 * (H_up + H_dn)

    S_ref = H_ref_local.Sk(format="csr")
    S_def = H_def_local.Sk(format="csr")

    # 5. Fit the energy shift independently on both sides.
    # Use spin-averaged onsite energies.
    def fit_offset(ref_orbs, def_orbs):
        h_ref = H_ref_mat.diagonal()[ref_orbs]
        h_def = H_avg.diagonal()[def_orbs]
        s_def = S_def.diagonal()[def_orbs]

        delta = h_ref - h_def

        dev_eshift = (
            np.vdot(s_def, delta).real
            / np.vdot(s_def, s_def).real
        )

        residual = delta - dev_eshift * s_def
        rms = np.sqrt(np.mean(np.abs(residual)**2))

        spin_split = (
            H_up.diagonal()[def_orbs]
            - H_dn.diagonal()[def_orbs]
        )

        # NEW: compare device and electrode spin splitting.
        ref_spin_split = (
            H_ref_up.diagonal()[ref_orbs]
            - H_ref_dn.diagonal()[ref_orbs]
        )

        spin_mismatch = np.max(
            np.abs(spin_split - ref_spin_split)
        )

        return (
            float(dev_eshift),
            float(rms),
            float(np.max(np.abs(spin_split))),
            float(spin_mismatch),  # NEW
        )

    # NEW: also retrieve the spin mismatch.
    shift_L, rms_L, spin_L, spin_mismatch_L = fit_offset(
        ref_L, orb_L
    )
    shift_R, rms_R, spin_R, spin_mismatch_R = fit_offset(
        ref_R, orb_R
    )

    # 6. Check whether one common shift is justified.
    if abs(shift_L - shift_R) > max_offset_difference:
        raise ValueError(
            "Left and right electrode offsets disagree: "
            f"{shift_L:.6f} vs {shift_R:.6f} eV. "
            "The terminal regions may not be pristine-like."
        )

    dev_eshift = 0.5 * (shift_L + shift_R)

    # 7. Check complete intra-cell Hamiltonian and overlap blocks.
    # Diagnostics beyond the diagonal onsite energies.
    def block_residual(ref_orbs, def_orbs):
        hr = H_ref_mat[ref_orbs, :][:, ref_orbs].toarray()
        hd = H_avg[def_orbs, :][:, def_orbs].toarray()

        sr = S_ref[ref_orbs, :][:, ref_orbs].toarray()
        sd = S_def[def_orbs, :][:, def_orbs].toarray()

        dh = hr - (hd + dev_eshift * sd)

        return (
            float(np.max(np.abs(dh))),
            float(np.max(np.abs(sr - sd))),
        )

    dh_L, ds_L = block_residual(ref_L, orb_L)
    dh_R, ds_R = block_residual(ref_R, orb_R)

    # 8. Apply one common shift to the entire device.
    # This preserves H_up - H_down for spin-independent overlap.
    H_aligned = H_def.copy()
    H_aligned.shift(dev_eshift)

    H0_up = (H_0.Hk(k=[0, 0, 0], spin=0, format="csr") if polarized_ref
             else H_0.Hk(k=[0, 0, 0], format="csr"))

    if polarized_ref:
        H0_dn = H_0.Hk(k=[0, 0, 0], spin=1, format="csr")
        dH0 = H0_up - H0_dn
        max_spin_elec = np.max(np.abs(dH0.data)) if dH0.nnz else 0.0
    else:
        max_spin_elec = 0.0
    # NEW: maximum spin splitting within each terminal block
    def max_spin_block(orbitals):
        dH = (H_up - H_dn)[orbitals, :][:, orbitals].tocsr()
        return float(np.max(np.abs(dH.data))) if dH.nnz else 0.0
    
    max_spin_L = max_spin_block(orb_L)
    max_spin_R = max_spin_block(orb_R)

    # 9. Return diagnostics and explicit interface indices.
    info = {
        "shift_left": shift_L,
        "shift_right": shift_R,
        "dev_eshift": dev_eshift,
        "left_atoms": left_atoms,
        "right_atoms": right_atoms,
        "left_orbs": orb_L,
        "right_orbs": orb_R,
        "rms_onsite_left": rms_L,
        "rms_onsite_right": rms_R,
        "spin_splitting_left": spin_L,
        "spin_splitting_right": spin_R,
        "spin_mismatch_left": spin_mismatch_L,    # NEW
        "spin_mismatch_right": spin_mismatch_R,   # NEW
        "max_dH_left": dh_L,
        "max_dH_right": dh_R,
        "max_dS_left": ds_L,
        "max_dS_right": ds_R,
        "geometry_left": geom_L,
        "geometry_right": geom_R,
    }

    if verbose:
        print("\n========== Electrode alignment ==========")
        print(f"Left offset:       {shift_L:+.6f} eV")
        print(f"Right offset:      {shift_R:+.6f} eV")
        print(f"Common shift:      {dev_eshift:+.6f} eV")
        print(f"Left-right diff:   {abs(shift_L-shift_R)*1000:.3f} meV")

        print("\n========== Spin polarization ==========")
        print(f"Electrode max |Hup-Hdn|:  {max_spin_elec*1000:.3f} meV")
        print(f"Device left max splitting:  {max_spin_L*1000:.3f} meV")
        print(f"Device right max splitting: {max_spin_R*1000:.3f} meV")

        print("\n========== Terminal onsite diagnostics ==========")
        print(f"Left RMS residual:  {rms_L*1000:.3f} meV")
        print(f"Right RMS residual: {rms_R*1000:.3f} meV")
        print(f"Left max splitting: {spin_L*1000:.3f} meV")
        print(f"Right max splitting:{spin_R*1000:.3f} meV")

        # NEW: spin polarization matching diagnostics
        print("\n========== Spin polarization matching ==========")
        print(f"Electrode spin polarized: {polarized_ref}")
        print(f"Device spin polarized:    {polarized}")
        print(f"Left spin mismatch:  {spin_mismatch_L*1000:.3f} meV")
        print(f"Right spin mismatch: {spin_mismatch_R*1000:.3f} meV")

        print("\n========== Intra-cell matrix comparison after alignment==========")
        print(f"Left max |dH|:      {dh_L*1000:.3f} meV")
        print(f"Right max |dH|:     {dh_R*1000:.3f} meV")
        print(f"Left max |dS|:      {ds_L:.3e}")
        print(f"Right max |dS|:     {ds_R:.3e}")

        print("\n========== Geometry mismatch ==========")
        print(f"Left:  {geom_L:.4f} Ang")
        print(f"Right: {geom_R:.4f} Ang")
        print(f"Interface orbitals: L={len(orb_L)}, R={len(orb_R)}")
        print("========================================\n")

    return H_aligned, dev_eshift, info


def calculate_dos_embedding(
    method,
    H_ref,
    HamL,
    semi_L,
    HamR,
    semi_R,
    energies,
    nk=(1, 0, 0),
    axes=("x", "y", "z"),
    trs=False,
    left_orbs=None,
    right_orbs=None,
    spin_independent_leads=False,
    eta=0.01,
    lead_eta=1e-5,
    orbitals=None,
    rhs_batch=128,
    return_orbital_dos=False,
    include_leads=True,
    verbose=False,
    comm=None,
):
    """Calculate the device DOS using the SAME A=zS-H-Sigma as calculate_exchange.

    method='rkky' or 'rkky_nm': one nonmagnetic DOS channel.
    method='liechtenstein': H_ref up/down; physically nonmagnetic leads allowed.
       spin_independent_leads=True: reuse lead spin 0 for both channels.
       spin_independent_leads=False: use lead spins 0 and 1 separately.

    energies is a 1D REAL energy grid in eV
    eta > 0: retarded broadening in z=E+1j*eta.

    DOS for selected orbitals is -Im diag(G S)/pi, integrated over k.
    With orbitals=None ALL device orbitals are included (total device DOS).
    For large devices select a few orbitals for a faster local DOS test.
    Set include_leads=False to obtain the isolated (unembedded) device DOS
    for comparison, using exactly the same device Hamiltonian and k-grid.
    No dense device Green's-function inverse is constructed; LU is solved
    in column batches instead (rhs_batch columns per solve).
    comm: optional MPI communicator; the energies are distributed over the
    ranks and the DOS is reduced to rank 0, which returns the dict (the other
    ranks return None).

    Output dict:
      'energies'       (nE,)
      'spins'          ('0',) for rkky or ('up','down') for liechtenstein
      'dos'            (nspin,nE), sum over selected device orbitals [states/eV]
      'orbital_dos'    (nspin,nE,nselected) if requested, else None
      'orbitals'       indices of selected device orbitals
      'kpoints','k_weights'  Monkhorst-Pack grid and weights
      'is_total'       True only if all device orbitals were selected
    """
    method = method.lower().strip()
    if method not in ("rkky", "rkky_nm", "liechtenstein"):
        raise ValueError("method must be 'rkky', 'rkky_nm' or 'liechtenstein'")
    if method == "liechtenstein" and not H_ref.spin.is_polarized:
        raise ValueError("Liechtenstein requires a spin-polarized device")
    if not np.isfinite(eta) or eta <= 0:
        raise ValueError("eta must be positive")
    if rhs_batch < 1:
        raise ValueError("rhs_batch must be >= 1")
    E = np.asarray(energies, dtype=float)
    if E.ndim != 1 or E.size == 0:
        raise ValueError("energies must be a nonempty real 1D array")

    allowed = ("+A", "-A", "+B", "-B", "+C", "-C")
    if semi_L not in allowed or semi_R not in allowed or semi_L[1] != semi_R[1]:
        raise ValueError("Electrodes must use valid directions on the same axis")
    open_axis = "ABC".index(semi_L[1])
    if len(nk) != 3 or len(axes) != 3 or set(a.lower() for a in axes) != set("xyz"):
        raise ValueError("nk and axes must have three entries, and axes must be x/y/z")
    
    if np.any(np.asarray(nk, dtype=int) < 0):
        raise ValueError("nk entries must be nonnegative")
    
    counts = {a.lower(): max(1, int(v)) for a, v in zip(axes, nk)}
    if counts["xyz"[open_axis]] > 1:
        raise ValueError("Cannot sample k along the open transport direction")
    for side, elec in (("left", HamL), ("right", HamR)):
        if elec.nsc[open_axis] != 3:
            raise ValueError(f"{side} electrode requires nsc=3 along transport")
        for axis, axis_name in enumerate("xyz"):
            if counts[axis_name] > 1 and elec.nsc[axis] != 3:
                raise ValueError(f"{side} electrode requires nsc=3 along sampled {axis_name}")

    # Like calculate_exchange: open the device only along transport.
    if H_ref.nsc[open_axis] != 1:
        H_ref = H_ref.copy()
        H_ref.set_nsc([1 if a == open_axis else None for a in range(3)])

    n = H_ref.no
    if orbitals is None:
        selected = np.arange(n, dtype=int)
    else:
        selected = np.asarray(orbitals, dtype=int)
        if (selected.ndim != 1 or not selected.size
                or np.any(selected < 0) or np.any(selected >= n)
                or len(np.unique(selected)) != len(selected)):
            raise ValueError("orbitals must be a nonempty array of unique device indices")

    # Exactly the same electrode self-energy provider and block embedding.
    self_energy = None
    if include_leads:
        self_energy = recursive_leads_from_hamiltonians(
            H_device=H_ref,
            HamL=HamL, semi_L=semi_L,
            HamR=HamR, semi_R=semi_R,
            left_orbs=left_orbs, right_orbs=right_orbs,
            eta=lead_eta,
            polarized_leads=(HamL.spin.is_polarized or HamR.spin.is_polarized),
        )
    kvals, kw = monkhorst_pack_grid(H_ref, nk=nk, axes=axes, trs=trs)

    spins = (0,) if method in ("rkky", "rkky_nm") else (0, 1)
    dos = np.zeros((len(spins), len(E)), dtype=float)
    orb_dos = (np.zeros((len(spins), len(E), len(selected)), dtype=float)
               if return_orbital_dos else None)

    rank = comm.Get_rank() if comm is not None else 0
    size = comm.Get_size() if comm is not None else 1
    for ik, kvec in enumerate(kvals):
        Sk = sparse.csc_matrix(H_ref.Sk(k=kvec, format="csr"), dtype=complex)
        # Match calculate_exchange: an unpolarized RKKY reference has no spin argument.
        if method in ("rkky", "rkky_nm") and not H_ref.spin.is_polarized:
            Hks = [sparse.csc_matrix(H_ref.Hk(k=kvec, format="csr"), dtype=complex)]
        else:
            Hks = [sparse.csc_matrix(
                H_ref.Hk(k=kvec, spin=spin, format="csr"), dtype=complex
            ) for spin in spins]

        for ie in range(rank, len(E), size):          # energies distributed over the ranks
            z = E[ie] + 1j * eta

            # RKKY: spin 0 lead reference. Liechtenstein: shared spin 0
            # or independently evaluated lead spin channels.
            shared = None
            if include_leads and method == "liechtenstein" and spin_independent_leads:
                shared = _lead_embedding(n, self_energy(z, kvec, 0))

            for ispin, (spin, Hk) in enumerate(zip(spins, Hks)):
                if not include_leads:
                    sigma = sparse.csc_matrix((n, n), dtype=complex)
                elif shared is not None:
                    sigma = shared
                else:
                    lead_spin = spin if method == "liechtenstein" else 0
                    sigma = _lead_embedding(n, self_energy(z, kvec, lead_spin))

                # IDENTICAL construction to exchange.factor():
                A = (z * Sk - Hk - sigma).tocsc()
                lu = splu(A)

                # diag(G @ S): solve A X = S[:, selected] in small batches.
                # Only requested orbital columns are calculated.
                for first in range(0, len(selected), rhs_batch):
                    last = min(first + rhs_batch, len(selected))
                    ids = selected[first:last]
                    rhs = Sk[:, ids].toarray()
                    X = lu.solve(rhs)
                    local_dos = -np.imag(X[ids, np.arange(len(ids))]) / np.pi
                    dos[ispin, ie] += kw[ik] * local_dos.sum()
                    if orb_dos is not None:
                        orb_dos[ispin, ie, first:last] += kw[ik] * local_dos

                if verbose:
                    print(f"k={ik+1}/{len(kvals)} E={ie+1}/{len(E)} "
                          f"spin={spin} DOS={dos[ispin, ie]:.6g}")

    if comm is not None and size > 1:
        from mpi4py import MPI
        tot = np.zeros_like(dos) if rank == 0 else None
        comm.Reduce(dos, tot, op=MPI.SUM, root=0)
        dos = tot
        if orb_dos is not None:
            tot_o = np.zeros_like(orb_dos) if rank == 0 else None
            comm.Reduce(orb_dos, tot_o, op=MPI.SUM, root=0)
            orb_dos = tot_o
        if rank != 0:
            return None
    return dict(energies=E, spins=("0",) if method in ("rkky", "rkky_nm") else ("up", "down"),
                dos=dos, orbital_dos=orb_dos, orbitals=selected,
                kpoints=kvals, k_weights=kw,
                is_total=(orbitals is None), include_leads=include_leads)


def get_orbital_names(H, orbital_indices):
    geom = H.geometry
    orbital_indices = np.asarray(orbital_indices, dtype=int)
    atoms = np.asarray(geom.o2a(orbital_indices), dtype=int)
    first_orbs = np.asarray(geom.a2o(atoms), dtype=int)
    names = [
        geom.atoms[int(a)].orbitals[int(o - first)].name()
        for o, a, first in zip(orbital_indices, atoms, first_orbs)
    ]
    return np.asarray(names, dtype=str)