# -*- coding: utf-8 -*-
"""Device builders for exchange calculations in GNRs.

Every builder returns a `Device`: the open device Hamiltonian, the two
electrodes, the electrode interface orbitals, the two magnetic vertices
(Delta_i, orb_i, Delta_j, orb_j) and the defect positions. Then

    device = builder.build(method)
    result = calculate_exchange(**device.exchange_kwargs(method),
                                energies=En, contour_weights=weights, ...)

Methods and builders
--------------------
The builder defines where the data come from; the method defines the kernel.

  method              builder             propagator            vertices
  'rkky'              PristineTiled       pristine G0           own Delta, single-defect files
  'rkky_sup'          SuperpositionDyson  G0 + V1 + V2 (Dyson)  own Delta, single-defect files
  'liechtenstein_sup' SuperpositionDyson  G0 + U1s + U2s        own Delta (default) or total
  'rkky_nm'           DFTDevice           NM two-defect DFT     FM two-defect DFT (default)
                                                                or single-defect files
  'liechtenstein'     DFTDevice           FM two-defect DFT     FM two-defect DFT

Aliases: 'lkag' = 'liechtenstein', 'lkag_sup' = 'liechtenstein_sup'.
SuperpositionDyson lives in Dyson_Exchange.py. Superposition (explicit H)
solves 'rkky_sup' and 'liechtenstein_sup' by direct inversion; it is the
reference for the Dyson engine and gives an explicit device for the DOS.

PristineTiled  (method 'rkky')
    Pristine host H_0 tiled along the transport axis. The vertices are the
    single-defect Delta = H_up - H_dn from extract_single_Delta (unchanged
    from the current RKKY script). G is the pristine G_0.
    downfold=True folds each H adatom into its carbons through
    calculate_exchange(..., downfold=True); the H blocks are first shifted
    into the host energy frame (H -> H + delta*S, delta from the carbons far
    from the defect).

Superposition  (methods 'rkky_sup', 'liechtenstein_sup')
    Pristine host plus the single-defect perturbations of both defects,
        H_s = H_0 + sum_d [ V_d + s * Delta_d / 2 ],   s = +1 (up), -1 (down)
    where V_d = (H_up + H_dn)/2 - H_0 is the scalar (nonmagnetic) part,
    aligned to the host far from the defect. Adatom atoms are inserted.
    'rkky_sup' uses G of H_0 + sum V_d; 'liechtenstein_sup' uses G_up, G_down.
    No two-defect DFT calculation is needed for a given n_dist.

DFTDevice  (methods 'liechtenstein', 'rkky_nm')
    Two-defect DFT Hamiltonians, pruned and aligned to the electrodes.
    'liechtenstein': G and Delta from the spin-polarized (FM) device.
    'rkky_nm': G from the nonmagnetic device; Delta from the FM device
    (vertex_source='fm', default) or from the two single-defect SP
    calculations mapped onto it (vertex_source='single'). Adatoms are
    explicit; no downfolding. With FM vertices the regions must be disjoint
    for both methods.

compute_dos(obj, method, energies) gives the DOS for every builder.

Host / electrode Hamiltonians may be nonmagnetic or spin polarized; a
spin-polarized pristine host is accepted if its spin splitting is negligible
and is then replaced by its spin average.
"""
from __future__ import annotations

import traceback
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import sisl
from scipy import sparse
import scipy.sparse.linalg  # noqa: F401  (sparse.linalg.norm)
from scipy.spatial import cKDTree

from defects_dependencies import (Adatom, Heteroatom, ResolvedDefect, detect_adatoms,
                     detect_heteroatoms, keep_orbitals_for, pruned_index)

__all__ = ["Device", "PristineTiled", "Superposition", "DFTDevice",
           "map_single_defect_vertex", "nonmagnetic", "electrode_tiling",
           "build_on_root", "compute_dos", "canonical_method", "METHOD_ALIASES"]

# Short names accepted everywhere
METHOD_ALIASES = {"lkag": "liechtenstein", "lkag_sup": "liechtenstein_sup"}
# Kernel of calculate_exchange used for each method (same kernel, other reference)
KERNEL = {"rkky": "rkky", "rkky_nm": "rkky_nm", "liechtenstein": "liechtenstein",
          "rkky_sup": "rkky_nm", "liechtenstein_sup": "liechtenstein"}


def canonical_method(method: str) -> str:
    """Lower-case method name with the aliases ('lkag', 'lkag_sup') resolved."""
    m = str(method).lower().strip()
    return METHOD_ALIASES.get(m, m)


def _JD():
    """J_dependencies is imported lazily (it pulls matplotlib/ase)."""
    import J_dependencies
    return J_dependencies


# ===========================================================================
# Hamiltonian utilities
# ===========================================================================
def _check_open_axes(H, axis: int, tol: float = 1e-6, label: str = "") -> None:
    """Raise if H couples to periodic images along a non-transport axis."""
    sc_off = np.asarray(H.geometry.lattice.sc_off)
    no = H.no
    bad = [ax for ax in range(3) if ax != axis and H.nsc[ax] > 1]
    if not bad:
        return
    for dim in range(H.dim):
        M = H.tocsr(dim).tocsc()
        for isc, off in enumerate(sc_off):
            if any(off[ax] != 0 for ax in bad):
                block = M[:, isc * no:(isc + 1) * no]
                if block.nnz and np.max(np.abs(block.data)) > tol:
                    raise ValueError(
                        f"{label}: couplings to periodic images along axis "
                        f"{bad} (offset {tuple(off)}); the ribbon must be "
                        "isolated in the non-transport directions.")


def nonmagnetic(H, tol: float = 1e-3, label: str = "Hamiltonian"):
    """Spin average of a polarized H, after checking it is nonmagnetic.

    Unpolarized Hamiltonians are returned unchanged. Raises if
    max |H_up - H_down| > tol (eV) anywhere, including periodic couplings.
    """
    if not H.spin.is_polarized:
        return H
    up, dn = H.tocsr(0), H.tocsr(1)
    diff = (up - dn).tocsr()
    split = float(np.max(np.abs(diff.data))) if diff.nnz else 0.0
    if split > tol:
        raise ValueError(f"{label} is magnetic: max |H_up - H_down| = "
                         f"{split * 1000:.3f} meV > {tol * 1000:.3f} meV")
    S = H.tocsr(H.dim - 1) if H.non_orthogonal else None
    return sisl.Hamiltonian.fromsp(H.geometry, 0.5 * (up + dn), S=S)


def electrode_tiling(H0, axis: int = 0, max_tile: int = 10) -> int:
    """Smallest tiling of H0 with nsc <= 3 along the transport axis."""
    for n in range(1, max_tile + 1):
        if H0.tile(n, axis).nsc[axis] <= 3:
            return n
    raise ValueError(f"H0 still couples beyond nearest cells after {max_tile} tiles")


def _semi(axis: int) -> Tuple[str, str]:
    letter = "ABC"[axis]
    return f"-{letter}", f"+{letter}"


def _section(title: str) -> None:
    """Section header, same style as align_defect_to_electrodes."""
    print(f"\n========== {title} ==========", flush=True)


def _close() -> None:
    print("========================================\n", flush=True)


def _spin_splitting(H) -> float:
    """max |H_up - H_down| over all matrix elements (0 for an unpolarized H)."""
    if not H.spin.is_polarized:
        return 0.0
    diff = (H.tocsr(0) - H.tocsr(1)).tocsr()
    return float(np.max(np.abs(diff.data))) if diff.nnz else 0.0


def _max_abs(D) -> float:
    D = D.toarray() if sparse.issparse(D) else np.asarray(D)
    return float(np.max(np.abs(D))) if D.size else 0.0


def _report_vertices(region_text: str, vertex_sets: dict, labels=("i", "j")) -> None:
    """Vertices of every method: orbitals, max |Delta|, overlap."""
    _section("Vertices")
    print(f"{'Region:':26s}{region_text}")
    for name, (Ds, os_) in vertex_sets.items():
        for k, (D, o) in enumerate(zip(Ds, os_)):
            print(f"{name + ' ' + labels[k] + ':':26s}{len(o):5d} orbitals, max |Delta| {_max_abs(D):.4f} eV")
        print(f"{name + ' overlap:':26s}{np.intersect1d(os_[0], os_[1]).size:5d} orbitals")


def _supercell_ends(g, H_mats, S_def, shift, amap, shift_xy, host_mats, S_host, host_geom,
                    axis: int, cell_length: float, electrode_atoms=None) -> dict:
    """Electrode-region check of a single-defect Hamiltonian (after its energy alignment).

    Returns, for 'left' and 'right', the end cell of the supercell, and for
    'left_electrode' / 'right_electrode' the atoms of the supercell placed in the
    electrode cells of the device (electrode_atoms = (host atoms left, right)).

    g, H_mats, S_def : pruned single-defect geometry and central-cell matrices
    shift            : energy alignment of the file (H -> H + shift * S)
    amap, shift_xy   : single-defect atom -> host atom (-1: outside), in-plane translation
    host_mats, S_host, host_geom : pristine host (central-cell matrices)
    The end cells are the heavy atoms within one pristine cell length of the two
    ends of the supercell along the transport axis: the parts next to the
    electrodes, where the spin cloud and the defect potential must have decayed.
    """
    a = np.asarray(g.cell[axis], dtype=float)
    a /= np.linalg.norm(a)
    heavy = np.isin(g.atoms.Z, (5, 6, 7, 8, 9))
    proj = g.xyz @ a
    lo, hi = proj[heavy].min(), proj[heavy].max()
    D = (H_mats[0] - H_mats[1]).tocsr()
    S_eff = S_def if S_def is not None else sparse.identity(D.shape[0], format="csr")
    avg = (0.5 * (H_mats[0] + H_mats[1]) + shift * S_eff).tocsr()
    out = {}
    regions = [("left", heavy & (proj < lo + cell_length - 1e-6)),
               ("right", heavy & (proj > hi - cell_length + 1e-6))]
    if electrode_atoms is not None:
        for side, host_atoms in zip(("left", "right"), electrode_atoms):
            regions.append((f"{side}_electrode", heavy & np.isin(amap, np.asarray(host_atoms))))
    for side, mask in regions:
        atoms = np.flatnonzero(mask)
        if len(atoms) == 0:
            out[side] = dict(n_atoms=0, n_outside=0, spin=0.0, dH=np.nan, dS=np.nan, mismatch=np.nan)
            continue
        o = np.concatenate([g.a2o(x, all=True) for x in atoms])
        rows = D[o]
        spin = float(np.max(np.abs(rows.data))) if rows.nnz else 0.0
        mapped = atoms[amap[atoms] >= 0]
        dH = dS = mism = np.nan
        if len(mapped):
            od = np.concatenate([g.a2o(x, all=True) for x in mapped])
            oh = np.concatenate([host_geom.a2o(amap[x], all=True) for x in mapped])
            if len(od) == len(oh):
                dH = float(np.max(np.abs(avg[od][:, od].toarray() - host_mats[0][oh][:, oh].toarray())))
                if S_def is not None and S_host is not None:
                    dS = float(np.max(np.abs(S_def[od][:, od].toarray() - S_host[oh][:, oh].toarray())))
            mism = float(np.max(np.linalg.norm(g.xyz[mapped, :2] + shift_xy
                                               - host_geom.xyz[amap[mapped], :2], axis=1)))
        out[side] = dict(n_atoms=len(atoms), n_outside=int(len(atoms) - len(mapped)),
                         spin=spin, dH=dH, dS=dS, mismatch=mism)
    return out


def _check_messages(what: str, e: dict, tol: float, overlap_tol: float) -> list:
    """Warnings for one region: spin splitting / |dH| above tol (eV), |dS| above overlap_tol."""
    msgs = []
    if e["spin"] > tol:
        msgs.append(f"{what}: spin splitting {1000 * e['spin']:.3f} meV (> {1000 * tol:.1f} meV); "
                    "the spin cloud has not decayed there")
    if not np.isnan(e["dH"]) and e["dH"] > tol:
        msgs.append(f"{what}: |dH| {1000 * e['dH']:.3f} meV after the energy alignment "
                    f"(> {1000 * tol:.1f} meV)")
    if not np.isnan(e["dS"]) and e["dS"] > overlap_tol:
        msgs.append(f"{what}: |dS| {e['dS']:.3e} (> {overlap_tol:.1e})")
    return msgs


def _warn(msgs, verbose: bool) -> None:
    """Print (verbose) and raise every warning; given also without the report."""
    for m in msgs:
        if verbose:
            print("WARNING: " + m, flush=True)
        warnings.warn(m)


def _fmt(e: dict) -> str:
    dH = "n/a" if np.isnan(e["dH"]) else f"{1000 * e['dH']:.3f} meV"
    dS = "n/a" if np.isnan(e["dS"]) else f"{e['dS']:.3e}"
    return f"max |Hup-Hdn| {1000 * e['spin']:.3f} meV, max |dH| {dH}, max |dS| {dS}"


def _report_defect_electrodes(entries, tails: dict, tol: float, overlap_tol: float,
                              verbose: bool = True) -> list:
    """Electrode regions of the single-defect Hamiltonians: report (verbose) and warnings.

    entries: [(label, regions of _supercell_ends)]. The electrode cells of the device are
    pristine; what matters is that each single-defect Hamiltonian, after its energy
    alignment, has decayed in the part placed in the electrode cells (if any) and in the
    end cells of its supercell, next to the electrodes.
    """
    if verbose:
        _section("Electrode regions (single-defect Hamiltonians)")
        print("(the electrode cells of the device are pristine; checked is each single-defect")
        print(" Hamiltonian after its energy alignment: the part placed in the electrode cells,")
        print(" if any, and the end cells of its supercell next to the electrodes)")
        for k, (label, reg) in enumerate(entries):
            print(f"Defect {'ij'[k]} ({label})")
            for side in ("left", "right"):
                e = reg.get(f"{side}_electrode")
                if e is None or e["n_atoms"] == 0:
                    print(f"  {side} electrode cells:".ljust(26) + "no atom of the supercell there (pristine)")
                else:
                    print(f"  {side} electrode cells:".ljust(26) + _fmt(e))
                    print(f"{'':26s}{e['n_atoms']} atoms of the supercell in the electrode cells")
                e = reg[side]
                print(f"  {side} supercell end:".ljust(26) + _fmt(e))
                print(f"{'':26s}{e['n_atoms']} atoms, {e['n_outside']} outside the device, "
                      f"geometry mismatch {e['mismatch']:.4f} Ang")
        print("Vertex splitting on electrode orbitals")
        for name, vals in tails.items():
            print(f"  {name}:".ljust(26) + ", ".join(f"{'ij'[k]} {1000 * v:.3f} meV" for k, v in enumerate(vals)))
    msgs = []
    for k, (label, reg) in enumerate(entries):
        for side in ("left", "right"):
            msgs += _check_messages(f"defect {'ij'[k]} ({label}), {side} end of its single-defect "
                                    "supercell", reg[side], tol, overlap_tol)
            e = reg.get(f"{side}_electrode")
            if e is not None and e["n_atoms"]:
                msgs += _check_messages(f"defect {'ij'[k]} ({label}), part of its single-defect "
                                        f"supercell in the {side} electrode cells", e, tol, overlap_tol)
    msgs += [f"vertex {'ij'[k]} of {name}: splitting {1000 * v:.3f} meV on electrode orbitals "
             f"(> {1000 * tol:.1f} meV)" for name, vals in tails.items()
             for k, v in enumerate(vals) if v > tol]
    _warn(msgs, verbose)
    return msgs


def _electrode_regions(H, HamL, HamR, left_orbs, right_orbs) -> dict:
    """Device cells coupled to the electrodes vs the electrodes (intra-cell, after alignment).

    spin: max |H_up - H_down| on the rows of the interface orbitals (all couplings);
    dH, dS: max difference between the device block of the interface orbitals and the
    central-cell block of the electrode (spin average for a polarized device).
    """
    mats, S = _R0_blocks(H)
    avg = (0.5 * (mats[0] + mats[1])).tocsr()
    diff = (mats[0] - mats[1]).tocsr()
    out = {}
    for side, orbs, elec in (("left", left_orbs, HamL), ("right", right_orbs, HamR)):
        o = np.asarray(orbs, dtype=int)
        e_mats, e_S = _R0_blocks(elec)
        e_avg = (0.5 * (e_mats[0] + e_mats[1])).toarray()
        rows = diff[o]
        spin = float(np.max(np.abs(rows.data))) if rows.nnz else 0.0
        dH = float(np.max(np.abs(avg[o][:, o].toarray() - e_avg))) if e_avg.shape[0] == len(o) else np.nan
        dS = (float(np.max(np.abs(S[o][:, o].toarray() - e_S.toarray())))
              if S is not None and e_S is not None and e_S.shape[0] == len(o) else np.nan)
        out[side] = dict(spin=spin, dH=dH, dS=dS, n_orbitals=len(o))
    return out


def _vertex_tails(vertex_sets: dict, left_orbs, right_orbs) -> dict:
    """max |Delta| of every vertex on electrode-interface orbitals."""
    elec = np.union1d(np.asarray(left_orbs), np.asarray(right_orbs))
    tails = {}
    for name, (Ds, os_) in vertex_sets.items():
        vals = []
        for D, o in zip(Ds, os_):
            mask = np.isin(np.asarray(o), elec)
            if mask.any():
                Dd = D.toarray() if sparse.issparse(D) else np.asarray(D)
                vals.append(float(np.max(np.abs(Dd[mask]))))
            else:
                vals.append(0.0)
        tails[name] = vals
    return tails


def _report_electrode_regions(results: dict, tails: dict, tol: float, overlap_tol: float,
                              verbose: bool = True, note: Optional[str] = None) -> list:
    """Electrode regions of every reference device: report (verbose) and warnings."""
    if verbose:
        _section("Electrode regions")
        print("(device cells coupled to the electrodes, after the energy alignment;")
        print(" compared with the central cell of the electrode)")
        for name, res in results.items():
            print(f"{name} (interface orbitals: L={res['left']['n_orbitals']}, "
                  f"R={res['right']['n_orbitals']})")
            for side in ("left", "right"):
                print(f"  {side}:".ljust(26) + _fmt(res[side]))
        print("Vertex splitting on electrode orbitals")
        for name, vals in tails.items():
            print(f"  {name}:".ljust(26) + ", ".join(f"{'ij'[k]} {1000 * v:.3f} meV" for k, v in enumerate(vals)))
        if note:
            print(note)
    msgs = [m for name, res in results.items() for side in ("left", "right")
            for m in _check_messages(f"{name}, {side} electrode region", res[side], tol, overlap_tol)]
    msgs += [f"vertex {'ij'[k]} of {name}: splitting {1000 * v:.3f} meV on electrode orbitals "
             f"(> {1000 * tol:.1f} meV)" for name, vals in tails.items()
             for k, v in enumerate(vals) if v > tol]
    _warn(msgs, verbose)
    return msgs


def _R0_blocks(H):
    """Central-cell matrices: ([H_up, H_down] or [H, H], S or None)."""
    Hl = H.copy()
    Hl.set_nsc([1, 1, 1])
    mats = ([Hl.tocsr(0).tocsr(), Hl.tocsr(1).tocsr()] if Hl.spin.is_polarized
            else [Hl.tocsr(0).tocsr()] * 2)
    S = Hl.tocsr(Hl.dim - 1).tocsr() if Hl.non_orthogonal else None
    return mats, S


def _reindex(M, index, n):
    """Sparse matrix M re-indexed by index (old -> new) into an n x n matrix."""
    C = M.tocoo()
    return sparse.coo_matrix((C.data, (index[C.row], index[C.col])),
                             shape=(n, n)).tocsr()


# ===========================================================================
# Device
# ===========================================================================
@dataclass(eq=False)
class Device:
    """Everything calculate_exchange needs for one defect pair."""
    kind: str
    methods: Tuple[str, ...]
    H: "sisl.Hamiltonian"                  # main reference (NM for rkky, SP otherwise)
    HamL: "sisl.Hamiltonian"
    HamR: "sisl.Hamiltonian"
    semi_L: str
    semi_R: str
    left_orbs: np.ndarray
    right_orbs: np.ndarray
    defects: Tuple[ResolvedDefect, ResolvedDefect]
    anchors: Tuple[int, int]               # device atom indices of the anchors
    Deltas: List[sparse.csr_matrix]
    orbs: List[np.ndarray]
    distance: float                        # center-center (Angstrom)
    anchor_distance: float                 # anchor-anchor (Angstrom)
    H_nm: Optional["sisl.Hamiltonian"] = None   # nonmagnetic reference if separate
    info: Dict = field(default_factory=dict)
    nm_tol: float = 1e-3
    # Vertices per method: {method: (Deltas, orbs)}. Deltas/orbs above are the
    # set of the method the device was built for (used by the plots).
    vertex_sets: Dict = field(default_factory=dict)
    # Extra calculate_exchange arguments of 'rkky' with downfolded adatoms
    # (downfold, hydrogen_i/j, mapped_c_orbs_i/j); empty without downfolding.
    downfold_kwargs: Dict = field(default_factory=dict)

    def _check(self, method: str) -> str:
        method = canonical_method(method)
        if method not in self.methods:
            raise ValueError(f"{self.kind} device supports {self.methods}, not '{method}'")
        return method

    def vertices(self, method: str):
        """(Deltas, orbs) used by `method`."""
        method = canonical_method(method)
        if method in self.vertex_sets:
            return self.vertex_sets[method]
        return self.Deltas, self.orbs

    def reference(self, method: str):
        """Reference Hamiltonian for the propagators of `method`."""
        method = self._check(method)
        if KERNEL[method] == "liechtenstein":
            if not self.H.spin.is_polarized:
                raise ValueError("liechtenstein needs a spin-polarized reference")
            return self.H
        if self.H_nm is not None:
            return self.H_nm
        return nonmagnetic(self.H, self.nm_tol, f"{self.kind} reference")

    def exchange_kwargs(self, method: str, distance: str = "anchor") -> dict:
        """Device-dependent keyword arguments of calculate_exchange."""
        if distance not in ("anchor", "center"):
            raise ValueError("distance must be 'anchor' or 'center'")
        H_ref = self.reference(method)
        method = canonical_method(method)
        Deltas, orbs = self.vertices(method)
        no, nl, nr = H_ref.no, self.HamL.no, self.HamR.no
        # Pass None for the default first/last-block convention, so existing
        # result folders (config.json) remain compatible.
        default = (np.array_equal(self.left_orbs, np.arange(nl))
                   and np.array_equal(self.right_orbs, np.arange(no - nr, no)))
        extra = dict(self.downfold_kwargs) if method == "rkky" else {}
        return dict(
            # calculate_exchange knows the kernels 'rkky', 'rkky_nm', 'liechtenstein'
            method=KERNEL[method], H_ref=H_ref,
            Delta_i=Deltas[0], orb_i=orbs[0],
            Delta_j=Deltas[1], orb_j=orbs[1],
            HamL=self.HamL, semi_L=self.semi_L,
            HamR=self.HamR, semi_R=self.semi_R,
            left_orbs=None if default else self.left_orbs,
            right_orbs=None if default else self.right_orbs,
            distance=self.anchor_distance if distance == "anchor" else self.distance,
            **extra,
        )

    def summary(self) -> str:
        lines = [f"{self.kind} device: {self.H.na} atoms, {self.H.no} orbitals, "
                 f"spin {'polarized' if self.H.spin.is_polarized else 'unpolarized'}",
                 f"  methods: {', '.join(self.methods)}",
                 f"  electrodes: {self.HamL.no} + {self.HamR.no} orbitals ({self.semi_L}/{self.semi_R})"]
        sets = self.vertex_sets or {self.methods[0]: (self.Deltas, self.orbs)}
        for m, (Ds, os_) in sets.items():
            lines.append(f"  vertices for {m}:")
            for d, a, D, o in zip(self.defects, self.anchors, Ds, os_):
                mx = float(np.max(np.abs(D.data))) if D.nnz else 0.0
                lines.append(f"    {d.label}: anchor atom {a}, {len(o)} vertex orbitals, "
                             f"max|Delta| = {mx:.4f} eV")
        lines.append(f"  distance: {self.anchor_distance:.3f} Ang (anchors), "
                     f"{self.distance:.3f} Ang (centers)")
        for k, v in self.info.items():
            if np.isscalar(v) or isinstance(v, (tuple, str)):
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)

    def write_geometry(self, filename) -> None:
        self.H.geometry.write(str(filename))


def _distances(geometry, anchors, resolved):
    xyz = geometry.xyz
    anchor = float(np.linalg.norm(xyz[anchors[1]] - xyz[anchors[0]]))
    ri = xyz[anchors[0]] + resolved[0].center_offset
    rj = xyz[anchors[1]] + resolved[1].center_offset
    return float(np.linalg.norm(rj - ri)), anchor


def build_on_root(builder, method: str, comm=None) -> Device:
    """Build on MPI rank 0 and broadcast; errors are re-raised on ALL ranks.

    Avoids the deadlock where rank 0 fails while other ranks wait in bcast.
    """
    if comm is None or comm.Get_size() == 1:
        return builder.build(method)
    device, error = None, None
    if comm.Get_rank() == 0:
        try:
            device = builder.build(method)
        except Exception:
            error = traceback.format_exc()
    device, error = comm.bcast((device, error), root=0)
    if error:
        raise RuntimeError("Device build failed on rank 0:\n" + error)
    return device


# ===========================================================================
# Pristine tiled host (RKKY)
# ===========================================================================
class PristineTiled:
    """Pristine host tiled along the transport axis; vertices from single defects.

    H0        : pruned, energy-aligned pristine cell (prune_ham output);
                unpolarized, or polarized with negligible splitting.
    defects   : (spec_i, spec_j), Adatom/Heteroatom from defects.py.
    n_dist    : cells between the two anchors.
    basis     : basis used to prune H0 ('pz', 'pz1', 'pz+dz', ...).
    N_imp     : None -> automatic layout from the defect supercell sizes.
                int  -> legacy layout N = 2*(N_imp+N_buffer)+1+n_dist,
                        anchor i in cell N_imp+N_buffer (current scripts).
    N_buffer  : extra pristine cells between defect supports and electrodes.
    n_tile    : electrode cells; None -> smallest with nsc <= 3.
    region, radius, atom_mask : magnetic vertex region (extract_single_Delta).
    downfold  : True -> fold each H adatom into the carbons it couples to
                (calculate_exchange(..., downfold=True)). Both defects must be
                adatoms. The single-defect Hamiltonian is first shifted into the
                host energy frame, H -> H + delta*S, with delta fitted on the
                carbons far from the defect (align_radius; None -> farthest half).
    ends_tol  : spin splitting and |dH| (eV) allowed in the end cells of each
                single-defect supercell and in the electrode regions of the device
                (after the energy alignment); above it a warning is given.
    overlap_tol : the same for |dS|.
    """

    kind = "pristine_tiled"
    methods = ("rkky",)

    def __init__(self, H0, defects, n_dist: int, basis: str = "pz", axis: int = 0,
                 N_imp: Optional[int] = None, N_buffer: int = 1,
                 n_tile: Optional[int] = None, tol: float = 0.25,
                 site_tol: float = 0.3, region: str = "all", radius: float = 3.5,
                 atom_mask=None, nm_tol: float = 1e-3, verbose: bool = True,
                 downfold: bool = False, align_radius: Optional[float] = None,
                 ends_tol: float = 2*1e-3, overlap_tol: float = 1e-3):
        if len(defects) != 2:
            raise ValueError("defects must contain two specs (i, j)")
        if int(n_dist) < 0:
            raise ValueError("n_dist must be >= 0")
        self.h0_polarized = H0.spin.is_polarized
        self.h0_splitting = _spin_splitting(H0)
        self.H0 = nonmagnetic(H0, nm_tol, "Pristine host H0")
        _check_open_axes(self.H0, axis, label="Pristine host H0")
        self.specs = tuple(defects)
        self.n_dist = int(n_dist)
        self.basis, self.axis = basis, int(axis)
        self.N_imp, self.N_buffer = N_imp, int(N_buffer)
        self.n_tile = electrode_tiling(self.H0, axis) if n_tile is None else int(n_tile)
        self.tol, self.site_tol = tol, site_tol
        self.region, self.radius, self.atom_mask = region, radius, atom_mask
        self.nm_tol, self.verbose = nm_tol, verbose
        self.downfold, self.align_radius = bool(downfold), align_radius
        self.ends_tol, self.overlap_tol = ends_tol, overlap_tol

    # -- layout ------------------------------------------------------------
    def _support_atoms(self, spec, geometry, anchor):
        """Atoms (full DFT file) whose extent the device must contain."""
        heavy = np.flatnonzero(np.isin(geometry.atoms.Z, (5, 6, 7, 8, 9)))
        return np.union1d(heavy, np.asarray(list(spec.adatoms()), dtype=int))

    def _extent_cells(self, spec) -> Tuple[int, int]:
        """Cells needed left/right of the anchor cell for this defect."""
        from defects_dependencies import read_geometry
        r = spec.resolve()
        g = read_geometry(spec.file)
        atoms = self._support_atoms(spec, g, r.anchor)
        a = np.asarray(self.H0.cell[self.axis], dtype=float)
        length = np.linalg.norm(a)
        proj = (g.xyz[atoms] - r.anchor_xyz) @ (a / length)
        span = proj.max() - proj.min()
        if span > 0:
            frac = (r.anchor_xyz @ (a / length) - (g.xyz[atoms] @ (a / length)).min()) / span
            if not 0.2 <= frac <= 0.8 and self.verbose:
                warnings.warn(f"{spec.label}: anchor is near the edge of its DFT "
                              "supercell; Delta tails on one side may be truncated.")
        left = int(np.ceil(max(0.0, -proj.min()) / length - 1e-6))
        right = int(np.ceil(max(0.0, proj.max()) / length - 1e-6))
        return left, right

    def layout(self) -> Dict[str, int]:
        """Number of cells N and anchor cells (c_i, c_j)."""
        if self.N_imp is not None:
            c_i = int(self.N_imp) + self.N_buffer
            N = 2 * (int(self.N_imp) + self.N_buffer) + 1 + self.n_dist
        else:
            (li, ri), (lj, rj) = (self._extent_cells(s) for s in self.specs)
            margin = self.n_tile + self.N_buffer
            c_i = margin + max(li, lj - self.n_dist)
            N = c_i + self.n_dist + max(rj, ri - self.n_dist) + margin + 1
        return dict(N=N, c_i=c_i, c_j=c_i + self.n_dist, n_tile=self.n_tile)

    def _electrode_atoms(self, lay):
        """Host atoms of the left and right electrode cells of the tiled device."""
        na0, nt = self.H0.na, self.n_tile
        return (np.arange(nt * na0), np.arange((lay["N"] - nt) * na0, lay["N"] * na0))

    def _host(self, lay):
        H = self.H0.tile(lay["N"], self.axis)
        H.set_nsc([1, 1, 1])
        anchors = [spec.host_anchor(self.H0.geometry, cell=c, axis=self.axis,
                                    tol=self.site_tol)
                   for spec, c in zip(self.specs, (lay["c_i"], lay["c_j"]))]
        return H, anchors

    def _electrodes(self):
        elec = self.H0.tile(self.n_tile, self.axis)
        return elec, elec.copy()

    # -- downfolding ---------------------------------------------------------
    def _map_single(self, spec, H, host_anchor) -> dict:
        """Single-defect file mapped onto the device and aligned to the host.

        The adatom is removed (as in rkky). Returns the full and pruned
        Hamiltonians, the atom map (pruned -> device atom), the central-cell
        matrices, the energy shift (fitted on the far carbons) and its residual.
        """
        JD = _JD()
        r = spec.resolve()
        H_full = sisl.get_sile(str(spec.file)).read_hamiltonian()
        # Pruned single-defect Hamiltonian without the adatom: mapping and alignment
        H_def, orig = JD.prepare_spin_basis(H_full, basis=self.basis,
                                            remove_atoms=tuple(spec.adatoms()))
        g, gh = H_def.geometry, H.geometry
        a_def = pruned_index(orig, r.anchor)
        shift_xy = gh.xyz[host_anchor, :2] - g.xyz[a_def, :2]
        tree = cKDTree(gh.xyz[:, :2])
        amap = np.full(g.na, -1, dtype=int)
        for a in range(g.na):
            d, h = tree.query(g.xyz[a, :2] + shift_xy)
            if d <= self.tol:
                amap[a] = int(h)
        amap[a_def] = int(host_anchor)

        # Energy shift: on-site energies of far carbons must match the host
        mats, S_def = _R0_blocks(H_def)
        host_diag = _R0_blocks(H)[0][0].diagonal()
        dist = np.linalg.norm(g.xyz - g.xyz[a_def], axis=1)
        carbons = np.array([a for a in range(g.na) if amap[a] >= 0 and g.atoms.Z[a] == 6])
        rad = np.median(dist[carbons]) if self.align_radius is None else self.align_radius
        far = carbons[dist[carbons] >= rad]
        if len(far) < 2:
            raise ValueError(f"{spec.label}: too few carbons for the energy alignment")
        o_def = np.concatenate([g.a2o(a, all=True) for a in far])
        o_host = np.concatenate([gh.a2o(amap[a], all=True) for a in far])
        if len(o_def) != len(o_host):
            raise ValueError(f"{spec.label}: orbital basis differs from the host")
        h_def = 0.5 * (mats[0].diagonal() + mats[1].diagonal())[o_def]
        s_def = S_def.diagonal()[o_def] if S_def is not None else np.ones(len(o_def))
        delta = host_diag[o_host] - h_def
        shift = float(np.vdot(s_def, delta).real / np.vdot(s_def, s_def).real)
        rms = float(np.sqrt(np.mean((delta - shift * s_def) ** 2)))
        return dict(H_full=H_full, H_def=H_def, orig=orig, amap=amap, mats=mats, S_def=S_def,
                    shift=shift, rms=rms, shift_xy=shift_xy)

    def _downfold_data(self, spec, H, host_anchor, m=None):
        """Aligned downfolding data of one adatom, and its carbons on the device.

        Returns (hydrogen data of prepare_hydrogen_downfold, device orbitals of
        the coupled carbons in the same order, energy shift, fit residual).
        """
        JD = _JD()
        if spec.kind != "adatom":
            raise ValueError(f"{spec.label}: downfold=True needs adatom defects")
        m = self._map_single(spec, H, host_anchor) if m is None else m
        H_full, orig, amap, shift, rms = m["H_full"], m["orig"], m["amap"], m["shift"], m["rms"]
        if not H_full.spin.is_polarized:
            raise ValueError(f"{spec.label}: single-defect file must be spin polarized")
        gh = H.geometry

        # Hydrogen blocks of the shifted single-defect Hamiltonian
        H_al = H_full.copy()
        H_al.shift(shift)                                   # H -> H + shift * S
        hdata = JD.prepare_hydrogen_downfold(H_al, spec.index, basis=self.basis)

        # Coupled carbons: single-defect orbitals -> device orbitals (same order)
        mapped = []
        gf = H_full.geometry
        for o in np.asarray(hdata["c_orbs"], dtype=int):
            ia = int(gf.o2a(o))
            name = gf.atoms[ia].orbitals[int(o - gf.firsto[ia])].name()
            h = amap[pruned_index(orig, ia)]
            if h < 0:
                raise ValueError(f"{spec.label}: carbon coupled to the adatom is outside the device")
            names = [orb.name() for orb in gh.atoms[h].orbitals]
            if name not in names:
                raise ValueError(f"{spec.label}: orbital {name} missing on device atom {h}")
            mapped.append(int(gh.firsto[h] + names.index(name)))
        return hdata, np.array(mapped, dtype=int), shift, rms

    # -- reports ---------------------------------------------------------------
    def _region_text(self) -> str:
        return self.region + (f" (R = {self.radius:.2f} Ang)" if self.region == "radius" else "")

    def _report_host(self, lay, H, device_anchors, dist, adist, left_orbs, right_orbs) -> None:
        """Pristine host, device layout and defect placement."""
        _section("Pristine host")
        print(f"Cell:               {self.H0.na} atoms, {self.H0.no} orbitals (basis {self.basis})")
        print(f"Spin polarized:     {self.h0_polarized} (max |Hup-Hdn|: {1000 * self.h0_splitting:.3f} meV"
              + (", spin averaged)" if self.h0_polarized else ")"))
        nsc = self.H0.tile(self.n_tile, self.axis).nsc[self.axis]
        print(f"Electrode tiling:   {self.n_tile} cell(s), nsc along transport: {nsc}")

        _section("Device layout")
        layout = ("automatic" if self.N_imp is None
                  else f"N_imp={self.N_imp}, N_buffer={self.N_buffer}")
        print(f"Layout:             {layout}")
        print(f"Cells:              N={lay['N']}, anchor cells c_i={lay['c_i']}, c_j={lay['c_j']}")
        print(f"Device:             {H.na} atoms, {H.no} orbitals")
        print(f"Interface orbitals: L={len(left_orbs)}, R={len(right_orbs)}")

        _section("Defect placement")
        for k, (spec, a) in enumerate(zip(self.specs, device_anchors)):
            r = spec.resolve()
            site = spec.locate_host_site(self.H0.geometry, axis=self.axis, tol=self.site_tol)
            print(f"Defect {'ij'[k]}:           {r.label} ({r.kind}), anchor {r.anchor} in its file "
                  f"-> host site {site} -> device atom {a}")
        print(f"Distance:           {adist:.3f} Ang (anchors), {dist:.3f} Ang (centers)")

    @staticmethod
    def _report_downfold(labels, data) -> None:
        """Aligned hydrogen blocks of PristineTiled(downfold=True)."""
        _section("H downfolding")
        for k, (label, (hdata, mapped, shift, rms)) in enumerate(zip(labels, data)):
            h_up = np.real(np.diag(hdata[0]["H_HH"])).mean()
            h_dn = np.real(np.diag(hdata[1]["H_HH"])).mean()
            print(f"Defect {'ij'[k]} ({label}):".ljust(20)
                  + f"shift {shift:+.6f} eV, RMS residual {1000 * rms:.3f} meV")
            print(f"{'':20s}{len(hdata['h_orbs'])} H orbital(s), {len(mapped)} coupled carbon orbital(s), "
                  f"H on-site (mean) up {h_up:+.3f} / down {h_dn:+.3f} eV")

    # -- build ---------------------------------------------------------------
    def build(self, method: str = "rkky") -> Device:
        method = canonical_method(method)
        if method not in self.methods:
            raise ValueError(f"PristineTiled supports {self.methods}; use SuperpositionDyson "
                             "(rkky_sup, liechtenstein_sup) or DFTDevice (rkky_nm, liechtenstein)")
        JD = _JD()
        lay = self.layout()
        H, anchors = self._host(lay)
        Deltas, orbs = [], []
        for spec, host_anchor in zip(self.specs, anchors):
            D, orb, _ = JD.extract_single_Delta(
                H_host=H, H_polarized_file=str(spec.file), host_anchor=host_anchor,
                basis=self.basis, tol=self.tol, region=self.region,
                radius=self.radius, atom_mask=self.atom_mask,
                **spec.delta_kwargs("rkky"))
            Deltas.append(sparse.csr_matrix(D))
            orbs.append(np.asarray(orb, dtype=int))
        HamL, HamR = self._electrodes()
        resolved = tuple(s.resolve() for s in self.specs)
        dist, adist = _distances(H.geometry, anchors, resolved)
        nl = HamL.no
        info = dict(layout=tuple(lay.values()), n_dist=self.n_dist, downfold=self.downfold)
        # Single-defect files on the device: supercell-end check and (optional) downfolding
        maps = [self._map_single(spec, H, a) for spec, a in zip(self.specs, anchors)]
        host_mats, S_host = _R0_blocks(H)
        cell_length = float(np.linalg.norm(self.H0.cell[self.axis]))
        elec_atoms = self._electrode_atoms(lay)
        ends = [_supercell_ends(m["H_def"].geometry, m["mats"], m["S_def"], m["shift"], m["amap"],
                                m["shift_xy"], host_mats, S_host, H.geometry, self.axis, cell_length,
                                electrode_atoms=elec_atoms)
                for m in maps]
        info["ends_max_splitting_meV"] = tuple(
            (round(1000 * e["left"]["spin"], 3), round(1000 * e["right"]["spin"], 3)) for e in ends)
        downfold_kwargs = {}
        if self.downfold:
            data = [self._downfold_data(spec, H, a, m)
                    for spec, a, m in zip(self.specs, anchors, maps)]
            downfold_kwargs = dict(downfold=True,
                                   hydrogen_i=data[0][0], hydrogen_j=data[1][0],
                                   mapped_c_orbs_i=data[0][1], mapped_c_orbs_j=data[1][1])
            info["downfold_energy_shifts_eV"] = tuple(round(d[2], 6) for d in data)
            info["downfold_align_rms_meV"] = tuple(round(1000 * d[3], 3) for d in data)
        dev = Device(self.kind, self.methods, H, HamL, HamR, *_semi(self.axis),
                     np.arange(nl), np.arange(H.no - nl, H.no), resolved,
                     tuple(anchors), Deltas, orbs, dist, adist,
                     info=info, nm_tol=self.nm_tol, downfold_kwargs=downfold_kwargs)
        tails = _vertex_tails({"rkky": (Deltas, orbs)}, dev.left_orbs, dev.right_orbs)
        entries = [(s_.label, e) for s_, e in zip(self.specs, ends)]
        if not self.verbose:          # warnings also without the report
            _report_defect_electrodes(entries, tails, self.ends_tol, self.overlap_tol, verbose=False)
        if self.verbose:
            self._report_host(lay, H, anchors, dist, adist, dev.left_orbs, dev.right_orbs)
            _report_defect_electrodes(entries, tails, self.ends_tol, self.overlap_tol)
            _report_vertices(self._region_text(), {"rkky": (Deltas, orbs)})
            if self.downfold:
                self._report_downfold([s.label for s in self.specs], data)
            _close()
            print(dev.summary(), flush=True)
        return dev


# ===========================================================================
# Superposition of single-defect perturbations
# ===========================================================================
class Superposition(PristineTiled):
    """Pristine host + scalar and spin perturbations of two single defects.

    Additional parameters (see PristineTiled for the others)
    ----------------------------------------------------------
    embed_radius : Angstrom around each anchor where V and Delta are embedded;
                   None -> the whole single-defect supercell.
    align        : 'far'  -> shift each defect Hamiltonian so that atoms far
                             from the defect match the host on-site energies;
                   'none' or a float (eV) -> manual shift.
    align_radius : atoms beyond this distance from the anchor are used by
                   align='far'; None -> farthest half of the mapped atoms.
    delta_scale  : multiplies every Delta (device and vertices); 1e-3 gives
                   the weak-coupling limit (LKAG -> RKKY check).
    region, radius, atom_mask : vertex region (must lie inside the embedding).
    vertices     : vertices of 'liechtenstein_sup':
                   'own'    (default) OWN splitting of each single-defect file:
                            each defect's spin cloud rotates rigidly with its
                            moment (cloud model);
                   'region' TOTAL splitting of the superposed device inside each
                            region (both defects), as in a two-defect DFT
                            device (region model).
                   'rkky_sup' always uses the OWN splitting (perturbative).
    ends_tol, overlap_tol : warning thresholds of the supercell-end and
                   electrode-region checks (see PristineTiled).
                   NOTE: this explicit builder solves with calculate_exchange,
                   which refuses overlapping Liechtenstein regions; overlapping
                   'own' vertices need SuperpositionDyson.

    Methods 'rkky_sup' and 'liechtenstein_sup'.
    """

    kind = "superposition"
    methods = ("rkky_sup", "liechtenstein_sup")

    def __init__(self, H0, defects, n_dist: int, basis: str = "pz",
                 embed_radius: Optional[float] = None, align="far",
                 align_radius: Optional[float] = None, delta_scale: float = 1.0,
                 align_warn: float = 0.01, vertices: str = "own", **kwargs):
        super().__init__(H0, defects, n_dist, basis=basis, **kwargs)
        self.embed_radius = embed_radius
        self.align, self.align_radius = align, align_radius
        self.delta_scale, self.align_warn = float(delta_scale), align_warn
        if vertices not in ("own", "region"):
            raise ValueError("vertices must be 'own' or 'region'")
        self.vertices = vertices

    def _support_atoms(self, spec, geometry, anchor):
        atoms = super()._support_atoms(spec, geometry, anchor)
        if self.embed_radius is None:
            return atoms
        d = np.linalg.norm(geometry.xyz[atoms] - geometry.xyz[anchor], axis=1)
        return atoms[d <= self.embed_radius]

    # -- one defect -----------------------------------------------------------
    def _prepare(self, spec, Hhost_geom, host_anchor):
        """Prune, map, align one defect; return a dict of local data."""
        JD = _JD()
        r = spec.resolve()
        H_full = sisl.get_sile(str(spec.file)).read_hamiltonian()
        if not H_full.spin.is_polarized:
            warnings.warn(f"{spec.label}: unpolarized defect Hamiltonian, Delta = 0")
        keep = spec.adatoms()
        H_def, orig = JD.prepare_spin_basis(H_full, basis=self.basis, keep_orbitals=keep)
        g = H_def.geometry
        anchor = pruned_index(orig, r.anchor)
        adatoms = [pruned_index(orig, a) for a in keep]
        defect_atoms = {pruned_index(orig, a) for a in r.defect_atoms}

        # Map non-adatom atoms by in-plane translation (anchor on host_anchor)
        hx = np.asarray(Hhost_geom.xyz)
        shift = hx[host_anchor, :2] - g.xyz[anchor, :2]
        tree = cKDTree(hx[:, :2])
        amap = np.full(g.na, -1, dtype=int)
        used = set()
        for a in range(g.na):
            if a in adatoms:
                continue
            if a == anchor:
                h = host_anchor
            else:
                dist, h = tree.query(g.xyz[a, :2] + shift)
                if dist > self.tol:
                    continue                      # outside the device
                h = int(h)
            if h in used:
                raise ValueError(f"{spec.label}: host atom {h} mapped twice")
            if (Hhost_geom.atoms.Z[h] != g.atoms.Z[a] and a not in defect_atoms):
                raise ValueError(f"{spec.label}: species mismatch at defect atom {a} "
                                 f"(Z={g.atoms.Z[a]}) / host atom {h} "
                                 f"(Z={Hhost_geom.atoms.Z[h]})")
            dn = [o.name() for o in g.atoms[a].orbitals]
            hn = [o.name() for o in Hhost_geom.atoms[h].orbitals]
            if dn != hn:
                raise ValueError(f"{spec.label}: orbital basis mismatch at defect atom "
                                 f"{a} / host atom {h}: {dn} vs {hn}")
            amap[a] = h
            used.add(h)
        mapped = np.flatnonzero(amap >= 0)
        mismatch = np.linalg.norm(g.xyz[mapped, :2] + shift - hx[amap[mapped], :2], axis=1)
        dz = float(np.median(hx[amap[mapped], 2] - g.xyz[mapped, 2]))
        ad_xyz = {a: g.xyz[a] + np.array([shift[0], shift[1], dz]) for a in adatoms}

        # Embedding and vertex regions (pruned defect atom indices)
        dist = np.linalg.norm(g.xyz - g.xyz[anchor], axis=1)
        if self.embed_radius is None:
            embed = set(mapped.tolist())
            lost = [a for a in range(g.na) if amap[a] < 0 and a not in adatoms]
            if lost:
                warnings.warn(f"{spec.label}: {len(lost)} atoms of the defect "
                              "supercell lie outside the device and are not embedded")
        else:
            inside = set(np.flatnonzero(dist <= self.embed_radius).tolist()) - set(adatoms)
            missing = sorted(a for a in inside if amap[a] < 0)
            if missing:
                raise ValueError(f"{spec.label}: atoms {missing} within embed_radius "
                                 "fall outside the device; enlarge the device")
            embed = inside
        embed |= set(adatoms)
        if self.region == "all":
            vertex = set(embed)
        elif self.region == "radius":
            vertex = set(np.flatnonzero(dist <= self.radius).tolist())
        elif self.region == "atom_mask":
            vertex = {pruned_index(orig, a) for a in np.atleast_1d(self.atom_mask)}
        else:
            raise ValueError("region must be 'all', 'radius' or 'atom_mask'")
        outside = sorted(vertex - embed)
        if outside:
            raise ValueError(f"{spec.label}: vertex atoms {outside} are not embedded; "
                             "increase embed_radius or reduce the vertex region")
        return dict(spec=spec, r=r, H=H_def, orig=orig, anchor=anchor, amap=amap,
                    adatoms=adatoms, ad_xyz=ad_xyz, embed=embed, vertex=vertex,
                    dist=dist, mapped=mapped, shift_xy=shift,
                    map_stats=dict(n_atoms=g.na, n_mapped=len(mapped),
                                   n_outside=int(np.sum(amap < 0)) - len(adatoms),
                                   max_mismatch=float(mismatch.max()) if len(mismatch) else 0.0))

    def _align_shift(self, d, H_host_mats, host_o2dev, def_o2dev, H_mats, S_def):
        """Energy shift of one defect Hamiltonian relative to the host (eV)."""
        d["align_stats"] = None
        if isinstance(self.align, (int, float, np.floating)):
            return float(self.align), np.nan
        if str(self.align).lower() == "none":
            return 0.0, np.nan
        g = d["H"].geometry
        heavy = np.array([a for a in d["mapped"] if g.atoms.Z[a] == 6], dtype=int)
        if not len(heavy):
            raise ValueError(f"{d['spec'].label}: no mapped carbon atoms for alignment")
        rad = (np.median(d["dist"][heavy]) if self.align_radius is None
               else self.align_radius)
        far = heavy[d["dist"][heavy] >= rad]
        if len(far) < 2:
            raise ValueError(f"{d['spec'].label}: too few atoms beyond align_radius")
        orbs = np.concatenate([g.a2o(a, all=True) for a in far])
        h_def = 0.5 * (H_mats[0].diagonal() + H_mats[1].diagonal())[orbs]
        s_def = (S_def.diagonal()[orbs] if S_def is not None else np.ones(len(orbs)))
        h_ref = H_host_mats.diagonal()[def_o2dev[orbs]]
        delta = h_ref - h_def
        shift = float(np.vdot(s_def, delta).real / np.vdot(s_def, s_def).real)
        rms = float(np.sqrt(np.mean((delta - shift * s_def) ** 2)))
        d["align_stats"] = dict(n_far=len(far), radius=float(rad),
                                max_residual=float(np.max(np.abs(delta - shift * s_def))))
        if rms > self.align_warn:
            warnings.warn(f"{d['spec'].label}: alignment residual {rms * 1000:.1f} meV; "
                          "far region of the defect supercell is not pristine-like")
        return shift, rms

    def _ends_of(self, d, H_mats, S_def, shift, host_mats, S_host, host_geom) -> dict:
        """Electrode-region check of one prepared single-defect Hamiltonian."""
        return _supercell_ends(d["H"].geometry, H_mats, S_def, shift, d["amap"], d["shift_xy"],
                               host_mats, S_host, host_geom, self.axis,
                               float(np.linalg.norm(self.H0.cell[self.axis])),
                               electrode_atoms=self._electrode_atoms(self.layout()))

    # -- reports ---------------------------------------------------------------
    def _report_superposition(self, data, shifts, rmss, max_deltas, embed_overlap,
                              adatom_mode, tails=None) -> None:
        """Alignment, mapping, embedding and spin polarization of the superposition."""
        _section("Single-defect energy alignment")
        if str(self.align).lower() != "far":
            print(f"Alignment:          {self.align} (no fit)")
        for k, (d, sh, rms) in enumerate(zip(data, shifts, rmss)):
            st = d.get("align_stats")
            print(f"Defect {'ij'[k]} ({d['r'].label}):".ljust(20) + f"shift {sh:+.6f} eV"
                  + (f", RMS residual {1000 * rms:.3f} meV, max {1000 * st['max_residual']:.3f} meV "
                     f"({st['n_far']} carbons beyond {st['radius']:.2f} Ang)" if st else ""))

        _section("Mapping onto the host")
        for k, d in enumerate(data):
            m = d["map_stats"]
            print(f"Defect {'ij'[k]} ({d['r'].label}):".ljust(20)
                  + f"{m['n_mapped']}/{m['n_atoms']} atoms mapped, {m['n_outside']} outside the device, "
                  f"max in-plane mismatch {m['max_mismatch']:.4f} Ang")
            if d["adatoms"]:
                print(f"{'':20s}adatoms {adatom_mode}: {len(d['adatoms'])}")

        _section("Embedding")
        er = "whole single-defect supercell" if self.embed_radius is None else f"{self.embed_radius:.2f} Ang"
        print(f"Embedding region:   {er}")
        for k, d in enumerate(data):
            print(f"Defect {'ij'[k]} ({d['r'].label}):".ljust(20) + f"{len(d['embed'])} atoms embedded")
        print(f"Overlap of the two embeddings: {embed_overlap} orbitals (perturbations added there)")

        _section("Spin polarization")
        print(f"Pristine host / electrodes, max |Hup-Hdn|: {1000 * self.h0_splitting:.3f} meV")
        for k, (d, mx) in enumerate(zip(data, max_deltas)):
            print(f"Defect {'ij'[k]} ({d['r'].label}):".ljust(20) + f"max |Hup-Hdn| {1000 * mx:.3f} meV "
                  f"in its single-defect file (delta_scale {self.delta_scale:g})")

        _report_defect_electrodes([(d["r"].label, d["ends"]) for d in data], tails or {},
                                  self.ends_tol, self.overlap_tol)

    # -- build ---------------------------------------------------------------
    def build(self, method: str = "liechtenstein_sup") -> Device:
        method = canonical_method(method)
        if method not in self.methods:
            raise ValueError(f"Superposition supports {self.methods}")
        lay = self.layout()
        Hhost, host_anchors = self._host(lay)
        hg = Hhost.geometry
        data = [self._prepare(s, hg, a) for s, a in zip(self.specs, host_anchors)]

        # Device atom order: host atoms, each adatom right after its host atom
        inserts: Dict[int, List[Tuple[int, int]]] = {}
        for k, d in enumerate(data):
            for a in d["adatoms"]:
                bonded = d["amap"][d["anchor"]]
                inserts.setdefault(int(bonded), []).append((k, a))
        atoms, xyz, host_a2dev = [], [], np.empty(hg.na, dtype=int)
        ad_a2dev: Dict[Tuple[int, int], int] = {}
        for ia in range(hg.na):
            host_a2dev[ia] = len(atoms)
            atoms.append(hg.atoms[ia]); xyz.append(hg.xyz[ia])
            for k, a in inserts.get(ia, []):
                ad_a2dev[(k, a)] = len(atoms)
                atoms.append(data[k]["H"].geometry.atoms[a]); xyz.append(data[k]["ad_xyz"][a])
        lattice = sisl.Lattice(hg.cell, nsc=[1, 1, 1])
        geom = sisl.Geometry(np.array(xyz), atoms=atoms, lattice=lattice)
        n = geom.no

        # Host matrices in device orbital indexing
        host_o2dev = np.concatenate([geom.firsto[host_a2dev[ia]] + np.arange(hg.atoms[ia].no)
                                     for ia in range(hg.na)])
        host_mats, S_host = _R0_blocks(Hhost)
        H0_dev = _reindex(host_mats[0], host_o2dev, n)
        S0_dev = None if S_host is None else _reindex(S_host, host_o2dev, n)
        avg_sum = sparse.csr_matrix((n, n))
        diff_sum = sparse.csr_matrix((n, n))
        dS_sum = sparse.csr_matrix((n, n))
        touched_all = np.zeros(n, dtype=int)
        Deltas, orbs, shifts, rmss, anchors = [], [], [], [], []
        Deltas_own = []

        for k, d in enumerate(data):
            Hd, gd = d["H"], d["H"].geometry
            H_mats, S_def = _R0_blocks(Hd)
            if (S_def is None) != (S0_dev is None):
                raise ValueError(f"{d['spec'].label}: host and defect overlap conventions differ")
            # Defect orbital -> device orbital (-1: not in the device)
            def_o2dev = np.full(Hd.no, -1, dtype=int)
            emb_orb = np.zeros(Hd.no, dtype=bool)
            for a in range(gd.na):
                if a in d["adatoms"]:
                    dev_atom = ad_a2dev[(k, a)]
                elif d["amap"][a] >= 0:
                    dev_atom = host_a2dev[d["amap"][a]]
                else:
                    continue
                o = gd.a2o(a, all=True)
                def_o2dev[o] = geom.firsto[dev_atom] + np.arange(len(o))
                emb_orb[o] = a in d["embed"]
            shift, rms = self._align_shift(d, H0_dev, host_o2dev, def_o2dev, H_mats, S_def)
            d["max_delta"] = self.delta_scale * _max_abs(H_mats[0] - H_mats[1])
            d["ends"] = self._ends_of(d, H_mats, S_def, shift, host_mats, S_host, hg)
            S_eff = S_def if S_def is not None else sparse.identity(Hd.no, format="csr")
            H_up = H_mats[0] + shift * S_eff
            H_dn = H_mats[1] + shift * S_eff

            def embed(M):
                """Defect matrix entries inside the embedding, in device indices."""
                C = M.tocoo()
                keep = ((def_o2dev[C.row] >= 0) & (def_o2dev[C.col] >= 0)
                        & (emb_orb[C.row] | emb_orb[C.col]))
                return sparse.coo_matrix((C.data[keep], (def_o2dev[C.row[keep]],
                                                         def_o2dev[C.col[keep]])),
                                         shape=(n, n)).tocsr()

            dev_emb = np.zeros(n, dtype=bool)
            dev_emb[def_o2dev[emb_orb & (def_o2dev >= 0)]] = True
            dev_img = np.zeros(n, dtype=bool)
            dev_img[def_o2dev[def_o2dev >= 0]] = True

            def host_part(M):
                """Host entries replaced by the defect (same positions)."""
                C = M.tocoo()
                keep = dev_img[C.row] & dev_img[C.col] & (dev_emb[C.row] | dev_emb[C.col])
                return sparse.coo_matrix((C.data[keep], (C.row[keep], C.col[keep])),
                                         shape=(n, n)).tocsr()

            up, dn = embed(H_up), embed(H_dn)
            avg_sum = avg_sum + 0.5 * (up + dn) - host_part(H0_dev)
            diff_sum = diff_sum + (up - dn)
            if S0_dev is not None:
                dS_sum = dS_sum + embed(S_def) - host_part(S0_dev)
            touched_all += dev_emb

            # Vertex orbitals of this defect, device indices
            vorb = np.concatenate([gd.a2o(a, all=True) for a in sorted(d["vertex"])])
            orbs.append(def_o2dev[vorb].astype(int))
            # rkky_nm: OWN splitting of this single-defect calculation
            D_own = (H_mats[0] - H_mats[1]).tocsr()[vorb][:, vorb].tocsr()
            D_own.eliminate_zeros()
            Deltas_own.append(self.delta_scale * D_own)
            shifts.append(shift); rmss.append(rms)
            anchors.append(int(host_a2dev[host_anchors[k]]))

        overlap = int(np.sum(touched_all > 1))
        if overlap:
            warnings.warn(f"Embedding regions overlap on {overlap} orbitals; the "
                          "perturbations are added there (superposition approximation)")

        # Electrodes and interfaces (host cells, device indexing)
        HamL, HamR = self._electrodes()
        na0, nt = self.H0.na, self.n_tile
        left_atoms = np.arange(nt * na0)
        right_atoms = np.arange((lay["N"] - nt) * na0, lay["N"] * na0)
        to_orbs = lambda at: np.concatenate([geom.firsto[host_a2dev[a]] + np.arange(hg.atoms[a].no)
                                             for a in at])
        left_orbs, right_orbs = to_orbs(left_atoms), to_orbs(right_atoms)
        if np.any(touched_all[np.union1d(left_orbs, right_orbs)]):
            raise ValueError("Embedded perturbation reaches the electrode cells; "
                             "increase N_buffer/N_imp or set embed_radius")

        Hs = 0.5 * self.delta_scale * diff_sum
        # liechtenstein vertices: TOTAL spin splitting of the superposed device inside
        # each vertex region (includes the tail of the other defect), like Delta of
        # a two-defect DFT device. rkky_nm vertices: OWN splitting of each
        # single-defect calculation (Deltas_own), the perturbative definition.
        D_total = (self.delta_scale * diff_sum).tocsr()
        for o in orbs:
            D_loc = D_total[o][:, o].tocsr()
            D_loc.eliminate_zeros()
            Deltas.append(D_loc)
        H_nm_mat = (H0_dev + avg_sum).tocsr()
        S_dev = None if S0_dev is None else (S0_dev + dS_sum).tocsr()
        H_sp = sisl.Hamiltonian.fromsp(geom, [(H_nm_mat + Hs).tocsr(), (H_nm_mat - Hs).tocsr()],
                                       S=S_dev)
        H_nm = sisl.Hamiltonian.fromsp(geom, H_nm_mat, S=S_dev)

        resolved = tuple(d["r"] for d in data)
        dist, adist = _distances(geom, anchors, resolved)
        # Deltas: total splitting in each region; Deltas_own: own single-defect splitting
        vertex_sets = {"rkky_sup": (Deltas_own, orbs),
                       "liechtenstein_sup": (Deltas_own if self.vertices == "own" else Deltas, orbs)}
        dev = Device(self.kind, self.methods, H_sp, HamL, HamR, *_semi(self.axis),
                     left_orbs, right_orbs, resolved, tuple(anchors),
                     *vertex_sets[method], dist, adist, H_nm=H_nm,
                     info=dict(layout=tuple(lay.values()), n_dist=self.n_dist,
                               energy_shifts_eV=tuple(round(s, 6) for s in shifts),
                               align_rms_meV=tuple(round(1000 * r, 3) for r in rmss),
                               embedding_overlap_orbitals=overlap,
                               delta_scale=self.delta_scale,
                               vertices=self.vertices),
                     nm_tol=self.nm_tol, vertex_sets=vertex_sets)
        tails = _vertex_tails({m: vertex_sets[m] for m in self.methods}, left_orbs, right_orbs)
        if not self.verbose:          # warnings also without the report
            _report_defect_electrodes([(d["r"].label, d["ends"]) for d in data], tails,
                                      self.ends_tol, self.overlap_tol, verbose=False)
        if self.verbose:
            self._report_host(lay, H_sp, anchors, dist, adist, left_orbs, right_orbs)
            self._report_superposition(
                data, shifts, rmss, [d["max_delta"] for d in data], overlap,
                "inserted (explicit H)", tails=tails)
            _report_vertices(self._region_text() + f", liechtenstein_sup vertices='{self.vertices}'",
                             {m: vertex_sets[m] for m in self.methods})
            _close()
            print(dev.summary(), flush=True)
        return dev


# ===========================================================================
# Single-defect vertices mapped onto a device
# ===========================================================================
def map_single_defect_vertex(spec, H_device, device_anchor: int, basis: str = "pz",
                             region: str = "all", radius: float = 7.5, atom_mask=None,
                             tol: float = 0.25, partner_atoms=(), exclude_orbs=(),
                             verbose: bool = True):
    """OWN Delta = H_up - H_down of one single-defect SP calculation, on a device.

    The single-defect Hamiltonian is pruned to `basis` keeping the adatom
    orbitals (spec.adatoms()), translated so that its anchor sits on
    `device_anchor`, and matched atom by atom and orbital by orbital:
    heavy atoms by in-plane position, adatoms by 3D position onto the explicit
    adatom of the device (no downfolding).

    region       : 'all' (every atom of the single-defect file that lies in the
                   device), 'radius' (sphere around the anchor) or 'atom_mask'
                   (atom indices of the FULL single-defect file).
    partner_atoms: device atoms of the OTHER defect; a species mismatch is
                   allowed only there (e.g. B of the partner on a carbon of
                   this file).
    exclude_orbs : device orbitals removed from the vertex (electrode
                   interfaces); the discarded |Delta| is reported.

    Returns Delta (compact csr), orb (device orbital indices), info (dict).
    """
    JD = _JD()
    r = spec.resolve()
    H_full = sisl.get_sile(str(spec.file)).read_hamiltonian()
    if not H_full.spin.is_polarized:
        raise ValueError(f"{spec.label}: single-defect file must be spin polarized")
    H_def, orig = JD.prepare_spin_basis(H_full, basis=basis, keep_orbitals=spec.adatoms())
    g = H_def.geometry
    anchor = pruned_index(orig, r.anchor)
    dist = np.linalg.norm(g.xyz - g.xyz[anchor], axis=1)
    if region == "all":
        wanted = np.arange(g.na)
    elif region == "radius":
        wanted = np.flatnonzero(dist <= radius)
    elif region == "atom_mask":
        wanted = np.array(sorted(pruned_index(orig, a) for a in np.atleast_1d(atom_mask)))
    else:
        raise ValueError("region must be 'all', 'radius' or 'atom_mask'")

    gd = H_device.geometry
    shift = np.asarray(gd.xyz[device_anchor]) - np.asarray(g.xyz[anchor])
    Zd = np.asarray(gd.atoms.Z)
    heavy_dev = np.flatnonzero(np.isin(Zd, (5, 6, 7, 8, 9)))
    light_dev = np.flatnonzero(~np.isin(Zd, (5, 6, 7, 8, 9)))
    tree_h = cKDTree(gd.xyz[heavy_dev, :2])
    tree_l = cKDTree(gd.xyz[light_dev]) if len(light_dev) else None
    partner = set(int(a) for a in partner_atoms)

    amap, unmapped, used = {}, [], {}
    for a in wanted:
        a = int(a)
        za = int(g.atoms.Z[a])
        if a == anchor:
            h = int(device_anchor)
        elif za in (5, 6, 7, 8, 9):
            d, k = tree_h.query(g.xyz[a, :2] + shift[:2])
            if d > tol:
                unmapped.append(a); continue
            h = int(heavy_dev[k])
        else:
            if tree_l is None:
                unmapped.append(a); continue
            d, k = tree_l.query(g.xyz[a] + shift)
            if d > tol:
                unmapped.append(a); continue
            h = int(light_dev[k])
        if h in used:
            raise ValueError(f"{spec.label}: device atom {h} matched by atoms "
                             f"{used[h]} and {a}; lower tol")
        if Zd[h] != za and h not in partner:
            raise ValueError(f"{spec.label}: species mismatch, single-defect atom {a} "
                             f"(Z={za}) on device atom {h} (Z={Zd[h]})")
        dn = [o.name() for o in g.atoms[a].orbitals]
        hn = [o.name() for o in gd.atoms[h].orbitals]
        if dn != hn:
            raise ValueError(f"{spec.label}: orbital mismatch at single-defect atom {a} / "
                             f"device atom {h}: {dn} vs {hn}")
        amap[a] = h
        used[h] = a
    adatoms = {pruned_index(orig, ia) for ia in spec.adatoms()}
    missing_ad = sorted(adatoms - set(amap))
    if missing_ad:
        raise ValueError(f"{spec.label}: adatom {missing_ad} has no partner in the device "
                         f"within {tol} Å (relaxation differences? increase tol)")
    if unmapped and region != "all":
        raise ValueError(f"{spec.label}: {len(unmapped)} vertex atoms are outside the "
                         f"device or not matched within {tol} Å: {unmapped[:10]}...")

    mats, _ = _R0_blocks(H_def)
    D = (mats[0] - mats[1]).tocsr()
    atoms = sorted(amap)
    d_orb = np.concatenate([g.a2o(a, all=True) for a in atoms])
    v_orb = np.concatenate([gd.a2o(amap[a], all=True) for a in atoms])
    excl = np.isin(v_orb, np.asarray(exclude_orbs, dtype=int))
    D_full = D[d_orb][:, d_orb]
    dropped_max = 0.0
    if np.any(excl):
        Dx = D_full.tocsr()
        dropped = Dx[np.flatnonzero(excl)]
        dropped_max = float(np.max(np.abs(dropped.data))) if dropped.nnz else 0.0
    keep = ~excl
    D_loc = D_full.tocsr()[np.flatnonzero(keep)][:, np.flatnonzero(keep)].tocsr()
    D_loc.eliminate_zeros()
    orb = v_orb[keep].astype(int)

    # Fraction of the single-defect splitting captured by the vertex
    norm_all = sparse.linalg.norm(D) if D.nnz else 1.0
    captured = float(sparse.linalg.norm(D_loc) / norm_all) if norm_all else 1.0
    info = dict(label=spec.label, region=region, n_atoms=len(atoms),
                n_unmapped=len(unmapped), n_interface_orbitals_dropped=int(excl.sum()),
                max_dropped_meV=1000 * dropped_max, captured_fraction=captured)
    if verbose:
        print(f"{spec.label}: single-defect vertex, region '{region}': {len(atoms)} atoms, "
              f"{len(orb)} orbitals, ||Delta|| captured {100 * captured:.2f}%"
              + (f", {len(unmapped)} atoms outside the device" if unmapped else "")
              + (f", {int(excl.sum())} interface orbitals dropped "
                 f"(max |Delta| {1000 * dropped_max:.3f} meV)" if excl.any() else ""),
              flush=True)
    if dropped_max > 1.0e-3:
        warnings.warn(f"{spec.label}: |Delta| up to {1000 * dropped_max:.2f} meV on the "
                      "electrode interfaces was dropped; the device is too short for "
                      "this single-defect vertex")
    return D_loc, orb, info


# ===========================================================================
# Two-defect DFT device
# ===========================================================================
class DFTDevice:
    """Two-defect DFT device, pruned and aligned to pristine electrodes.

    Propagators come from the two-defect DFT Hamiltonians:
      'liechtenstein': spin-polarized device `file` (e.g. the FM solution);
                       vertices = Delta of that device in each region.
      'rkky_nm'      : nonmagnetic device `nm_file`; vertices from
                       vertex_source = 'fm'     (default when `file` is given):
                                       the same FM vertices as 'liechtenstein';
                                     = 'single' the OWN Delta of the two
                                       single-defect SP calculations
                                       `single_defects`, mapped onto the device.
    Adatoms are explicit; there is no downfolding. FM vertices of both methods
    come from the same Hamiltonian, so their regions must be disjoint
    (checked). Single-defect vertices are not allowed for 'liechtenstein'.

    file           : spin-polarized two-defect file (None -> no liechtenstein).
    H0             : pruned, aligned pristine cell (electrodes).
    defects        : (spec_i, spec_j) in the device file, or None to detect them
                     (defect_type 'adatom'/'heteroatom', expect=2, i = left).
    nm_file        : nonmagnetic two-defect file (same structure and atom ordering
                     as `file` when both are given).
    single_defects : (spec_i, spec_j) of the single-defect SP files, in the same
                     order as the device defects (i = left); needed only for
                     vertex_source='single'.
    vertex_source  : 'fm' or 'single' (rkky_nm only); None -> 'fm' if `file`
                     is given, else 'single'.
    region, radius, atom_mask : FM vertex region (device file): 'radius' or
                     'atom_mask', disjoint for i and j.
    single_region, single_radius, single_atom_mask : vertex region for
                     vertex_source='single' (single-defect files); default
                     'all'; the two regions may overlap.
    map_tol        : Angstrom, matching tolerance single-defect -> device
                     (the two relaxations differ slightly).
    ends_tol, overlap_tol : warnings when, in the electrode regions of the FM / NM
                     device after the alignment, the spin splitting or |dH| exceeds
                     ends_tol (eV) or |dS| exceeds overlap_tol.
    """

    kind = "dft"

    def __init__(self, file=None, H0=None, basis: str = "pz", defects=None,
                 defect_type: str = "adatom", species="H", Z=5,
                 z_thresh: float = 0.3, group_cutoff: float = 3.0,
                 nm_file=None, single_defects=None, axis: int = 0,
                 n_tile: Optional[int] = None, tol: float = 0.1,
                 region: str = "radius", radius: float = 7.5, atom_mask=None,
                 single_region: str = "all", single_radius: Optional[float] = None,
                 single_atom_mask=None, map_tol: float = 0.25,
                 align_tol: float = 0.2, max_offset_difference: float = 0.01,
                 geom_tol: float = 0.1, nm_tol: float = 1e-3, verbose: bool = True,
                 vertex_source: Optional[str] = None, ends_tol: float = 1e-3,
                 overlap_tol: float = 1e-4):
        if H0 is None:
            raise ValueError("H0 (pristine electrode cell) is required")
        if file is None and nm_file is None:
            raise ValueError("Give the spin-polarized `file`, the nonmagnetic `nm_file`, or both")
        self.file = None if file is None else Path(file)
        self.nm_file = None if nm_file is None else Path(nm_file)
        device_file = self.file if self.file is not None else self.nm_file
        self.H0 = nonmagnetic(H0, nm_tol, "Pristine host H0")
        self.basis, self.axis = basis, int(axis)
        self.n_tile = electrode_tiling(self.H0, axis) if n_tile is None else int(n_tile)
        if defects is None:
            if defect_type == "adatom":
                defects = detect_adatoms(device_file, species=species, z_thresh=z_thresh,
                                         axis=axis, expect=2)
            elif defect_type == "heteroatom":
                defects = detect_heteroatoms(device_file, Z=Z, group_cutoff=group_cutoff,
                                             axis=axis, expect=2)
            else:
                raise ValueError("defect_type must be 'adatom' or 'heteroatom'")
        if len(defects) != 2:
            raise ValueError("Two defects are required")
        for s in defects:
            if Path(s.file).resolve() != Path(device_file).resolve():
                raise ValueError(f"{s.label}: spec file differs from the device file")
        self.specs = tuple(defects)
        if single_defects is not None:
            single_defects = tuple(single_defects)
            if len(single_defects) != 2:
                raise ValueError("single_defects must be (spec_i, spec_j)")
            for dev_s, one in zip(self.specs, single_defects):
                if dev_s.kind != one.kind:
                    raise ValueError(f"{one.label}: single-defect type '{one.kind}' differs "
                                     f"from the device defect '{dev_s.kind}'")
        self.single_defects = single_defects
        self.tol, self.region, self.radius, self.atom_mask = tol, region, radius, atom_mask
        self.single_region = single_region
        self.single_radius = radius if single_radius is None else single_radius
        self.single_atom_mask, self.map_tol = single_atom_mask, map_tol
        self.align_tol = align_tol
        self.max_offset_difference, self.geom_tol = max_offset_difference, geom_tol
        self.nm_tol, self.verbose = nm_tol, verbose
        if vertex_source is None:
            vertex_source = "fm" if self.file is not None else "single"
        if vertex_source not in ("fm", "single"):
            raise ValueError("vertex_source must be 'fm' or 'single'")
        if vertex_source == "fm" and self.file is None:
            raise ValueError("vertex_source='fm' needs the spin-polarized two-defect `file`")
        if vertex_source == "single" and single_defects is None:
            raise ValueError("vertex_source='single' needs single_defects=(spec_i, spec_j)")
        self.vertex_source = vertex_source
        self.ends_tol, self.overlap_tol = ends_tol, overlap_tol
        methods = []
        if self.file is not None:
            methods.append("liechtenstein")
        if self.nm_file is not None:
            methods.append("rkky_nm")
        if not methods:
            raise ValueError("rkky_nm needs nm_file; liechtenstein needs file")
        self.methods = tuple(methods)

    # -- helpers ----------------------------------------------------------------
    def _load(self, file, keep, HamL, label):
        JD = _JD()
        H_full = sisl.get_sile(str(file)).read_hamiltonian()
        H_dev, orig = JD.prepare_spin_basis(H_full, basis=self.basis, keep_orbitals=keep)
        # align_defect_to_electrodes calls Hk(spin=...) on the electrode, which
        # recent sisl rejects for unpolarized matrices: give it a polarized copy.
        elec = HamL if HamL.spin.is_polarized else HamL.transform(spin="polarized")
        H_al, shift, info = JD.align_defect_to_electrodes(
            H_dev, elec, axis=self.axis, tol=self.align_tol,
            max_offset_difference=self.max_offset_difference, verbose=self.verbose)
        _check_open_axes(H_al, self.axis, label=label)
        H_al.set_nsc([1, 1, 1])
        return H_full, H_al, orig, shift, info

    def single_vertices(self, device, region=None, radius=None, atom_mask=None,
                        verbose=None):
        """rkky_nm vertices from the single-defect files, on an already built device.

        Useful for convergence scans in the region without rebuilding the device.
        Returns (Deltas, orbs, infos).
        """
        if self.single_defects is None or device.H_nm is None:
            raise ValueError("single_defects and nm_file are required for rkky_nm vertices")
        region = self.single_region if region is None else region
        radius = self.single_radius if radius is None else radius
        atom_mask = self.single_atom_mask if atom_mask is None else atom_mask
        verbose = self.verbose if verbose is None else verbose
        orig = device.info["original_atoms"]
        defect_dev = [[pruned_index(orig, a) for a in r.defect_atoms] + [device.anchors[k]]
                      for k, r in enumerate(device.defects)]
        interfaces = np.union1d(device.left_orbs, device.right_orbs)
        Deltas, orbs, infos = [], [], []
        for k, one in enumerate(self.single_defects):
            D, o, inf = map_single_defect_vertex(
                one, device.H_nm, device.anchors[k], basis=self.basis, region=region,
                radius=radius, atom_mask=atom_mask, tol=self.map_tol,
                partner_atoms=defect_dev[1 - k], exclude_orbs=interfaces, verbose=verbose)
            Deltas.append(D); orbs.append(o); infos.append(inf)
        return Deltas, orbs, infos

    # -- build ------------------------------------------------------------------
    def build(self, method: str = "liechtenstein") -> Device:
        method = canonical_method(method)
        if method not in self.methods:
            hint = {"rkky_nm": "; rkky_nm needs nm_file",
                    "liechtenstein": "; liechtenstein needs the spin-polarized file"}
            raise ValueError(f"DFTDevice supports {self.methods}" + hint.get(method, ""))
        JD = _JD()
        elec = self.H0.tile(self.n_tile, self.axis)
        HamL, HamR = elec, elec.copy()
        keep = keep_orbitals_for(self.specs)
        resolved = tuple(s.resolve() for s in self.specs)

        H_sp = H_nm = None
        info = info_nm = shift = nm_shift = orig = orig_nm = None
        if self.file is not None:
            H_full, H_sp, orig, shift, info = self._load(self.file, keep, HamL, "SP device")
            if not H_full.spin.is_polarized:
                raise ValueError(f"{self.file} is not spin polarized; pass a nonmagnetic "
                                 "device as nm_file instead")
        if self.nm_file is not None:
            _, H_nm, orig_nm, nm_shift, info_nm = self._load(self.nm_file, keep, HamL,
                                                             "NM device")
            if H_sp is not None:
                if (H_nm.na != H_sp.na or not np.array_equal(orig_nm, orig)
                        or not np.array_equal(H_nm.atoms.Z, H_sp.atoms.Z)):
                    raise ValueError("nm_file atoms/ordering differ from the SP device")
                dmax = float(np.max(np.linalg.norm(H_nm.geometry.xyz - H_sp.geometry.xyz,
                                                   axis=1)))
                if dmax > self.geom_tol:
                    raise ValueError(f"nm_file geometry differs by {dmax:.3f} Ang "
                                     f"(> geom_tol {self.geom_tol})")
                if not (np.array_equal(info_nm["left_orbs"], info["left_orbs"])
                        and np.array_equal(info_nm["right_orbs"], info["right_orbs"])):
                    raise ValueError("Electrode interfaces of nm_file and SP device differ")
            H_nm = nonmagnetic(H_nm, self.nm_tol, "nm_file device")

        H_main = H_sp if H_sp is not None else H_nm
        orig = orig if orig is not None else orig_nm
        info = info if info is not None else info_nm
        anchors = [pruned_index(orig, r.anchor) for r in resolved]
        dist, adist = _distances(H_main.geometry, anchors, resolved)
        dev = Device(self.kind, self.methods, H_main, HamL, HamR, *_semi(self.axis),
                     np.asarray(info["left_orbs"], dtype=int),
                     np.asarray(info["right_orbs"], dtype=int),
                     resolved, tuple(anchors), [], [], dist, adist, H_nm=H_nm,
                     info=dict(energy_shift_eV=None if shift is None else round(float(shift), 6),
                               nm_energy_shift_eV=None if nm_shift is None
                               else round(float(nm_shift), 6),
                               original_atoms=orig),
                     nm_tol=self.nm_tol)

        if H_sp is not None:            # FM vertices (liechtenstein; rkky_nm by default)
            Deltas, orbs = [], []
            for spec, a in zip(self.specs, anchors):
                D, orb, _ = JD.extract_single_Delta(
                    H_host=H_sp, H_polarized_file=str(self.file), host_anchor=a,
                    basis=self.basis, tol=self.tol, region=self.region,
                    radius=self.radius, atom_mask=self.atom_mask,
                    **spec.delta_kwargs("liechtenstein", companions=self.specs))
                Deltas.append(sparse.csr_matrix(D))
                orbs.append(np.asarray(orb, dtype=int))
            overlap = np.intersect1d(orbs[0], orbs[1]).size
            message = (f"The FM vertex regions share {overlap} orbitals; both vertices come "
                       "from the same Hamiltonian, so they must be disjoint (region 'radius' "
                       f"below about {adist / 2:.2f} Ang, or 'atom_mask')")
            fm_needed = method == "liechtenstein" or (method == "rkky_nm"
                                                      and self.vertex_source == "fm")
            if overlap and fm_needed:
                raise ValueError(message)
            if overlap:
                # FM vertices not needed for this build: the device does not offer them
                dev.methods = tuple(m for m in dev.methods if m != "liechtenstein")
                dev.info["liechtenstein_unavailable"] = message
            else:
                if "liechtenstein" in self.methods:
                    dev.vertex_sets["liechtenstein"] = (Deltas, orbs)
                if "rkky_nm" in self.methods and self.vertex_source == "fm":
                    dev.vertex_sets["rkky_nm"] = (Deltas, orbs)
        if "rkky_nm" in self.methods and self.vertex_source == "single":
            Deltas, orbs, infos = self.single_vertices(dev)
            dev.vertex_sets["rkky_nm"] = (Deltas, orbs)
            dev.info["single_vertices"] = tuple(
                f"{i['label']}: {i['n_atoms']} atoms, captured {100 * i['captured_fraction']:.2f}%"
                for i in infos)
        dev.info["vertex_source_rkky_nm"] = self.vertex_source
        dev.Deltas, dev.orbs = dev.vertex_sets[method]
        electrode = {}
        if H_sp is not None:
            electrode["FM device"] = _electrode_regions(H_sp, HamL, HamR, dev.left_orbs, dev.right_orbs)
        if H_nm is not None:
            electrode["NM device"] = _electrode_regions(H_nm, HamL, HamR, dev.left_orbs, dev.right_orbs)
        tails = _vertex_tails(dev.vertex_sets, dev.left_orbs, dev.right_orbs)
        dev.info["electrode_regions_meV"] = {
            name: {side: (round(1000 * r[side]["spin"], 3), round(1000 * r[side]["dH"], 3))
                   for side in ("left", "right")} for name, r in electrode.items()}
        _report_electrode_regions(electrode, tails, self.ends_tol, self.overlap_tol,
                                  verbose=self.verbose)
        if self.verbose:
            region = self.region + (f" (R = {self.radius:.2f} Ang)" if self.region == "radius" else "")
            if "rkky_nm" in dev.vertex_sets and self.vertex_source == "single":
                region += (f"; rkky_nm from single-defect files, region {self.single_region}"
                           + (f" (R = {self.single_radius:.2f} Ang)" if self.single_region == "radius" else ""))
            elif "rkky_nm" in dev.vertex_sets:
                region += "; rkky_nm uses the FM vertices"
            _report_vertices(region, dev.vertex_sets)
            if "liechtenstein_unavailable" in dev.info:
                print("liechtenstein:      not offered (" + dev.info["liechtenstein_unavailable"][:60] + "...)")
            _close()
            print(dev.summary(), flush=True)
        return dev


# ===========================================================================
# Density of states for every builder
# ===========================================================================
def compute_dos(obj, method: str, energies, eta: float = 0.01, orbitals=None,
                return_orbital_dos: bool = False, lead_eta: float = 1e-5, nk=(1, 0, 0),
                rhs_batch: int = 16, include_leads: bool = True, verbose: bool = False):
    """DOS of the reference used by `method`, for any builder output.

    obj    : Device (PristineTiled, Superposition, DFTDevice) or DysonSystem
             (SuperpositionDyson).
    method : selects the reference Hamiltonian
             'rkky'                          pristine device
             'rkky_nm', 'rkky_sup'           nonmagnetic reference (one channel)
             'liechtenstein', 'liechtenstein_sup' (and 'lkag', 'lkag_sup')
                                             spin up and down
    Device     -> calculate_dos_embedding on the same A = zS - H - Sigma as the
                  exchange (Mulliken DOS, -Im diag(G S)/pi); orbitals=None gives
                  the total device DOS.
    DysonSystem -> DOS projected on the defect region R, plus the orbitals of
                  downfolded adatoms (see Dyson_Exchange.dos_dyson).
    Returns a dict with 'energies', 'spins', 'dos' (nspin, nE), 'orbital_dos', ...
    """
    if type(obj).__name__ == "DysonSystem":
        from Dyson_Exchange import dos_dyson
        return dos_dyson(obj, method, energies, eta=eta, lead_eta=lead_eta)
    JD = _JD()
    kw = obj.exchange_kwargs(method)
    return JD.calculate_dos_embedding(
        method=kw["method"], H_ref=kw["H_ref"], HamL=kw["HamL"], semi_L=kw["semi_L"],
        HamR=kw["HamR"], semi_R=kw["semi_R"], energies=energies, nk=nk,
        left_orbs=kw["left_orbs"], right_orbs=kw["right_orbs"],
        spin_independent_leads=True, eta=eta, lead_eta=lead_eta, orbitals=orbitals,
        rhs_batch=rhs_batch, return_orbital_dos=return_orbital_dos,
        include_leads=include_leads, verbose=verbose)
