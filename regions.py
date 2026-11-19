# -*- coding: utf-8 -*-
"""Disjoint magnetic regions (exchange vertices) for every builder.

Why disjoint
------------
The exchange kernel J = -1/2pi Im sum_z w Tr[D_i G_ij D_j G_ji] is the force-theorem
energy of rigid rotations of the local exchange fields. The force theorem needs
|m(a)| fixed on every orbital a, i.e. each orbital belongs to ONE rotating region.
Overlapping vertices (the former region='all') rotate the tail of cloud i together
with the core of cloud j: the band term then contains sum_a Delta_i(a) m_j(a), which is
cancelled by the double-counting energy that the band formula omits. Overlapping
regions are therefore rejected by every builder.

Rule
----
Every atom a is first ASSIGNED to one impurity k, then KEPT if it is significant:

    assignment 'dominant' : k = argmax_l w_l(a)   (own single-defect clouds; ties and
                            atoms outside every cloud -> nearest centre)
               'nearest'  : k = argmin_l |r_a - R_l|   (Voronoi cell of the centre R_l)
    kept if   w_k(a) >= w_min   and   |r_a - R_k| <= r_cut (r_cut None: no cutoff)

The cells never overlap. 'dominant' needs the own cloud of each defect and is the
default of the single-defect builders (PristineTiled, Superposition,
SuperpositionDyson); 'nearest' is the only rule of DFTDevice (only the total
splitting of the two-defect device is known). For identical defects 'dominant'
and 'nearest' coincide up to the asymmetry of the clouds.

Weights
-------
'delta'    : w(a) = ||(H_up - H_dn)_aa||_F, Frobenius norm of the R=0 on-site block of
             atom a in the basis of the vertex (eV); w_min in eV.
'mulliken' : w(a) = |m_a|, Mulliken spin moment (mu_B) of atom a; w_min in mu_B.

The Mulliken moment captured by each region is always reported (both weights): it
is read from siesta.out next to the fdf file (sisl stdoutSileSiesta, final
Mulliken charges) or, if no output is found, computed from the density matrix,
m_a = sum_{mu in a} [(D_up - D_dn) S]_{mu mu}.
"""
from __future__ import annotations

import glob
import os
import warnings
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

__all__ = ["REGION_MODES", "atom_delta_weights", "mulliken_moments", "moments_or_none", "partition",
           "check_protected", "region_moment_report", "format_moment_report"]

REGION_MODES = ("disjoint", "radius", "square", "atom_mask")
REMOVED_ALL = ("region='all' was removed: overlapping vertices double count the "
               "exchange (sum_a Delta_i(a) m_j(a) is cancelled by the double-counting "
               "energy). Use region='disjoint' (default), or 'radius'/'square'/'atom_mask' "
               "with non-overlapping regions.")


def check_region(region: str) -> str:
    region = str(region).lower()
    if region == "all":
        raise ValueError(REMOVED_ALL)
    if region not in REGION_MODES:
        raise ValueError(f"region must be one of {REGION_MODES}")
    return region


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------
def atom_delta_weights(H) -> np.ndarray:
    """||(H_up - H_dn)_aa||_F (eV) per atom of a spin-polarized sisl Hamiltonian (R=0)."""
    if not H.spin.is_polarized:
        raise ValueError("a collinear spin-polarized Hamiltonian is required")
    Hl = H.copy()
    Hl.set_nsc([1, 1, 1])
    D = (Hl.Hk(spin=0, format="csr") - Hl.Hk(spin=1, format="csr")).tocsr()
    g = H.geometry
    w = np.zeros(g.na)
    for a in range(g.na):
        o = g.a2o(a, all=True)
        blk = D[o][:, o]
        w[a] = float(np.sqrt(np.sum(np.abs(blk.data) ** 2))) if blk.nnz else 0.0
    return w


def _moments_from_dm(fdf) -> np.ndarray:
    import sisl
    DM = sisl.get_sile(str(fdf)).read_density_matrix(order=["DM"])
    m = np.asarray(DM.mulliken("atom"))
    if m.ndim != 2 or m.shape[0] < 2:
        raise ValueError("density matrix is not spin polarized")
    return np.asarray(m[1], dtype=float)


def _find_output(fdf) -> Optional[Path]:
    d = Path(fdf).expanduser().resolve().parent
    for name in ("siesta.out",):
        if (d / name).is_file():
            return d / name
    outs = sorted(glob.glob(str(d / "*.out")))
    return Path(outs[0]) if outs else None


def mulliken_moments(fdf, na: Optional[int] = None, out_file=None, column: int = 2,
                     verbose: bool = False) -> tuple:
    """Mulliken spin moment per atom of the calculation behind `fdf`.

    Returns (moments, source). siesta.out (or *.out) next to the fdf is read with
    stdoutSileSiesta.read_charge('mulliken'); the final step is taken and `column`
    is the spin column (2 for spin-polarized SIESTA >= 5.2 output, as read by
    ``read_charge('mulliken')[0][:, 2]``). The result is cross-checked with the
    density matrix when both exist (total moment within 0.05 mu_B). Without an
    output file, or if it cannot be parsed, the DM is used.
    """
    import sisl
    out = Path(out_file) if out_file is not None else _find_output(fdf)
    from_out = None
    if out is not None and out.is_file():
        try:
            q = sisl.io.siesta.stdoutSileSiesta(str(out)).read_charge("mulliken")
            arr = q[-1] if isinstance(q, (list, tuple)) else q
            arr = np.asarray(arr, dtype=float)
            if arr.ndim == 3:            # (steps, na, ncol)
                arr = arr[-1]
            m = arr[:, column]
            if na is not None and len(m) != na:
                raise ValueError(f"{len(m)} atoms in {out.name}, expected {na}")
            from_out = m
        except Exception as exc:          # pragma: no cover - depends on the output
            warnings.warn(f"Mulliken charges of {out} not usable ({exc}); using the DM")
    m_dm = None
    try:
        m_dm = _moments_from_dm(fdf)
    except Exception as exc:
        if from_out is None:
            raise ValueError(f"No Mulliken moments for {fdf}: no usable output file and "
                             f"density matrix failed ({exc})")
    if from_out is not None:
        if m_dm is not None and abs(from_out.sum() - m_dm.sum()) > 0.05:
            warnings.warn(f"{out.name}: total Mulliken moment {from_out.sum():.3f} differs "
                          f"from the DM ({m_dm.sum():.3f}); check `column`")
        return from_out, str(out)
    return m_dm, "density matrix"


def moments_or_none(fdf, na: Optional[int] = None, column: int = 2, required: bool = False):
    """mulliken_moments, or (zeros, 'unavailable') with a warning when there is neither an
    output file nor a density matrix (only allowed when the moments are not the weight)."""
    try:
        return mulliken_moments(fdf, na=na, column=column)
    except Exception as exc:
        if required:
            raise
        warnings.warn(f"No Mulliken moments for {fdf} ({exc}); the region report is skipped")
        if na is None:
            import sisl
            na = sisl.get_sile(str(fdf)).read_geometry().na
        return np.zeros(na), "unavailable"


# ---------------------------------------------------------------------------
# Partition
# ---------------------------------------------------------------------------
def partition(xyz, centers, weights, assign: str = "dominant", w_min: float = 1e-4,
              r_cut: Optional[float] = None, protected: Optional[Sequence] = None,
              names: Optional[Sequence[str]] = None):
    """Disjoint regions on a set of atoms.

    xyz      : (na, 3) atom positions (device frame)
    centers  : (K, 3) impurity centres
    weights  : (K, na) own weight of every impurity on every atom ('dominant', or
               'nearest' with own clouds) or (na,) one total weight ('nearest')
    protected: per impurity, atoms that must belong to it (e.g. carbons coupled to a
               downfolded adatom); they are kept whatever their weight, and an error
               is raised if they are assigned elsewhere or lie beyond r_cut.
    Returns (labels, kept): labels[a] = impurity of atom a (assignment), kept[k] =
    sorted atom indices of region k.
    """
    xyz = np.asarray(xyz, dtype=float)
    C = np.atleast_2d(np.asarray(centers, dtype=float))
    K, na = len(C), len(xyz)
    W = np.asarray(weights, dtype=float)
    if W.ndim == 1:
        if assign == "dominant":
            raise ValueError("assign='dominant' needs one weight array per impurity")
        W = np.broadcast_to(W, (K, na))
    if W.shape != (K, na):
        raise ValueError(f"weights must have shape ({K}, {na})")
    dist = np.linalg.norm(xyz[None, :, :] - C[:, None, :], axis=2)   # (K, na)
    nearest = np.argmin(dist, axis=0)
    if assign == "nearest":
        labels = nearest
    elif assign == "dominant":
        labels = np.argmax(W, axis=0)
        top = np.sort(W, axis=0)
        tie = (top[-1] <= 0) | (np.abs(top[-1] - top[-2]) <= 1e-12 * np.maximum(top[-1], 1e-300)) \
            if K > 1 else (top[-1] <= 0)
        labels = np.where(tie, nearest, labels)
    else:
        raise ValueError("assign must be 'dominant' or 'nearest'")
    rows = np.arange(na)
    keep = W[labels, rows] >= w_min
    if r_cut is not None:
        keep &= dist[labels, rows] <= r_cut
    kept = []
    names = [f"impurity {k}" for k in range(K)] if names is None else list(names)
    prot = [np.asarray(p, dtype=int) for p in (protected or [()] * K)]
    for k in range(K):
        sel = set(np.flatnonzero(keep & (labels == k)).tolist())
        p = prot[k]
        if p.size:
            wrong = p[labels[p] != k]
            if wrong.size:
                raise ValueError(
                    f"{names[k]} ({'ij'[k] if K == 2 else k}): {wrong.size} atoms coupled to its downfolded adatom are "
                    f"assigned to another impurity (closest at "
                    f"{dist[k, wrong].min():.2f} Ang). The impurities are closer than the "
                    "downfolding range allows: increase the distance or use downfold=False.")
            if r_cut is not None and np.any(dist[k, p] > r_cut):
                raise ValueError(f"{names[k]}: atoms coupled to its downfolded adatom lie "
                                 f"beyond r_cut = {r_cut} Ang (up to "
                                 f"{dist[k, p].max():.2f} Ang); increase r_cut")
            sel |= set(p.tolist())
        kept.append(np.array(sorted(sel), dtype=int))
    for k in range(K):
        for l in range(k + 1, K):
            if np.intersect1d(kept[k], kept[l]).size:          # cannot happen; guard
                raise RuntimeError("partition produced overlapping regions")
    return labels, kept


def check_protected(region_atoms, protected_atoms, label: str, other_atoms=()) -> None:
    """Raise if atoms coupled to a downfolded adatom are outside its region (any mode)."""
    p = set(np.asarray(protected_atoms, dtype=int).tolist())
    missing = sorted(p - set(np.asarray(region_atoms, dtype=int).tolist()))
    if missing:
        raise ValueError(f"{label}: {len(missing)} atoms coupled to the downfolded adatom are "
                         "outside its vertex region; enlarge the region (or downfold=False)")
    clash = sorted(p & set(np.asarray(other_atoms, dtype=int).tolist()))
    if clash:
        raise ValueError(f"{label}: atoms coupled to the downfolded adatom belong to the other "
                         "impurity's region; the impurities are too close for downfolding")


def check_disjoint(orbs_i, orbs_j, what: str = "vertex") -> None:
    n = np.intersect1d(np.asarray(orbs_i), np.asarray(orbs_j)).size
    if n:
        raise ValueError(f"The two {what} regions share {n} orbitals. Overlapping regions "
                         "double count the exchange and are not allowed; use "
                         "region='disjoint' or smaller regions.")


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
def region_moment_report(moments: Dict[str, np.ndarray], regions) -> dict:
    """Mulliken moment of every source inside every region.

    moments : {source label: per-atom moments in the SAME atom indexing as regions}
    regions : list of atom-index arrays (one per impurity)
    """
    rep = {}
    for name, m in moments.items():
        m = np.asarray(m, dtype=float)
        tot = float(m.sum())
        rep[name] = dict(total=tot, regions=[float(m[np.asarray(r, dtype=int)].sum())
                                             for r in regions])
    return rep


def format_moment_report(rep: dict, labels=("i", "j")) -> str:
    lines = []
    for name, r in rep.items():
        parts = ", ".join(f"region {labels[k]} {v:+.3f}" for k, v in enumerate(r["regions"]))
        lines.append(f"  {name}: total {r['total']:+.3f} muB; {parts} muB")
    return "\n".join(lines)
