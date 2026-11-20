# -*- coding: utf-8 -*-
"""LKAG on the periodic two-defect DFT cell (no electrode cut).

    run = run_exchange("liechtenstein", basis="pz+dz", n_dist=3, defect_j="H0",
                       periodic=True, time_reversal=True, inputs="defect_files.py")

The two-defect SIESTA run is itself periodic along the ribbon (DFT k grid). Instead of
cutting it and attaching pristine electrodes (DFTDevice), the kernel is evaluated on the
periodic cell with the DFT k grid:

    J = prefactor * Im sum_z w(z) sum_k w_k Tr[D_I(k) G_up_IJ(k,z) D_J(k) G_dn_JI(k,z)]

    G_s(k, z) = (z S(k) - H_s(k))^-1,   D(k) = H_up(k) - H_dn(k)  on the region orbitals.

This is the force-theorem energy of flipping region J against region I in EVERY cell, i.e.
the same FM -> AFM change as the periodic DFT runs (images included). On the n_dist = 3
H-adatom cells it agrees with the device LKAG within 2 % (pz+dz) and with an independent
eigenvector implementation to 1e-5.

Regions: 'disjoint' with the nearest-centre rule and the minimum image along the periodic
axis; weights 'delta' (||D_aa||, eV) or 'mulliken' (|m_a|, muB) as in DFTDevice.

time_reversal=True (collinear H(R), S(R) are real): G_s(-k) = G_s(k)^T and D(-k) = D(k)^T, so
    Tr[D_I G_up_IJ D_J G_dn_JI](-k) = Tr[(D_J G_up_JI)^T ... ]  -> computed exactly at +k
from the columns of G_s on I and J. Only k and -k pairs of the grid are folded; the result
equals the full grid to machine precision (NOT just the spin-averaged trace: the spins swap
between k and -k, which is why a plain k >= 0 sum with weight 2 is not exact). It halves the
number of factorizations; the number of right-hand sides is unchanged.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

import regions as RG
from defects_dependencies import (detect_adatoms, detect_heteroatoms, keep_orbitals_for,
                                  pruned_index)

__all__ = ["PeriodicDFT", "PeriodicDevice", "calculate_exchange_periodic", "kgrid_from_fdf",
           "monkhorst_pack_1d"]


# ---------------------------------------------------------------------------
# k grid
# ---------------------------------------------------------------------------
def kgrid_from_fdf(fdf, axis: int = 0) -> Tuple[int, float]:
    """(n_k, shift) along `axis` from the kgrid_Monkhorst_Pack block of the DFT run."""
    import sisl
    blk = sisl.get_sile(str(fdf)).get("kgrid_Monkhorst_Pack")
    if not blk:
        raise ValueError(f"{fdf}: no kgrid_Monkhorst_Pack block; give kgrid=n explicitly")
    rows = [[float(x) for x in line.split()[:4]] for line in blk]
    n = int(round(rows[axis][axis]))
    for i in range(3):
        for j in range(3):
            if i != j and (i == axis or j == axis) and abs(rows[i][j]) > 0:
                raise ValueError("only a diagonal k grid along the periodic axis is supported")
    return max(n, 1), float(rows[axis][3])


def monkhorst_pack_1d(n: int, shift: float = 0.0):
    """Monkhorst-Pack points (fractional, in (-1/2, 1/2]) and equal weights, as SIESTA."""
    k = (np.arange(n) + shift + 0.5 * (1 - n % 2)) / n
    k = (k + 0.5) % 1.0 - 0.5
    k[np.isclose(k, -0.5)] = 0.5
    return np.sort(k), np.full(n, 1.0 / n)


def _trs_pairs(k, tol=1e-9):
    """Group the grid into (k, -k) pairs. Returns list of (k, weight, has_partner)."""
    used = np.zeros(len(k), bool)
    out = []
    for i, ki in enumerate(k):
        if used[i]:
            continue
        used[i] = True
        mk = ((-ki + 0.5) % 1.0) - 0.5
        j = [j for j in range(len(k)) if not used[j]
             and abs(((k[j] - mk + 0.5) % 1.0) - 0.5) < tol]
        self_partner = abs(((ki - mk + 0.5) % 1.0) - 0.5) < tol      # k = 0 or 1/2
        if j:
            used[j[0]] = True
            out.append((ki, 2, True))
        elif self_partner:
            out.append((ki, 1, False))
        else:
            raise ValueError("the k grid is not symmetric under k -> -k; use time_reversal=False")
    return out


# ---------------------------------------------------------------------------
# Built object
# ---------------------------------------------------------------------------
@dataclass(eq=False)
class PeriodicDevice:
    """Periodic two-defect cell with its regions (plays the role of Device)."""
    H: "sisl.Hamiltonian"                       # reduced basis, nsc of the DFT run
    defects: tuple
    anchors: Tuple[int, int]
    orbs: List[np.ndarray]                      # region orbitals (I, J)
    Deltas: List[sparse.csr_matrix]             # R = 0 region blocks (plots / inspection)
    distance: float
    anchor_distance: float
    kgrid: Tuple[int, float]
    axis: int = 0
    vertex: str = "full"
    kind: str = "periodic"
    methods: Tuple[str, ...] = ("liechtenstein",)
    info: Dict = field(default_factory=dict)
    vertex_sets: Dict = field(default_factory=dict)

    @property
    def left_orbs(self):                        # no electrodes (plots)
        return np.zeros(0, dtype=int)

    right_orbs = left_orbs

    def vertices(self, method: str = "liechtenstein"):
        return self.Deltas, self.orbs

    def summary(self) -> str:
        n, s = self.kgrid
        lines = [f"periodic two-defect cell: {self.H.na} atoms, {self.H.no} orbitals, "
                 f"k grid {n} (shift {s}) along axis {self.axis}, vertex '{self.vertex}'"]
        for d, a, o in zip(self.defects, self.anchors, self.orbs):
            lines.append(f"  {d.label}: anchor atom {a}, {len(o)} vertex orbitals")
        lines.append(f"  distance: {self.anchor_distance:.3f} Ang (anchors), {self.distance:.3f} Ang (centers)")
        for k, v in self.info.items():
            if np.isscalar(v) or isinstance(v, (tuple, str)):
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)

    def write_geometry(self, filename) -> None:
        self.H.geometry.write(str(filename))


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
class PeriodicDFT:
    """Builder of PeriodicDevice from a spin-polarized two-defect DFT file (FM or AFM)."""

    kind = "periodic"
    methods = ("liechtenstein",)

    def __init__(self, file, basis: str = "pz", defect_type: str = "adatom", species="H", Z=5,
                 z_thresh: float = 0.3, group_cutoff: float = 3.0, axis: int = 0,
                 region: str = "disjoint", weight: str = "delta", w_min: float = 1e-3,
                 mulliken_column: int = 2, kgrid=None, vertex: str = "full",
                 verbose: bool = True, **unused):
        if region != "disjoint":
            raise ValueError("the periodic mode supports region='disjoint' only")
        if weight not in ("delta", "mulliken"):
            raise ValueError("weight must be 'delta' or 'mulliken'")
        self.file = Path(file)
        self.basis, self.axis = basis, int(axis)
        self.weight, self.w_min, self.mulliken_column = weight, float(w_min), int(mulliken_column)
        self.vertex = RG.check_vertex(vertex)
        self.verbose = verbose
        if defect_type == "adatom":
            specs = detect_adatoms(self.file, species=species, z_thresh=z_thresh, axis=axis, expect=2)
        elif defect_type == "heteroatom":
            specs = detect_heteroatoms(self.file, Z=Z, group_cutoff=group_cutoff, axis=axis, expect=2)
        else:
            raise ValueError("defect_type must be 'adatom' or 'heteroatom'")
        self.specs = tuple(specs)
        if kgrid is None:
            self.kgrid = kgrid_from_fdf(self.file, self.axis)
        elif np.isscalar(kgrid):
            self.kgrid = (int(kgrid), 0.0)
        else:
            self.kgrid = (int(kgrid[0]), float(kgrid[1]))

    def build(self, method: str = "liechtenstein") -> PeriodicDevice:
        import J_dependencies as JD
        if method not in ("liechtenstein", "lkag"):
            raise ValueError("the periodic mode is only implemented for liechtenstein")
        H_full = JD.read_hamiltonian(str(self.file))
        if not H_full.spin.is_polarized:
            raise ValueError(f"{self.file} is not spin polarized")
        keep = keep_orbitals_for(self.specs)
        Hb, orig = JD.prepare_spin_basis(H_full, basis=self.basis, keep_orbitals=keep,
                                         dz2_centres=[s.resolve().anchor for s in self.specs])
        g = Hb.geometry
        resolved = tuple(s.resolve() for s in self.specs)
        anchors = tuple(pruned_index(orig, r.anchor) for r in resolved)
        L = float(np.linalg.norm(g.cell[self.axis]))
        centres = np.array([g.xyz[a] + r.center_offset for a, r in zip(anchors, resolved)])
        wd = RG.atom_delta_weights(Hb)
        moments, src = RG.moments_or_none(self.file, na=H_full.na, column=self.mulliken_column,
                                          required=self.weight == "mulliken")
        m_dev = np.asarray(moments, dtype=float)[np.asarray(orig, dtype=int)]
        w = wd if self.weight == "delta" else np.abs(m_dev)
        _, kept = RG.partition(g.xyz, centres, w, assign="nearest", w_min=self.w_min,
                               periodic=(self.axis, L))
        orbs = [np.concatenate([g.a2o(a, all=True) for a in k]).astype(int) for k in kept]
        Hl = Hb.copy(); Hl.set_nsc([1, 1, 1])
        D0 = (Hl.Hk(spin=0, format="csr") - Hl.Hk(spin=1, format="csr")).tocsr()
        Deltas = []
        for o in orbs:
            D = sparse.csr_matrix(D0[o][:, o])
            if self.vertex == "onsite":
                D = RG.onsite_vertex(D, g.o2a(o))
            Deltas.append(D)
        dvec = g.xyz[anchors[1]] - g.xyz[anchors[0]]
        dvec[self.axis] = (dvec[self.axis] + L / 2) % L - L / 2
        cvec = centres[1] - centres[0]
        cvec[self.axis] = (cvec[self.axis] + L / 2) % L - L / 2
        rep = RG.region_moment_report({f"periodic cell ({src})": m_dev}, kept)
        dev = PeriodicDevice(Hb, resolved, anchors, orbs, Deltas, float(np.linalg.norm(cvec)),
                             float(np.linalg.norm(dvec)), self.kgrid, self.axis, self.vertex,
                             info=dict(file=str(self.file), original_atoms=orig,
                                       region_text=f"disjoint (nearest, minimum image; {self.weight} "
                                                   f">= {self.w_min:g})",
                                       region_moments_muB=rep, built_method="liechtenstein",
                                       vertex=self.vertex))
        dev.vertex_sets["liechtenstein"] = (Deltas, orbs)
        if self.verbose:
            print(dev.summary())
            print("Mulliken moment per region:\n" + RG.format_moment_report(rep), flush=True)
        self.built = dev
        return dev


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------
def calculate_exchange_periodic(dev: PeriodicDevice, energies, contour_weights,
                                prefactor: float = -1.0 / (2 * np.pi), time_reversal: bool = True,
                                orbital_resolved: bool = True, orbital_kernel: str = "classes",
                                comm=None, verbose: bool = False):
    """ExchangeResult of LKAG on the periodic cell (see the module docstring).

    The vertex is D(k) on the region orbitals ('full', the whole region block) or its
    atomic on-site blocks ('onsite', dev.vertex). (k, z) pairs are distributed over
    the MPI ranks of `comm`; rank 0 returns the result, the others None.
    """
    import J_dependencies as JD
    if orbital_kernel not in ("classes", "none"):
        raise ValueError("the periodic mode supports orbital_kernel='classes' or 'none'")
    H, ax = dev.H, dev.axis
    g = H.geometry
    oI, oJ = (np.asarray(o, dtype=int) for o in dev.orbs)
    nI, nJ = len(oI), len(oJ)
    cols = np.concatenate([oI, oJ])
    atoms_I, atoms_J = g.o2a(oI), g.o2a(oJ)
    n_k, shift = dev.kgrid
    kpts, kw_full = monkhorst_pack_1d(n_k, shift)
    if time_reversal:
        plan = _trs_pairs(kpts)
        klist = np.array([p[0] for p in plan]); kw = np.array([p[1] for p in plan]) / n_k
        partner = [p[2] for p in plan]
    else:
        klist, kw, partner = kpts, kw_full, [False] * n_k
    z = np.asarray(energies, dtype=complex)
    w = np.asarray(contour_weights, dtype=complex)
    rank = comm.Get_rank() if comm is not None else 0
    size = comm.Get_size() if comm is not None else 1
    tasks = [(ik, ie) for ik in range(len(klist)) for ie in range(len(z))][rank::size]
    K = np.zeros((len(klist), len(z)), dtype=complex)
    orb = np.zeros((nI, nJ), dtype=complex) if orbital_resolved else None
    if orbital_resolved and orbital_kernel == "classes":
        lab_i, names_i = JD.orbital_class_labels(g, oI)
        lab_j, names_j = JD.orbital_class_labels(g, oJ)
        cls = np.zeros((1, len(z), len(names_i), len(names_j)), dtype=complex)   # k-summed
    else:
        cls = None
    rhs = sparse.csc_matrix((np.ones(nI + nJ), (cols, np.arange(nI + nJ))), shape=(H.no, nI + nJ))
    R_all = rhs.toarray(); R_I, R_J = R_all[:, :nI], R_all[:, nI:]
    p = complex(prefactor)
    t0 = time.perf_counter()
    cur_k, mats = None, None
    for ik, ie in tasks:
        if ik != cur_k:                                   # H(k), S(k), D(k) once per k
            kv = np.zeros(3); kv[ax] = klist[ik]
            Hu = sparse.csc_matrix(H.Hk(k=kv, spin=0), dtype=complex)
            Hd = sparse.csc_matrix(H.Hk(k=kv, spin=1), dtype=complex)
            S = sparse.csc_matrix(H.Sk(k=kv), dtype=complex)
            D = (Hu - Hd).tocsr()
            DI, DJ = D[oI][:, oI], D[oJ][:, oJ]
            if dev.vertex == "onsite":
                DI, DJ = RG.onsite_vertex(DI, atoms_I), RG.onsite_vertex(DJ, atoms_J)
            mats, cur_k = (Hu, Hd, S, DI.tocsr(), DJ.tocsr()), ik
        Hu, Hd, S, DI, DJ = mats
        zz = z[ie]
        if time_reversal:
            Xu = splu((zz * S - Hu).tocsc()).solve(R_all)[cols]
            Xd = splu((zz * S - Hd).tocsc()).solve(R_all)[cols]
            Gu_IJ, Gu_JI = Xu[:nI, nI:], Xu[nI:, :nI]
            Gd_IJ, Gd_JI = Xd[:nI, nI:], Xd[nI:, :nI]
        else:
            Gu_IJ = splu((zz * S - Hu).tocsc()).solve(R_J)[oI]
            Gd_JI = splu((zz * S - Hd).tocsc()).solve(R_I)[oJ]
        x = np.asarray((DJ.T @ Gu_IJ.T).T)                 # G_up_IJ D_J
        y = np.asarray((DI.T @ Gd_JI.T).T)                 # G_dn_JI D_I
        contrib = x * y.T                                  # (nI, nJ); sum = Tr[D_I G_IJ D_J G_JI]
        if partner[ik]:                                    # the -k point, exactly
            xm = np.asarray((DJ @ Gu_JI).T)                # G_up_IJ(-k) D_J(-k)
            ym = np.asarray((DI @ Gd_IJ).T)                # G_dn_JI(-k) D_I(-k)
            contrib = 0.5 * (contrib + xm * ym.T)
        wc = p * w[ie]
        K[ik, ie] = wc * contrib.sum()                     # per k (k weight applied in J)
        if orb is not None:
            orb += kw[ik] * wc * contrib
        if cls is not None:
            cls[0, ie] += kw[ik] * wc * JD.class_sums(contrib, lab_i, lab_j, len(names_i), len(names_j))
        if verbose and verbose >= 2:
            print(f"k {ik + 1}/{len(klist)}  E {ie + 1}/{len(z)}  {time.perf_counter() - t0:.0f} s", flush=True)
    if comm is not None and size > 1:
        from mpi4py import MPI
        K = comm.reduce(K, op=MPI.SUM, root=0)
        orb = comm.reduce(orb, op=MPI.SUM, root=0) if orb is not None else None
        cls = comm.reduce(cls, op=MPI.SUM, root=0) if cls is not None else None
        if rank != 0:
            return None
    J_k = 1000.0 * np.imag(K.sum(axis=1))                 # J of each k point (time_reversal:
    kp3 = np.zeros((len(klist), 3)); kp3[:, ax] = klist   # average of k and -k); J = sum_k w_k J_k
    return JD.ExchangeResult(
        method="liechtenstein", J_meV=float(np.dot(kw, J_k)), J_k_meV=J_k,
        kpoints=kp3, k_weights=kw, energies=z, contour_weights=w, weighted_kernel_kE=K,
        J_orb_meV=None if orb is None else 1000.0 * np.imag(orb), prefactor=p,
        spin_average=False, orb_i=oI, orb_j=oJ, distance=dev.anchor_distance,
        weighted_class_kernel_kE=cls,
        class_names_i=None if cls is None else tuple(names_i),
        class_names_j=None if cls is None else tuple(names_j))
