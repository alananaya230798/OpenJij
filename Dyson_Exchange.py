# -*- coding: utf-8 -*-
"""Exchange from two single-defect calculations with Dyson's equation.

The pristine Green's function G0 of the tiled host (with exact electrodes) is
computed once per contour energy on the defect region R only. The defects
enter through a perturbation U(z) on R:

    G = G0 + G0 U G   ->   G_RR = (1 - G0_RR U)^(-1) G0_RR        (exact)

    U_s(z) = dH_s - z dS + sum_adatoms Sigma_H,s(z)

dH, dS are the superposed single-defect perturbations (as in Superposition:
aligned single-defect Hamiltonian minus pristine, embedded around each
anchor), Sigma_H the self-energy of a downfolded adatom,

    Sigma_H(z) = (H_CH - z S_CH) (z S_HH - H_HH)^(-1) (H_HC - z S_HC).

Methods (all from the SAME two single-defect calculations)
----------------------------------------------------------
'rkky_sup'          G = G_V, V = V_1 + V_2 (spin-averaged single-defect potentials)
                    vertices: own Delta of each single-defect file
'liechtenstein_sup' G_s = G_V,s, V_s = sum_d (V_d + s Delta_d / 2)
  (alias 'lkag_sup')
                    vertices='own' (default): own Delta of each single-defect
                        file. Each defect's spin cloud rotates rigidly with its
                        moment (cloud model); exact second derivative of the
                        superposed state, also when the clouds overlap, so any
                        region ('all', 'radius', 'atom_mask') is allowed.
                    vertices='region': total Delta of the superposed state in
                        each region (region model, as for a DFT two-defect
                        device); the regions must be disjoint.
In the weak-coupling limit liechtenstein_sup with vertices='own' tends to
rkky_sup at every distance.

The pristine-G0 method 'rkky' belongs to PristineTiled; the Dyson engine keeps
it only as an internal cross-check (calculate_exchange_dyson(...,
internal_checks=True)).

Adatoms (downfold)
------------------
downfold=False : the adatom is removed, as in plain rkky (its hybridization
                 with the carbons is lost; its effect on the carbon block of
                 the single-defect Hamiltonian is kept).
downfold=True  : the adatom is folded into the carbons it couples to, exactly
                 (Loewdin partitioning). Then (explicit H, Superposition)
                   rkky_sup          == Superposition 'rkky_sup'
                   liechtenstein_sup == Superposition 'liechtenstein_sup'
                 to machine precision. Scattering uses Sigma_H[H_avg] for
                 rkky_sup and Sigma_H,s for liechtenstein_sup. Vertex
                 correction: Sigma_up - Sigma_dn for liechtenstein_sup (exact);
                 for rkky_sup the term linear in Delta (vertex_downfold=
                 'linear', exact counterpart of the explicit-H kernel) or
                 Sigma_up - Sigma_dn (vertex_downfold='full').

DOS
---
dos_dyson(system, method, energies) gives the Mulliken DOS of the defect
region (and of the folded adatoms) for the reference of `method`.

Usage
-----
    system = SuperpositionDyson(H_0, (spec_i, spec_j), n_dist, basis="pz",
                                downfold=True, region="radius", radius=7.5).build()
    result = calculate_exchange_dyson(system, "rkky_sup", energies=En,
                                      contour_weights=weights)
    result = calculate_exchange_dyson(system, "liechtenstein_sup", energies=En,
                                      contour_weights=weights, vertices="own")
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from Device_Builder import (Superposition, _R0_blocks, _distances, _semi, _section, _close,
                            _max_abs, _report_vertices, _vertex_tails, _report_defect_electrodes)
from defects_dependencies import pruned_index

__all__ = ["SuperpositionDyson", "DysonSystem", "calculate_exchange_dyson",
           "dos_dyson", "DYSON_METHODS"]

DYSON_METHODS = ("rkky_sup", "liechtenstein_sup")
_ALIASES = {"lkag_sup": "liechtenstein_sup"}
_CHECK_METHODS = ("rkky",)          # pristine G0: PristineTiled's method, kept for tests


def _method(name: str, internal_checks: bool = False) -> str:
    m = name.lower().strip()
    m = _ALIASES.get(m, m)
    allowed = DYSON_METHODS + (_CHECK_METHODS if internal_checks else ())
    if m not in allowed:
        hint = " ('rkky' is computed with PristineTiled)" if m == "rkky" else ""
        raise ValueError(f"Unknown Dyson method '{name}'. Use one of {DYSON_METHODS}{hint}")
    return m


# ===========================================================================
# Downfolded adatom
# ===========================================================================
@dataclass(eq=False)
class FoldedAdatom:
    """Blocks of one adatom (aligned, delta-scaled) and its coupled carbons."""
    owner: int                      # defect index (0 = i, 1 = j)
    c_loc: np.ndarray               # coupled carbon orbitals, local indices in R
    H_CH: List[np.ndarray]          # per spin
    H_HC: List[np.ndarray]
    H_HH: List[np.ndarray]
    S_CH: np.ndarray
    S_HC: np.ndarray
    S_HH: np.ndarray

    def _blocks(self, z, spin):
        if spin == "avg":
            H_CH = 0.5 * (self.H_CH[0] + self.H_CH[1])
            H_HC = 0.5 * (self.H_HC[0] + self.H_HC[1])
            H_HH = 0.5 * (self.H_HH[0] + self.H_HH[1])
        else:
            H_CH, H_HC, H_HH = self.H_CH[spin], self.H_HC[spin], self.H_HH[spin]
        P = H_CH - z * self.S_CH
        Q = z * self.S_HH - H_HH
        R = H_HC - z * self.S_HC
        return P, Q, R

    def sigma(self, z, spin):
        """Sigma_H(z) for spin 0/1, or of the spin-averaged blocks ('avg')."""
        P, Q, R = self._blocks(z, spin)
        return P @ np.linalg.solve(Q, R)

    def vertex_full(self, z):
        """Sigma_up - Sigma_dn (exact LKAG vertex correction)."""
        return self.sigma(z, 0) - self.sigma(z, 1)

    def vertex_linear(self, z):
        """Term of Sigma_up - Sigma_dn linear in Delta, around the averaged blocks."""
        P, Q, R = self._blocks(z, "avg")
        dCH = self.H_CH[0] - self.H_CH[1]
        dHC = self.H_HC[0] - self.H_HC[1]
        dHH = self.H_HH[0] - self.H_HH[1]
        QiR = np.linalg.solve(Q, R)
        PQi = np.linalg.solve(Q.T, P.T).T
        return dCH @ QiR + PQi @ dHC + PQi @ dHH @ QiR


# ===========================================================================
# System
# ===========================================================================
@dataclass(eq=False)
class DysonSystem:
    """Pristine device, electrodes and the superposed perturbation on R."""
    H: "object"                          # pristine tiled device (sisl, nonmagnetic)
    HamL: "object"
    HamR: "object"
    semi_L: str
    semi_R: str
    left_orbs: np.ndarray
    right_orbs: np.ndarray
    R: np.ndarray                        # device orbitals of the defect region
    dH_avg: np.ndarray                   # (nR, nR) scalar perturbation V_1 + V_2
    dH_diff: np.ndarray                  # (nR, nR) Delta_1 + Delta_2 (delta-scaled)
    dS: Optional[np.ndarray]             # (nR, nR) overlap change or None
    adatoms: List[FoldedAdatom]
    loc: List[np.ndarray]                # vertex orbitals of i and j (local in R)
    vertex_own: List[np.ndarray]         # static own Delta on loc[k]
    vertex_total: List[np.ndarray]       # static total Delta on loc[k]
    defects: tuple
    anchors: tuple
    distance: float
    anchor_distance: float
    downfold: bool
    info: Dict = field(default_factory=dict)

    methods = DYSON_METHODS

    @property
    def orbs(self):
        """Vertex orbitals of i and j in device indices."""
        return [self.R[l] for l in self.loc]

    def summary(self) -> str:
        lines = [f"Dyson superposition system: device {self.H.na} atoms / {self.H.no} orbitals, "
                 f"defect region R = {len(self.R)} orbitals, downfold={self.downfold}",
                 f"  methods: {', '.join(self.methods)}",
                 f"  folded adatoms: {len(self.adatoms)}"]
        for d, a, l, Do, Dt in zip(self.defects, self.anchors, self.loc,
                                   self.vertex_own, self.vertex_total):
            lines.append(f"  {d.label}: anchor atom {a}, {len(l)} vertex orbitals, "
                         f"max|Delta| own {np.max(np.abs(Do)):.4f} / total {np.max(np.abs(Dt)):.4f} eV")
        lines.append(f"  distance: {self.anchor_distance:.3f} Ang (anchors), "
                     f"{self.distance:.3f} Ang (centers)")
        for k, v in self.info.items():
            if np.isscalar(v) or isinstance(v, (tuple, str)):
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)


# ===========================================================================
# Builder
# ===========================================================================
class SuperpositionDyson(Superposition):
    """Superposition of two single-defect calculations, for Dyson-equation methods.

    Same inputs as Superposition (layout, alignment, embed_radius, vertex region,
    delta_scale), plus

    downfold : True  -> adatoms folded into their carbons (exact),
               False -> adatoms removed (as plain rkky).
    vertices : default vertex choice of 'liechtenstein_sup' ('own' or
               'region'); can be overridden in calculate_exchange_dyson.

    Default region 'all' (inherited): each vertex is the whole single-defect
    spin cloud, the natural choice of the cloud model (vertices='own').

    build() returns a DysonSystem for calculate_exchange_dyson and dos_dyson;
    no method is chosen at build time.
    """

    kind = "superposition_dyson"
    methods = DYSON_METHODS

    def __init__(self, H0, defects, n_dist: int, basis: str = "pz",
                 downfold: bool = False, **kwargs):
        super().__init__(H0, defects, n_dist, basis=basis, **kwargs)
        self.downfold = bool(downfold)

    def build(self, method: Optional[str] = None) -> DysonSystem:
        lay = self.layout()
        Hhost, host_anchors = self._host(lay)
        hg = Hhost.geometry
        n = Hhost.no
        host_mats, S_host = _R0_blocks(Hhost)
        H0m = host_mats[0]
        lam = self.delta_scale

        dH_avg = sparse.csr_matrix((n, n))
        dH_diff = sparse.csr_matrix((n, n))
        dS = sparse.csr_matrix((n, n))
        support = np.zeros(n, dtype=int)
        per_defect, folded, shifts, rmss = [], [], [], []

        for k, (spec, h_anchor) in enumerate(zip(self.specs, host_anchors)):
            d = self._prepare(spec, hg, h_anchor)
            Hd, gd = d["H"], d["H"].geometry
            H_mats, S_def = _R0_blocks(Hd)
            if (S_def is None) != (S_host is None):
                raise ValueError(f"{spec.label}: host and defect overlap conventions differ")
            # defect orbital -> device orbital (heavy mapped atoms only)
            o2dev = np.full(Hd.no, -1, dtype=int)
            emb = np.zeros(Hd.no, dtype=bool)
            for a in range(gd.na):
                if a in d["adatoms"] or d["amap"][a] < 0:
                    continue
                o = gd.a2o(a, all=True)
                o2dev[o] = hg.firsto[d["amap"][a]] + np.arange(len(o))
                emb[o] = a in d["embed"]
            shift, rms = self._align_shift(d, H0m, None, o2dev, H_mats, S_def)
            d["max_delta"] = lam * _max_abs(H_mats[0] - H_mats[1])
            d["ends"] = self._ends_of(d, H_mats, S_def, shift, host_mats, S_host, hg)
            S_eff = S_def if S_def is not None else sparse.identity(Hd.no, format="csr")
            avg = 0.5 * (H_mats[0] + H_mats[1]) + shift * S_eff
            diff = lam * (H_mats[0] - H_mats[1])
            H_up, H_dn = (avg + 0.5 * diff).tocsr(), (avg - 0.5 * diff).tocsr()

            def embed(M):
                C = M.tocoo()
                keep = ((o2dev[C.row] >= 0) & (o2dev[C.col] >= 0)
                        & (emb[C.row] | emb[C.col]))
                return sparse.coo_matrix((C.data[keep], (o2dev[C.row[keep]],
                                                         o2dev[C.col[keep]])),
                                         shape=(n, n)).tocsr()
            dev_emb = np.zeros(n, dtype=bool); dev_emb[o2dev[emb & (o2dev >= 0)]] = True
            dev_img = np.zeros(n, dtype=bool); dev_img[o2dev[o2dev >= 0]] = True

            def host_part(M):
                C = M.tocoo()
                keep = dev_img[C.row] & dev_img[C.col] & (dev_emb[C.row] | dev_emb[C.col])
                return sparse.coo_matrix((C.data[keep], (C.row[keep], C.col[keep])),
                                         shape=(n, n)).tocsr()

            up, dn = embed(H_up), embed(H_dn)
            dH_avg = dH_avg + 0.5 * (up + dn) - host_part(H0m)
            dH_diff = dH_diff + (up - dn)
            if S_host is not None:
                dS = dS + embed(S_def) - host_part(S_host)
            support += dev_emb

            # vertex orbitals (heavy atoms of the vertex region)
            v_atoms = sorted(a for a in d["vertex"] if a not in d["adatoms"])
            v_def = np.concatenate([gd.a2o(a, all=True) for a in v_atoms])
            if np.any(o2dev[v_def] < 0):
                raise ValueError(f"{spec.label}: vertex orbitals outside the device")
            own = diff[v_def][:, v_def].toarray()
            v_dev = o2dev[v_def]

            # downfolded adatoms
            fold = []
            if self.downfold:
                for a in d["adatoms"]:
                    if a not in d["vertex"]:
                        raise ValueError(f"{spec.label}: the adatom must lie inside its "
                                         "vertex region (increase radius)")
                    h = gd.a2o(a, all=True)
                    heavy = np.flatnonzero(o2dev >= 0)
                    coupled = np.zeros(Hd.no, dtype=bool)
                    for M in (H_up, H_dn, S_eff):
                        Mc = M.tocsr()
                        coupled[np.unique(Mc[:, h].nonzero()[0])] = True
                        coupled[np.unique(Mc[h, :].nonzero()[1])] = True
                    coupled[h] = False
                    others = [o for o in np.flatnonzero(coupled) if o not in set(heavy)]
                    if others:
                        raise ValueError(f"{spec.label}: adatom {a} couples to orbitals "
                                         f"outside the device: {others[:5]}")
                    c = np.flatnonzero(coupled)
                    blk = lambda M, r_, c_: M.tocsr()[r_][:, c_].toarray()
                    fold.append(dict(owner=k, c_dev=o2dev[c],
                                     H_CH=[blk(H_up, c, h), blk(H_dn, c, h)],
                                     H_HC=[blk(H_up, h, c), blk(H_dn, h, c)],
                                     H_HH=[blk(H_up, h, h), blk(H_dn, h, h)],
                                     S_CH=blk(S_eff, c, h), S_HC=blk(S_eff, h, c),
                                     S_HH=blk(S_eff, h, h)))
            per_defect.append(dict(spec=spec, d=d, v_dev=v_dev, own=own, fold=fold,
                                   anchor=int(host_anchors[k])))
            folded += fold
            shifts.append(shift); rmss.append(rms)

        overlap = int(np.sum(support > 1))
        # Region R: every orbital touched by the perturbation or a vertex
        touched = set()
        for M in (dH_avg, dH_diff, dS):
            r_, c_ = M.nonzero(); touched.update(r_.tolist()); touched.update(c_.tolist())
        for p in per_defect:
            touched.update(p["v_dev"].tolist())
        for f in folded:
            touched.update(f["c_dev"].tolist())
        R = np.array(sorted(touched), dtype=int)
        locmap = {o: i for i, o in enumerate(R.tolist())}
        to_loc = lambda arr: np.array([locmap[int(o)] for o in arr], dtype=int)

        HamL, HamR = self._electrodes()
        nl = HamL.no
        left, right = np.arange(nl), np.arange(n - nl, n)
        if np.intersect1d(R, np.union1d(left, right)).size:
            raise ValueError("Defect region reaches the electrode cells; increase "
                             "N_buffer/N_imp or set embed_radius")

        sub = lambda M: M.tocsr()[R][:, R].toarray()
        dH_avg_R, dH_diff_R = sub(dH_avg), sub(dH_diff)
        dS_R = sub(dS) if S_host is not None else None

        adatoms = [FoldedAdatom(owner=f["owner"], c_loc=to_loc(f["c_dev"]), H_CH=f["H_CH"],
                                H_HC=f["H_HC"], H_HH=f["H_HH"], S_CH=f["S_CH"],
                                S_HC=f["S_HC"], S_HH=f["S_HH"]) for f in folded]
        loc, own_v, tot_v = [], [], []
        for k, p in enumerate(per_defect):
            # vertex region = heavy vertex orbitals + carbons coupled to own adatoms
            extra = [f["c_dev"] for f in p["fold"]]
            v_all = np.union1d(p["v_dev"], np.concatenate(extra)) if extra else np.sort(p["v_dev"])
            l = to_loc(v_all)
            own = np.zeros((len(l), len(l)))
            pos = np.searchsorted(v_all, p["v_dev"])
            own[np.ix_(pos, pos)] = p["own"]
            loc.append(l)
            own_v.append(own)
            tot_v.append(dH_diff_R[np.ix_(l, l)].copy())

        resolved = tuple(p["d"]["r"] for p in per_defect)
        anchors = tuple(p["anchor"] for p in per_defect)
        dist, adist = _distances(hg, anchors, resolved)
        system = DysonSystem(
            Hhost, HamL, HamR, *_semi(self.axis), left, right, R, dH_avg_R, dH_diff_R,
            dS_R, adatoms, loc, own_v, tot_v, resolved, anchors, dist, adist,
            self.downfold,
            info=dict(layout=tuple(lay.values()), n_dist=self.n_dist,
                      energy_shifts_eV=tuple(round(s, 6) for s in shifts),
                      align_rms_meV=tuple(round(1000 * r, 3) for r in rmss),
                      embedding_overlap_orbitals=overlap, delta_scale=lam,
                      vertex_overlap_orbitals=int(np.intersect1d(loc[0], loc[1]).size),
                      vertices=self.vertices))
        data = [p["d"] for p in per_defect]
        tails = _vertex_tails({"own": (own_v, list(system.orbs)), "region": (tot_v, list(system.orbs))},
                              left, right)
        if not self.verbose:          # warnings also without the report
            _report_defect_electrodes([(d["r"].label, d["ends"]) for d in data], tails,
                                      self.ends_tol, self.overlap_tol, verbose=False)
        if self.verbose:
            self._report_host(lay, Hhost, anchors, dist, adist, left, right)
            self._report_superposition(
                data, shifts, rmss, [d["max_delta"] for d in data], overlap,
                "folded into their carbons (Dyson)" if self.downfold else "removed (as in rkky)",
                tails=tails)
            _section("Defect region R")
            print(f"Orbitals in R:      {len(R)} of {n} device orbitals")
            print(f"R in electrode cells: {np.intersect1d(R, np.union1d(left, right)).size} orbitals")
            print(f"Overlap correction: {'yes' if dS_R is not None else 'no (orthogonal basis)'}")
            _report_vertices(self._region_text() + " (static part)",
                             {"rkky_sup / own": (own_v, list(system.orbs)),
                              "region (total)": (tot_v, list(system.orbs))})
            if adatoms:
                _section("Folded adatoms")
                for ad in adatoms:
                    h_up = np.real(np.diag(ad.H_HH[0])).mean()
                    h_dn = np.real(np.diag(ad.H_HH[1])).mean()
                    print(f"Defect {'ij'[ad.owner]} ({resolved[ad.owner].label}):".ljust(20)
                          + f"{ad.S_HH.shape[0]} H orbital(s), {len(ad.c_loc)} coupled carbon orbital(s), "
                          f"max |H_CH| {_max_abs(np.asarray(ad.H_CH[0])):.3f} eV")
                    print(f"{'':20s}H on-site (mean, host frame) up {h_up:+.3f} / down {h_dn:+.3f} eV")
            _close()
            print(system.summary(), flush=True)
        return system


# ===========================================================================
# Exchange
# ===========================================================================
def _perturbation(system: DysonSystem, z, spin):
    """U(z) on R: spin 'nm' (scalar, rkky_sup) or 0/1 (liechtenstein_sup)."""
    if spin == "nm":
        M = system.dH_avg.astype(complex)
    else:
        s = 1.0 if spin == 0 else -1.0
        M = (system.dH_avg + 0.5 * s * system.dH_diff).astype(complex)
    if system.dS is not None:
        M = M - z * system.dS
    for ad in system.adatoms:
        sig = ad.sigma(z, "avg" if spin == "nm" else spin)
        M[np.ix_(ad.c_loc, ad.c_loc)] += sig
    return M


def calculate_exchange_dyson(system: DysonSystem, method: str, energies, contour_weights,
                             prefactor=-1.0 / np.pi, lead_eta: float = 1e-5,
                             vertex_downfold: str = "linear", orbital_resolved: bool = False,
                             verbose: bool = False, calculation_folder=None,
                             results_name: Optional[str] = None, write_results: bool = False,
                             comm=None, vertices: Optional[str] = None,
                             internal_checks: bool = False):
    """J from the Dyson-equation methods. Returns an ExchangeResult (Gamma point).

    Same kernel and conventions as calculate_exchange:
        K(z) = Tr[Delta_i G_ij Delta_j G_ji]           rkky_sup
        K(z) = Tr[Delta_i G_up_ij Delta_j G_dn_ji]     liechtenstein_sup
        J    = 1000 * Im sum_z prefactor * w(z) * K(z)   [meV]
    vertices        : liechtenstein_sup only; None -> the builder's choice.
                      'own' (builder default): own Delta of
                      each single-defect file (cloud model; regions may
                      overlap). 'region': total Delta of the superposed state
                      in each region (region model; disjoint regions).
    vertex_downfold : 'linear' or 'full' correction of the rkky_sup vertices
                      of downfolded adatoms (liechtenstein_sup always 'full').
    comm            : optional MPI communicator; energies are distributed.
    verbose         : True -> setup and result sections (rank 0); 2 -> also one
                      line per contour energy.
    internal_checks : also accept 'rkky' (pristine G0), used by the tests to
                      cross-check PristineTiled.
    """
    import J_dependencies as JD
    method = _method(method, internal_checks=internal_checks)
    if vertex_downfold not in ("linear", "full"):
        raise ValueError("vertex_downfold must be 'linear' or 'full'")
    if vertices is None:
        vertices = system.info.get("vertices", "own")
    if vertices not in ("own", "region"):
        raise ValueError("vertices must be 'own' or 'region'")
    use_total = method == "liechtenstein_sup" and vertices == "region"
    if use_total and system.info.get("vertex_overlap_orbitals", 0):
        raise ValueError("liechtenstein_sup with vertices='region' requires disjoint vertex "
                         "regions; reduce radius or use vertices='own'")
    rank = comm.Get_rank() if comm is not None else 0
    size = comm.Get_size() if comm is not None else 1

    z_array = np.asarray(energies, dtype=complex)
    w_array = np.asarray(contour_weights, dtype=complex)
    H = system.H
    n = H.no
    nR = len(system.R)
    R = system.R
    Hm = sparse.csc_matrix(H.Hk(format="csr"), dtype=complex)
    Sm = sparse.csc_matrix(H.Sk(format="csr"), dtype=complex)
    self_energy = JD.recursive_leads_from_hamiltonians(
        H_device=H, HamL=system.HamL, semi_L=system.semi_L, HamR=system.HamR,
        semi_R=system.semi_R, left_orbs=system.left_orbs, right_orbs=system.right_orbs,
        eta=lead_eta, polarized_leads=False)
    k0 = np.zeros(3)
    li, lj = system.loc
    p = complex(prefactor)
    eye = np.eye(nR)
    rhs = np.zeros((n, nR), dtype=complex)
    rhs[R, np.arange(nR)] = 1.0

    def vertex(k, z):
        static = system.vertex_total[k] if use_total else system.vertex_own[k]
        D = static.astype(complex)
        l = system.loc[k]
        for ad in system.adatoms:
            if ad.owner != k:
                continue
            if method == "liechtenstein_sup" or vertex_downfold == "full":
                corr = ad.vertex_full(z)
            else:
                corr = ad.vertex_linear(z)
            pos = np.searchsorted(l, ad.c_loc)
            D[np.ix_(pos, pos)] += corr
        return D

    def U(z, spin):
        return _perturbation(system, z, spin)

    kernels = np.zeros((1, len(z_array)), dtype=complex)
    orb_int = np.zeros((len(li), len(lj)), dtype=complex) if orbital_resolved else None
    orb_k = (np.zeros((1, len(z_array), len(li), len(lj)), dtype=complex)
             if orbital_resolved and size == 1 else None)
    import time
    t_start = time.perf_counter()
    if verbose and rank == 0:
        _section("Dyson exchange")
        vert = (f"vertices='{vertices}'" if method == "liechtenstein_sup"
                else f"own vertices, vertex_downfold='{vertex_downfold}'")
        print(f"Method:             {method} ({vert})")
        print(f"Defect region R:    {nR} orbitals of {n}")
        print(f"Vertex orbitals:    i={len(li)}, j={len(lj)} "
              f"(overlap {np.intersect1d(system.R[li], system.R[lj]).size})")
        print(f"Folded adatoms:     {len(system.adatoms)}")
        print(f"Contour energies:   {len(z_array)}" + (f" on {size} ranks" if size > 1 else ""))
        print(f"Prefactor:          {prefactor.real if isinstance(prefactor, complex) else prefactor:.6f}")
        print("========================================", flush=True)

    for ie in range(rank, len(z_array), size):
        z = z_array[ie]
        sigma_leads = JD._lead_embedding(n, self_energy(z, k0, 0))
        lu = splu((z * Sm - Hm - sigma_leads).tocsc())
        G0 = lu.solve(rhs)[R, :]
        del lu
        if method == "rkky":
            Gs = (G0, G0)
        elif method == "rkky_sup":
            G = np.linalg.solve(eye - G0 @ U(z, "nm"), G0)
            Gs = (G, G)
        else:
            Gs = tuple(np.linalg.solve(eye - G0 @ U(z, s), G0) for s in (0, 1))
        Di, Dj = vertex(0, z), vertex(1, z)
        x = Gs[0][np.ix_(li, lj)] @ Dj          # G_ij Delta_j   (up for LKAG)
        y = Gs[1][np.ix_(lj, li)] @ Di          # G_ji Delta_i   (down for LKAG)
        contribution = x * y.T
        weighted = p * w_array[ie]
        kernels[0, ie] = weighted * contribution.sum()
        if orbital_resolved:
            orb_int += weighted * contribution
            if orb_k is not None:
                orb_k[0, ie] = weighted * contribution
        if verbose and verbose >= 2:
            print(f"E={ie + 1}/{len(z_array)} K={kernels[0, ie]:.6g}", flush=True)

    if comm is not None:
        from mpi4py import MPI
        tot = np.zeros_like(kernels) if rank == 0 else None
        comm.Reduce(kernels, tot, op=MPI.SUM, root=0)
        kernels = tot
        if orbital_resolved:
            tot_o = np.zeros_like(orb_int) if rank == 0 else None
            comm.Reduce(orb_int, tot_o, op=MPI.SUM, root=0)
            orb_int = tot_o
        if rank != 0:
            return None

    J_k = 1000.0 * np.imag(kernels.sum(axis=1))
    if verbose:
        _section("Result")
        print(f"J = {J_k[0]:.6f} meV   ({method}, {time.perf_counter() - t_start:.1f} s)")
        _close()
    result = JD.ExchangeResult(
        method=method, J_meV=float(J_k[0]), J_k_meV=J_k, kpoints=k0[None, :],
        k_weights=np.ones(1), energies=z_array, contour_weights=w_array,
        weighted_kernel_kE=kernels, J_orb_meV=(1000.0 * orb_int.imag if orbital_resolved else None),
        prefactor=p, spin_average=False, orb_i=R[li].copy(), orb_j=R[lj].copy(),
        weighted_orbital_kernel_kE=orb_k, distance=system.anchor_distance)
    if calculation_folder is not None and write_results:
        folder = os.path.join(str(calculation_folder), results_name or f"Results_{method}")
        os.makedirs(folder, exist_ok=True)
        result.save(os.path.join(folder, "ExchangeResult.npz"))
        if orb_k is not None:
            np.save(os.path.join(folder, "weighted_orbital_kernel.npy"), orb_k)
    return result


# ===========================================================================
# Density of states of the defect region
# ===========================================================================
def dos_dyson(system: DysonSystem, method: str, energies, eta: float = 0.01,
              lead_eta: float = 1e-5) -> dict:
    """Mulliken DOS of the defect region R (and of folded adatoms) from Dyson.

    method : 'rkky_sup'          nonmagnetic reference H0 + V1 + V2 (one channel)
             'liechtenstein_sup' (or 'lkag_sup') spin up and down
             'rkky'              pristine host on the same region, for comparison
    For z = E + i*eta:  DOS_a(E) = -Im (G S)_aa / pi  for every orbital a of R.
    G includes the coupling of R to its overlap neighbours, the overlap change
    dS, and for folded adatoms the C-H overlap terms, so the values equal the
    Mulliken DOS of calculate_dos_embedding on the explicit Superposition device
    for the same orbitals. This is a DOS PROJECTED on R, not the total device DOS.

    Returns dict(energies, spins, dos (nspin, nE): sum over R and the adatom
    orbitals, orbital_dos (nspin, nE, nR), orbitals (device orbitals of R),
    adatom_dos (nspin, nE, n_adatom_orbitals) or None).
    """
    import J_dependencies as JD
    m = _method(method, internal_checks=True)
    spins = ("nm",) if m in ("rkky", "rkky_sup") else (0, 1)
    E = np.asarray(energies, dtype=float)
    H = system.H
    n, R = H.no, system.R
    nR = len(R)
    Hm = sparse.csc_matrix(H.Hk(format="csr"), dtype=complex)
    Sm = sparse.csr_matrix(H.Sk(format="csr"), dtype=float)
    # R plus its overlap neighbours (needed for the Mulliken sum)
    Rp = np.union1d(R, np.unique(Sm[R].nonzero()[1]))
    posR = np.searchsorted(Rp, R)
    S_eff = Sm[R][:, Rp].toarray()                        # S_ab, a in R, b in Rp
    if system.dS is not None:
        S_eff[:, posR] += system.dS                       # perturbed overlap on R
    self_energy = JD.recursive_leads_from_hamiltonians(
        H_device=H, HamL=system.HamL, semi_L=system.semi_L, HamR=system.HamR,
        semi_R=system.semi_R, left_orbs=system.left_orbs, right_orbs=system.right_orbs,
        eta=lead_eta, polarized_leads=False)
    rhs = np.zeros((n, len(Rp)), dtype=complex)
    rhs[Rp, np.arange(len(Rp))] = 1.0
    use_ad = m != "rkky" and len(system.adatoms) > 0
    n_h = sum(ad.S_HH.shape[0] for ad in system.adatoms) if use_ad else 0
    orb_dos = np.zeros((len(spins), len(E), nR))
    ad_dos = np.zeros((len(spins), len(E), n_h)) if use_ad else None

    for ie, e in enumerate(E):
        z = e + 1j * eta
        A0 = (z * Sm - Hm - JD._lead_embedding(n, self_energy(z, np.zeros(3), 0))).tocsc()
        X = splu(A0).solve(rhs)[Rp, :]                    # G0(Rp, Rp)
        G0_RR, G0_RRp = X[np.ix_(posR, posR)], X[posR, :]
        for isp, spin in enumerate(spins):
            if m == "rkky":
                G_RRp = G0_RRp
            else:
                G_RRp = np.linalg.solve(np.eye(nR) - G0_RR @ _perturbation(system, z, spin),
                                        G0_RRp)
            GS = np.einsum("ab,ab->a", G_RRp, S_eff)      # (G S)_aa = sum_b G_ab S_ba
            if use_ad:
                G_RR = G_RRp[:, posR]
                k = 0
                for ad in system.adatoms:
                    P, Q, Rb = ad._blocks(z, "avg" if spin == "nm" else spin)
                    Qi = np.linalg.inv(Q)
                    c = ad.c_loc
                    G_cc = G_RR[np.ix_(c, c)]
                    G_cH = G_cc @ P @ Qi                  # G(C, H)
                    G_Hc = Qi @ Rb @ G_cc                 # G(H, C)
                    G_HH = Qi + Qi @ Rb @ G_cc @ P @ Qi   # G(H, H)
                    GS[c] += np.einsum("ah,ha->a", G_cH, ad.S_HC)
                    h_ds = np.einsum("hb,bh->h", G_HH, ad.S_HH) + np.einsum("hc,ch->h", G_Hc, ad.S_CH)
                    ad_dos[isp, ie, k:k + len(h_ds)] = -h_ds.imag / np.pi
                    k += len(h_ds)
            orb_dos[isp, ie] = -GS.imag / np.pi
    dos = orb_dos.sum(axis=2) + (ad_dos.sum(axis=2) if use_ad else 0.0)
    return dict(energies=E, spins=("0",) if spins == ("nm",) else ("up", "down"),
                dos=dos, orbital_dos=orb_dos, orbitals=R.copy(), adatom_dos=ad_dos,
                is_total=False)
