"""tb2j_bridge.py - run TB2J on any collinear spin-polarized sisl Hamiltonian and convert
its output to the OpenJij convention.

TB2J: X. He, N. Helbig, M. J. Verstraete, E. Bousquet, Comput. Phys. Commun. 264, 107938
(2021); https://github.com/mailhexu/TB2J (tested with TB2J 0.9.22, HamiltonIO, sisl 0.20).

Why a bridge: siesta2J.py reads the full SIESTA basis from the .fdf (for the 730-atom SZP
ribbon: 6570 orbitals, dense H(k), S(k), eigenvectors for every k -> tens of GB). This module
feeds TB2J the reduced sisl Hamiltonian that OpenJij already builds (e.g. pz+dz), so TB2J and
OpenJij see exactly the same H, S and k grid.

Conventions (checked numerically in the benchmark notebook)
    TB2J      E = - sum_{i != j} J_ij e_i . e_j  (unit vectors, each pair twice, J>0 FM)
              J_ij = Im[(1/4pi) int dz Tr(D_i G^up_ij(R) D_j G^dn_ji(-R))] / sign(m_i m_j)
              D_i = on-site atomic block of H_up - H_dn  (R = 0).
    OpenJij   J = (E_FM - E_AFM)/2  (J<0 FM),  J = -1/(2pi) Im sum_z w Tr[D_I G_IJ D_J G_JI]
              D_I = the whole region block (on-site AND inter-atomic elements).
    Rigid rotation of region J against region I (force theorem, FM reference):
              J_OpenJij(I,J) = -2 sum_{a in I, b in J, R} sign(m_a m_b) J^TB2J_ab(R)
                             = -2 sum raw_ab(R),   raw = Im(JJ) before TB2J divides by the sign.
    Identical only if the inter-atomic part of D inside each region is dropped; with the full
    region block (OpenJij default) the two differ by the D_ab (a != b) terms.

With OpenJij runs: tb2j_from_run(run) runs TB2J on the Hamiltonian and the regions of a
periodic liechtenstein run (run_exchange(..., periodic=True)), with its k grid and the
temperature of its contour, and returns TB2J's region J next to the OpenJij value.

Speed: compact=True stores Green's functions only on the magnetic orbitals (TB2J otherwise
builds the full nbasis x nbasis G(R) at every energy and copies (nR, n_i, nbasis) slices for
every atom pair). The arithmetic for the magnetic block is the same as TB2J's
(eigen_to_G with TB2J's own eigenvectors, band window and k weights); validated against the
unmodified TB2J in the notebook.
"""
from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.units import kB

from HamiltonIO.siesta import SislParser
from TB2J.exchangeCL2 import ExchangeCL2
from TB2J.green import TBGreen

__all__ = ["tb2j_models", "run_tb2j", "region_J", "pair_J", "TB2JResult",
           "magnetic_atoms_from_delta", "region_J_from_pickle", "tb2j_from_run"]


# --------------------------------------------------------------------------- models
class SislHamiltonianParser(SislParser):
    """HamiltonIO's SislParser, but from a sisl.Hamiltonian in memory instead of an .fdf."""

    def __init__(self, H, orth=False):
        self.fdf = None
        self.ham = H
        self.spin = H.spin
        self.ispin = None
        self.read_H_soc = False
        self.orth = orth

    def read_geometry(self, fdf=None):
        g = self.ham.geometry
        self.positions = g.xyz
        self.cell = np.array(g.lattice.cell)
        self.atoms = Atoms(numbers=[int(z) for z in g.atoms.Z], positions=g.xyz,
                           cell=self.cell, pbc=True)
        return g, self.atoms

    def read_nel(self, fdf=None):
        self.nel = None
        return None

    def read_so_strength(self, fdf=None):
        return 1.0


def tb2j_models(H):
    """(model_up, model_dn) for TB2J from a collinear spin-polarized sisl Hamiltonian.
    The energy zero of H must be the Fermi level (sisl/OpenJij Hamiltonians are)."""
    if not H.spin.is_polarized:
        raise ValueError("TB2J collinear mode needs a spin-polarized Hamiltonian")
    up, dn = SislHamiltonianParser(H).get_model()
    return up, dn


def magnetic_atoms_from_delta(H, w_min=0.01):
    """Atoms with ||(H_up - H_dn)_aa||_F >= w_min (eV); same weight as OpenJij's region_weight='delta'."""
    import regions as _regions                     # OpenJij module
    w = _regions.atom_delta_weights(H)
    return np.flatnonzero(w >= w_min), w


# --------------------------------------------------------------------------- speed-up
_ORIG_GET_GR = TBGreen.get_GR


def _compact_get_GR(self, Rpts, energy, Gk_all=None):
    """G(R) on the magnetic orbitals only (shape (nR, n_mag_orb, n_mag_orb))."""
    m = getattr(self, "_orb_subset", None)
    if m is None or Gk_all is not None:
        return _ORIG_GET_GR(self, Rpts, energy, Gk_all)
    Rvecs = np.array(Rpts)
    GR = np.zeros((len(Rvecs), len(m), len(m)), dtype=complex)
    for ik, kpt in enumerate(self.kpts):
        ev = self.get_evalue(ik)
        V = np.asarray(self.get_evecs(ik))[m]          # rows of the magnetic orbitals
        Gk = (V * (1.0 / (-ev + (energy + self.efermi)))) @ V.conj().T
        phase = np.exp(self.k2Rfactor * np.dot(Rvecs, kpt)) * self.kweights[ik]
        GR += Gk[None, :, :] * phase[:, None, None]
    return GR


TBGreen.get_GR = _compact_get_GR                     # inactive unless _orb_subset is set


def _compactify(exch):
    mag = list(exch.ind_mag_atoms)
    m = np.concatenate([np.asarray(exch.orb_dict[a], dtype=int) for a in mag])
    new, s = {}, 0
    for a in mag:
        n = len(exch.orb_dict[a])
        new[a] = list(range(s, s + n))
        s += n
    exch.Delta = exch.Delta[np.ix_(m, m)]
    exch.rho_up = exch.rho_up[np.ix_(m, m)]
    exch.rho_dn = exch.rho_dn[np.ix_(m, m)]
    exch.orb_dict = new
    exch.Gup._orb_subset = m
    exch.Gdn._orb_subset = m


def _index_spin_from_list(exch):
    """TB2J 0.9.22 numbers the spins in the output from magnetic_elements, ignoring
    index_magnetic_atoms; use the list actually used in the calculation instead."""
    def _prep():
        pos = {a: i for i, a in enumerate(exch.ind_mag_atoms)}
        exch.index_spin = [pos.get(i, -1) for i in range(len(exch.atoms))]
    exch._prepare_index_spin = _prep


# --------------------------------------------------------------------------- run
class TB2JResult:
    """raw[(R, a, b)]: Im(JJ) in meV before TB2J divides by sign(m_a m_b);
    J[(R, a, b)]: TB2J's J_ab in meV (TB2J convention); spin[a]: TB2J moment (density matrix)."""

    def __init__(self, exch):
        self.exch = exch
        self.atoms = exch.atoms
        self.magnetic_atoms = list(exch.ind_mag_atoms)
        self.spin = np.asarray(exch.spinat[:, 2]).copy()
        self.raw = {k: 1e3 * float(np.imag(v)) for k, v in exch.JJ.items()
                    if not (k[0] == (0, 0, 0) and k[1] == k[2])}
        s = np.sign(self.spin)
        self.J = {k: v / (s[k[1]] * s[k[2]]) for k, v in self.raw.items()}

    def to_openjij(self, a, b, R=None):
        """-2 raw_ab summed over R (or at one R): OpenJij-convention contribution of the pair."""
        return -2.0 * sum(v for (r, i, j), v in self.raw.items()
                          if i == a and j == b and (R is None or tuple(r) == tuple(R)))


def run_tb2j(H, magnetic_atoms, kmesh=(9, 1, 1), smearing_K=100.0, nz=200, Rcut=None,
             compact=True, output_path=None, use_cache=False, nproc=1, description="",
             progress=False):
    """Run TB2J's collinear LKAG (ExchangeCL2) on the sisl Hamiltonian H.

    magnetic_atoms : 0-based atom indices of H treated as spins (pairs of all of them)
    kmesh          : the DFT k grid (Gamma-centred, odd); Rcut=None keeps every R of that
                     grid, so sum_R J_ab(R) equals the k-space trace exactly
    smearing_K, nz : TB2J's Ozaki continued-fraction contour (Fermi function at T);
                     at 50-100 K use nz >= 200 when the basis reaches high energies (d's)
    output_path    : if given, TB2J's usual files (exchange.out, TB2J.pickle, ...) are written
    progress       : show TB2J's progress bar over the energy points
    """
    import TB2J.exchangeCL2 as _ecl2
    _tqdm = _ecl2.tqdm
    if not progress:
        _ecl2.tqdm = lambda it, *a, **k: it
    up, dn = tb2j_models(H)
    basis = dict(zip(up.orbs, range(up.nbasis)))
    exch = ExchangeCL2(tbmodels=(up, dn), atoms=up.atoms, basis=basis, efermi=0.0,
                       magnetic_elements=[], index_magnetic_atoms=[int(a) for a in magnetic_atoms],
                       kmesh=list(kmesh), smearing=smearing_K * kB, nz=nz, Rcut=Rcut,
                       use_cache=use_cache, nproc=nproc, description=description,
                       output_path=output_path or "TB2J_results")
    _index_spin_from_list(exch)
    if compact:
        _compactify(exch)
    try:
        exch.calculate_all()
    finally:
        _ecl2.tqdm = _tqdm
    if output_path:
        exch.write_output(output_path)
    return TB2JResult(exch)


# --------------------------------------------------------------------------- analysis
def region_J(res: TB2JResult, region_i, region_j):
    """OpenJij-convention J (meV) between two disjoint atom sets from TB2J pair data."""
    I, J = set(int(a) for a in region_i), set(int(a) for a in region_j)
    if I & J:
        raise ValueError("regions overlap")
    return -2.0 * sum(v for (R, a, b), v in res.raw.items() if a in I and b in J)


def pair_J(res: TB2JResult, a, b):
    """TB2J J_ab summed over R (TB2J convention, meV) and its OpenJij-convention value."""
    tb = sum(v for (R, i, j), v in res.J.items() if i == a and j == b)
    return tb, res.to_openjij(a, b)


def region_J_from_pickle(path, region_i, region_j, magnetic_atoms=None):
    """OpenJij-convention J (meV) from a finished TB2J run (TB2J_results/TB2J.pickle),
    e.g. produced by siesta2J.py.  region_i/j: 0-based atom indices of the TB2J structure.

    magnetic_atoms: the 0-based list given to --index_magnetic_atoms (minus one), if any.
    TB2J 0.9.22 numbers the spins in the output from --elements only, so with a subset the
    spin -> atom map stored in the pickle is wrong; pass the list to rebuild it."""
    from TB2J.io_exchange import SpinIO
    sp = SpinIO.load_pickle(path=path)
    if magnetic_atoms is not None:
        atom_of = {i: int(a) for i, a in enumerate(magnetic_atoms)}
    else:
        atom_of = dict(sp.ind_atoms)
    m = np.asarray(sp.spinat)
    m = m[:, 2] if m.ndim == 2 else m
    s = np.sign(m)
    I, J = set(int(a) for a in region_i), set(int(a) for a in region_j)
    tot = 0.0
    for (R, i, j), val in sp.exchange_Jdict.items():
        a, b = atom_of[i], atom_of[j]
        if a in I and b in J:
            tot += 1e3 * float(np.real(val)) * s[a] * s[b]      # raw = J_ab sign(m_a m_b)
    return -2.0 * tot


def tb2j_from_run(run, smearing_K=None, nz=None, output_path=None, compact=True):
    """TB2J on the Hamiltonian, regions and k grid of a periodic OpenJij LKAG run.

    run : ExchangeRun of run_exchange('liechtenstein', ..., periodic=True)
          (the device runs have open boundaries and electrodes, which TB2J cannot model).
    smearing_K : default = temperature of the run's contour; nz default: 200 below
          80 K, else 100 (Ozaki CFR; TB2J keeps bands up to E_F + 5.1 eV).
    Returns dict(tb2j=TB2JResult, J_tb2j=region J from TB2J (on-site vertex, meV),
    J_openjij=run.J_meV, vertex=run's vertex).  With vertex='onsite' in the run the two
    numbers must agree up to the contour and the band window (1e-4 relative or better).
    """
    obj = run.obj
    if getattr(obj, "kind", None) != "periodic":
        raise ValueError("tb2j_from_run needs a periodic liechtenstein run (periodic=True)")
    n_k, shift = obj.kgrid
    if abs(shift) > 1e-9 or n_k % 2 == 0:
        raise ValueError("TB2J uses a Gamma-centred odd k grid; the run's grid is "
                         f"{n_k} (shift {shift})")
    T = smearing_K if smearing_K is not None else float(run.contour.get("T") or 100.0)
    nz = nz if nz is not None else (200 if T < 80 else 100)
    g = obj.H.geometry
    I, J = (np.unique(g.o2a(o)) for o in obj.orbs)
    mag = np.sort(np.concatenate([I, J]))
    kmesh = [1, 1, 1]
    kmesh[obj.axis] = n_k
    res = run_tb2j(obj.H, mag, kmesh=tuple(kmesh), smearing_K=T, nz=nz, compact=compact,
                   output_path=output_path)
    sign = -1.0 if run.config.get("reference") == "AFM" else 1.0      # same convention as OpenJij
    return dict(tb2j=res, J_tb2j=sign * region_J(res, I, J), J_openjij=run.J_meV,
                vertex=run.config.get("vertex", "full"), n_spins=len(mag), smearing_K=T, nz=nz)
