"""Minimal disk/MPI helpers for J_dependencies.py.

TBTGF uses bulk B(z)=z*S-H-Sigma, not the raw self-energy Sigma.
This module writes both spin channels, regardless of which one exchange uses.
"""
from pathlib import Path
import hashlib
import json
import os
import numpy as np
from scipy import sparse
import warnings

def mpi_comm(enabled=False):
    if not enabled:
        return None, 0, 1
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    return comm, comm.Get_rank(), comm.Get_size()


def _atomic_npy(path, array):
    path = Path(path)
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with open(tmp, 'wb') as f:
        np.save(f, array, allow_pickle=False)
    os.replace(tmp, path)


def _atomic_npz(path, **data):
    path = Path(path)
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with open(tmp, 'wb') as f:
        np.savez_compressed(f, **data)
    os.replace(tmp, path)


def save_local_delta(filename, Delta, orbitals, **metadata):
    """Write a compact CSR Delta without object arrays/pickle."""
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    D = sparse.csr_matrix(Delta, dtype=np.complex128)
    _atomic_npz(path, data=D.data, indices=D.indices, indptr=D.indptr,
                shape=np.asarray(D.shape), orb_idx=np.asarray(orbitals, dtype=np.int64),
                **metadata)


def load_local_delta(filename):
    """Return (compact CSR Delta, global orbital indices) without pickle."""
    with np.load(filename, allow_pickle=False) as f:
        D = sparse.csr_matrix((f['data'], f['indices'], f['indptr']),
                              shape=tuple(f['shape']))
        idx = f['orb_idx'].copy()
    return D, idx


def _hash_array(hasher, arr):
    arr = np.ascontiguousarray(arr)
    hasher.update(str(arr.dtype).encode())
    hasher.update(str(arr.shape).encode())
    hasher.update(memoryview(arr).cast('B'))


def _fingerprint_hamiltonian(H):
    """Cheap matrix-based fingerprint for cache mismatch detection."""
    h = hashlib.sha256()
    _hash_array(h, np.asarray(H.nsc))
    _hash_array(h, np.asarray(H.cell))
    _hash_array(h, np.asarray(H.geometry.xyz))
    _hash_array(h, np.asarray(H.geometry.atoms.Z))
    # Hash occupied sparse matrix entries, not unused SparseCSR storage slots.
    # Two non-equivalent k-points also detect changes to periodic couplings.
    for k in ((0.0, 0.0, 0.0), (0.137, 0.271, 0.193)):
        for spin in ((0, 1) if H.spin.is_polarized else (0,)):
            # spin= only for polarized matrices: recent sisl rejects it otherwise
            M = sparse.csr_matrix(H.Hk(k=k, spin=spin, format="csr") if H.spin.is_polarized
                                  else H.Hk(k=k, format="csr"))
            M.sort_indices()
            for a in (M.data, M.indices, M.indptr):
                _hash_array(h, a)
        S = sparse.csr_matrix(H.Sk(k=k, format="csr"))
        S.sort_indices()
        for a in (S.data, S.indices, S.indptr):
            _hash_array(h, a)
    return h.hexdigest()


def _fingerprint_delta(D, idx):
    h = hashlib.sha256()
    D = sparse.csr_matrix(D)
    for x in (D.data, D.indices, D.indptr, np.asarray(D.shape), np.asarray(idx)):
        _hash_array(h, x)
    return h.hexdigest()


def _k_counts(nk, axes):
    counts = [1, 1, 1]
    for x, a in zip(nk, axes):
        counts['xyz'.index(a.lower())] = max(1, int(x))
    return counts


def _orbital_interfaces(H_device, HamL, HamR, left_orbs, right_orbs):
    nl, nr = HamL.no, HamR.no
    li = np.arange(nl) if left_orbs is None else np.asarray(left_orbs, dtype=int)
    ri = (np.arange(H_device.no - nr, H_device.no) if right_orbs is None
          else np.asarray(right_orbs, dtype=int))
    if li.shape != (nl,) or ri.shape != (nr,):
        raise ValueError('Interface orbital indices do not match electrode dimensions')
    for idx in (li, ri):
        if (np.any(idx < 0) or np.any(idx >= H_device.no)
                or len(np.unique(idx)) != len(idx)):
            raise ValueError('Invalid interface orbital indices')
    # NEW: disallow embedding two distinct leads on the same orbitals.
    if np.intersect1d(li, ri).size:
        raise ValueError('Left and right electrode interfaces overlap')
    return li, ri

def _tbtgf_metadata(H_device, HamL, HamR, semi_L, semi_R,
                    E, kvals, kw, li, ri, lead_eta, mu):
    """
    Metadata for reusable electrode self-energy cache.

    Device size and device orbital indices are deliberately excluded
    because the TBTGF files store electrode-local quantities.
    """
    return dict(
        version=2,
        format='sisl TBTGF bulk B=zS-H-Sigma',

        energies=[
            [float(v.real), float(v.imag)] for v in E
        ],

        kpoints=np.asarray(kvals).tolist(),
        k_weights=np.asarray(kw).tolist(),

        semi_L=semi_L,
        semi_R=semi_R,

        lead_eta=float(lead_eta),
        mu=float(mu),
        nspin=2,

        electrode_left_sha256=_fingerprint_hamiltonian(HamL),
        electrode_right_sha256=_fingerprint_hamiltonian(HamR),
    )

# Test change
def _check_metadata(path, expected, allow_electrode_hash_mismatch=False):
    # NEW: only electrode caches may opt into the existing hash warning.
    with open(path) as f: 
        actual = json.load(f)
    different_keys = [ k for k in (actual.keys() | expected.keys()) if actual.get(k) != expected.get(k)]
    if not different_keys:
        return False
    hash_keys = {"electrode_left_sha256","electrode_right_sha256",}
    if (allow_electrode_hash_mismatch and set(different_keys).issubset(hash_keys)):
        print(f"WARNING: Electrode fingerprints differ in {path}: {different_keys}.Reusing cached self-energies.",flush=True)
        return True
    raise ValueError(f"Incompatible cache: {path}. Different parameters: {different_keys}")

def _write_json(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    os.replace(tmp, path)

def _scratch_name(tmp_dir, side, spin, ik, ie):
    return tmp_dir / f'{side}_S{spin}_K{ik:04d}_E{ie:08d}.npy'


def _validate_tbtgf(path, E, kvals, n):
    import sisl
    with sisl.io.tbtgfSileTBtrans(str(path), 'r') as f:
        nspin, no, ks, es = f.read_header()
        if nspin != 2 or no != n:
            raise ValueError(f'{path}: expected two spin channels and {n} orbitals')
        if (len(es) != len(E) or len(ks) != len(kvals)
                or not np.allclose(es, E, rtol=0, atol=2e-8)
                or not np.allclose(ks, kvals, rtol=0, atol=1e-10)):
            raise ValueError(f'{path}: energy/k-point grid differs')


def write_electrode_tbtgf(folder, H_device, HamL, semi_L, HamR, semi_R,
                            energies, nk=(1, 0, 0), axes=('x', 'y', 'z'),
                            trs=False, left_orbs=None, right_orbs=None,
                            lead_eta=1e-5, mu=0.0, mpi=False,
                            keep_tmp=False):
    """MPI-safe writer of Left.TBTGF and Right.TBTGF with *both* spins.

    MPI ranks calculate disjoint complex energies to per-entry .npy files.
    Rank 0 assembles one valid sequential TBTGF for each electrode.
    With unpolarized electrodes, both spins are independently computed from
    an up-converted polarized electrode (numerically identical channels).
    No rank holds self-energies for all energies in RAM.
    """
    import sisl
    comm, rank, size = mpi_comm(mpi)
    E = np.asarray(energies, dtype=np.complex128)
    if E.ndim != 1 or E.size == 0:
        raise ValueError('energies must be a nonempty complex contour')
    folder = Path(folder)
    tmp_dir = folder / 'TBTGF_tmp'
    bz = sisl.MonkhorstPack(H_device, _k_counts(nk, axes), trs=trs)
    kvals = np.asarray(bz.k)
    kw = np.asarray(bz.weight)
    li, ri = _orbital_interfaces(H_device, HamL, HamR, left_orbs, right_orbs)
    meta = _tbtgf_metadata(H_device, HamL, HamR, semi_L, semi_R,
                           E, kvals, kw, li, ri, lead_eta, mu)
    paths = (folder / 'Left.TBTGF', folder / 'Right.TBTGF')
    manifest = folder / 'config.json'
    setup_error = None
    if rank == 0:
        try:
            folder.mkdir(parents=True, exist_ok=True)
            tmp_dir.mkdir(parents=True, exist_ok=True)
            if manifest.exists():
                _check_metadata(manifest, meta, allow_electrode_hash_mismatch=True)
            else:
                _write_json(manifest, meta)
        except Exception as exc:
            setup_error = f'{type(exc).__name__}: {exc}'
    if comm is not None:
        setup_error = comm.bcast(setup_error, root=0)
    if setup_error:
        raise RuntimeError('TBTGF setup failed: ' + setup_error)
    # Rank 0 alone verifies existing binary files; broadcast the result.
    complete = False
    if rank == 0 and all(p.exists() for p in paths):
        try:
            _validate_tbtgf(paths[0], E, kvals, HamL.no)
            _validate_tbtgf(paths[1], E, kvals, HamR.no)
            complete = True
        except Exception:
            pass
    if comm is not None:
        complete = comm.bcast(complete if rank == 0 else None, root=0)
    if complete:
        return paths

    # TBTGF must have nspin=2, including when original leads are unpolarized.
    leads = [HamL if HamL.spin.is_polarized else HamL.transform(spin='polarized'),
             HamR if HamR.spin.is_polarized else HamR.transform(spin='polarized')]
    local_error = None
    try:
        for side, H, semi in zip(('Left', 'Right'), leads, (semi_L, semi_R)):
            rsi = sisl.RecursiveSI(H, semi, eta=lead_eta)
            for ie in range(rank, len(E), size):
                for ik, k in enumerate(kvals):
                    for spin in (0, 1):
                        path = _scratch_name(tmp_dir, side, spin, ik, ie)
                        if path.exists():
                            try:
                                arr = np.load(path, allow_pickle=False, mmap_mode='r')
                                if arr.shape == (H.no, H.no) and np.all(np.isfinite(arr)):
                                    continue
                            except Exception:
                                pass
                        sigma = rsi.self_energy(E[ie], k=k, spin=spin)
                        _atomic_npy(path, np.asarray(sigma, dtype=np.complex128))
                        del sigma
    except Exception as exc:
        local_error = f'rank {rank}: {type(exc).__name__}: {exc}'
    if comm is not None:
        errors = comm.allgather(local_error)
    else:
        errors = [local_error]
    errors = [e for e in errors if e is not None]
    if errors:
        raise RuntimeError('TBTGF generation failed: ' + '; '.join(errors))

    # One writer: the Fortran sequential TBTGF format cannot be written
    # concurrently by MPI ranks. Stream one energy at a time from scratch.
    assembly_error = None
    if rank == 0:
        try:
            for side, H, semi, final in zip(('Left', 'Right'), leads,
                                            (semi_L, semi_R), paths):
                axis = 'ABC'.index(semi[1])
                Hsurf = H.copy()
                Hsurf.set_nsc([1 if i == axis else None for i in range(3)])
                temp_tbtgf = final.with_name(final.name + '.tmp')
                try:
                    with sisl.io.tbtgfSileTBtrans(str(temp_tbtgf), 'w') as f:
                        f.write_header(bz, E, mu=mu, obj=H)
                        current_H = current_S = None
                        for counter, (spin, new_k, k, z) in enumerate(f):
                            ie = counter % len(E)
                            ik = (counter // len(E)) % len(kvals)
                            if new_k:
                                current_H = np.asarray(
                                    Hsurf.Hk(k=k, spin=spin, format='array'), dtype=np.complex128)
                                current_S = np.asarray(
                                    Hsurf.Sk(k=k, format='array'), dtype=np.complex128)
                                f.write_hamiltonian(current_H, current_S)
                            sigma = np.load(_scratch_name(tmp_dir, side, spin, ik, ie),
                                            allow_pickle=False, mmap_mode='r')
                            bulk = z * current_S - current_H - sigma
                            f.write_self_energy(bulk)
                            del sigma, bulk
                    os.replace(temp_tbtgf, final)
                finally:
                    if temp_tbtgf.exists():
                        temp_tbtgf.unlink()
            _validate_tbtgf(paths[0], E, kvals, HamL.no)
            _validate_tbtgf(paths[1], E, kvals, HamR.no)
            if not keep_tmp:
                for p in tmp_dir.glob('*.npy'):
                    p.unlink()
                tmp_dir.rmdir()
        except Exception as exc:
            assembly_error = f'{type(exc).__name__}: {exc}'
    if comm is not None:
        assembly_error = comm.bcast(assembly_error, root=0)
    if assembly_error:
        raise RuntimeError('TBTGF assembly failed: ' + assembly_error)
    if comm is not None:
        comm.Barrier()
    return paths


class TBTGFProvider:
    """Read raw electrode Sigma from TBTGF, indexed exactly by (ie,ik,spin)."""
    def __init__(self, folder, H_device, HamL, HamR, semi_L, semi_R,
                 energies, kpoints, k_weights, left_orbs=None, right_orbs=None,
                 lead_eta=1e-5, mu=0.0):
        import sisl
        folder = Path(folder)
        E = np.asarray(energies, dtype=np.complex128)
        self.E, self.k = E, np.asarray(kpoints)
        li, ri = _orbital_interfaces(H_device, HamL, HamR,
                                     left_orbs, right_orbs)
        expected = _tbtgf_metadata(H_device, HamL, HamR, semi_L, semi_R,
                                    E, self.k, k_weights, li, ri, lead_eta, mu)
        _check_metadata(folder / 'config.json', expected,
                        allow_electrode_hash_mismatch=True)
        self.files = [sisl.io.tbtgfSileTBtrans(str(folder / name), 'r')
                      for name in ('Left.TBTGF', 'Right.TBTGF')]
        self.idx = [li, ri]
        self.HS = [None, None]
        self.last_key = [None, None]
        for f, nelec in zip(self.files, (HamL.no, HamR.no)):
            nspin, n, ks, es = f.read_header()
            if nspin != 2 or n != nelec or len(es) != len(E) or len(ks) != len(self.k):
                raise ValueError('Wrong TBTGF dimensions')
            if (not np.allclose(es, E, rtol=0, atol=2e-8)
                    or not np.allclose(ks, self.k, rtol=0, atol=1e-10)):
                raise ValueError('Wrong TBTGF energy or k-point grid')

    def __call__(self, z, k, spin, ie, ik):
        if abs(z - self.E[ie]) > 1e-10 or not np.allclose(k, self.k[ik], atol=1e-12):
            raise ValueError('Requested energy/k-point does not match TBTGF index')
        s = 0 if spin is None else int(spin)
        ret = []
        for ifile, f in enumerate(self.files):
            key = (s, ik)
            if self.last_key[ifile] != key:
                self.HS[ifile] = f.HkSk(k=int(ik), spin=s)
                self.last_key[ifile] = key
            H, S = self.HS[ifile]
            bulk = f.self_energy(int(ie), k=int(ik), spin=s)
            sigma = z * S - H - bulk
            ret.append((self.idx[ifile], sigma))
        return ret

    def close(self):
        for f in self.files:
            f._close_gf()


def _fingerprint_downfold(data, device_orbs):
    """Hash source downfolding blocks AND placement for cache compatibility."""
    if data is None:
        return None
    h = hashlib.sha256()
    for name in ("source_id", "basis", "H_idx", "C_idx"):
        h.update(str(data.get(name, "")).encode())
        h.update(b"\0")
    for a in (data["c_orbs"], data["h_orbs"], device_orbs,
              data["S_CH"], data["S_HC"], data["S_HH"],
              data[0]["H_CH"], data[0]["H_HC"], data[0]["H_HH"],
              data[1]["H_CH"], data[1]["H_HC"], data[1]["H_HH"]):
        _hash_array(h, np.asarray(a))
    return h.hexdigest()


def prepare_calculation_folder(folder, H_refs, method, Di, oi, Dj, oj, E, W,
                               nk, axes, trs, spin_independent_leads,
                               orbital_resolved, prefactor, left_orbs, right_orbs,
                               lead_eta, semi_L, semi_R, rank=0, results_name="Results",
                               downfold=False, hydrogen_i=None, hydrogen_j=None,
                               mapped_c_orbs_i=None, mapped_c_orbs_j=None):
    """Write Delta_i/j and run configuration once; reject incompatible reruns."""
    folder = Path(folder)
    results_folder = folder / results_name

    if rank != 0:
        return

    # Create required directories
    for sub in (folder / 'Deltas', folder / 'Self_Energies',
                results_folder, results_folder / 'tmp'):
        sub.mkdir(parents=True, exist_ok=True)

    conf = dict(method=method, nk=list(nk), axes=list(axes), trs=bool(trs),
                nref=len(H_refs), n_orbitals=int(H_refs[0].no),
                refs_sha256=[_fingerprint_hamiltonian(h) for h in H_refs],
                spin_independent_leads=bool(spin_independent_leads),
                orbital_resolved=bool(orbital_resolved),
                prefactor=[float(complex(prefactor).real), float(complex(prefactor).imag)],
                orbitals_i=np.asarray(oi).tolist(), orbitals_j=np.asarray(oj).tolist(),
                delta_i_sha256=_fingerprint_delta(Di, oi),
                delta_j_sha256=_fingerprint_delta(Dj, oj),
                energies=[[float(z.real), float(z.imag)] for z in E],
                weights=[[float(w.real), float(w.imag)] for w in W],
                left_orbs=None if left_orbs is None else np.asarray(left_orbs).tolist(),
                right_orbs=None if right_orbs is None else np.asarray(right_orbs).tolist(),
                semi_L=semi_L, semi_R=semi_R, lead_eta=float(lead_eta),
                downfold=bool(downfold),
                hydrogen_i_sha256=_fingerprint_downfold(
                    hydrogen_i, mapped_c_orbs_i),
                hydrogen_j_sha256=_fingerprint_downfold(
                    hydrogen_j, mapped_c_orbs_j))

    conf_path = results_folder / "config.json"
    contour = results_folder / "contour.npz"

    if conf_path.exists():
        _check_metadata(conf_path, conf)
    else:
        _write_json(conf_path, conf)

    if not contour.exists():
        _atomic_npz(contour, energies=E, contour_weights=W)


def save_downfold_data(data, filename, source_id, H_idx,C_idx):
    """
    Save hydrogen downfolding data independently of the device.
    source_id : identifier/path of the original DFT Hamiltonian
    H_idx     : hydrogen atom index in that Hamiltonian
    C_idx     : C atom index in that Hamiltonian
    """
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        filename,

        # Original Hamiltonian information
        source_id=str(source_id),
        H_idx=int(H_idx),
        C_idx=int(C_idx),
        basis=data["basis"],

        # Original, unpruned orbital indices
        c_orbs=data["c_orbs"],
        h_orbs=data["h_orbs"],

        # Overlap
        S_CH=data["S_CH"],
        S_HC=data["S_HC"],
        S_HH=data["S_HH"],

        # Spin up
        Hu_CH=data[0]["H_CH"],
        Hu_HC=data[0]["H_HC"],
        Hu_HH=data[0]["H_HH"],

        # Spin down
        Hd_CH=data[1]["H_CH"],
        Hd_HC=data[1]["H_HC"],
        Hd_HH=data[1]["H_HH"],
    )


def load_downfold_data(filename):
    """Load data compatible with hydrogen_self_energy()."""
    with np.load(filename, allow_pickle=False) as f:
        data = {
            "source_id": str(f["source_id"].item()),
            "H_idx": int(f["H_idx"].item()),
            "C_idx": int(f["C_idx"].item()),
            "basis": str(f["basis"].item()),

            "c_orbs": f["c_orbs"].copy(),
            "h_orbs": f["h_orbs"].copy(),

            "S_CH": f["S_CH"].copy(),
            "S_HC": f["S_HC"].copy(),
            "S_HH": f["S_HH"].copy(),

            0: {
                "H_CH": f["Hu_CH"].copy(),
                "H_HC": f["Hu_HC"].copy(),
                "H_HH": f["Hu_HH"].copy(),
            },

            1: {
                "H_CH": f["Hd_CH"].copy(),
                "H_HC": f["Hd_HC"].copy(),
                "H_HH": f["Hd_HH"].copy(),
            },
        }
    return data
