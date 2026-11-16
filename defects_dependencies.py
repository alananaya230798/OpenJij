"""Defect specifications for exchange calculations in GNRs.

A defect spec describes WHAT a defect is and WHERE it lives in its own DFT
file. Placement in a host (cell index, n_dist, device size) belongs to the
run, not to the defect.

Two defect types are supported:

    Adatom(label, file, index, species="H")
        `index` is the adatom atom index in the full (unpruned) DFT file.
        The bonded host atom is found automatically and used as anchor.

    Heteroatom(label, file, indices, Z=None, anchor=None)
        `indices` are the substituted atom indices in the full DFT file,
        `Z` their atomic numbers/symbols (used as a consistency check), and
        `anchor` the atom index (one of `indices`) used to place the defect.

Each spec produces the keyword arguments of the existing functions
`extract_single_Delta` and `prepare_spin_basis` (J_dependencies.py),
so the solver is unchanged:

    Delta, orb, _ = extract_single_Delta(
        H_host=H, H_polarized_file=spec.file,
        host_anchor=spec.host_anchor(H_0.geometry, cell=N_imp + N_buffer),
        basis=basis, tol=tol, **spec.delta_kwargs("rkky"))

Run `python defects.py siesta.fdf` to list the adatom and heteroatom
candidates (with indices) of a DFT file.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import sisl

# Atoms retained by prune_ham / prepare_spin_basis for basis != 'full'.
HEAVY_Z = (5, 6, 7, 8, 9)
# Host is pristine: the defect atoms are not in the host, adatoms are removed.
PRISTINE_METHODS = ("rkky",)
# Host contains the defects: adatom orbitals are kept explicitly.
EXPLICIT_METHODS = ("rkky_nm", "liechtenstein")
ADATOM_ORBITALS = ("s", "p", "pz", "s+pz", "all")

_PT = sisl.PeriodicTable()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _as_Z(value) -> int:
    """Atomic number from an int or an element symbol."""
    if isinstance(value, (int, np.integer)):
        return int(value)
    return int(_PT.Z(str(value).strip()))


def _symbol(Z: int) -> str:
    return str(sisl.Atom(int(Z)).symbol)


@lru_cache(maxsize=32)
def _read_geometry_cached(path: str, mtime_ns: int):
    return sisl.get_sile(path).read_geometry()


def read_geometry(file) -> "sisl.Geometry":
    """Read (and cache) a geometry; the cache is invalidated if the file changes."""
    path = Path(file).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"DFT file not found: {path}")
    return _read_geometry_cached(str(path), path.stat().st_mtime_ns)


def _check_method(method: str) -> str:
    method = method.lower().strip()
    if method not in PRISTINE_METHODS + EXPLICIT_METHODS:
        raise ValueError(f"Unknown method '{method}'. "
                         f"Use one of {PRISTINE_METHODS + EXPLICIT_METHODS}")
    return method


def _check_index(geometry, index: int, label: str) -> None:
    if not 0 <= index < geometry.na:
        raise IndexError(f"{label}: atom index {index} outside 0..{geometry.na - 1}")


def _transport_direction(geometry, axis: int) -> np.ndarray:
    a = np.asarray(geometry.cell[axis], dtype=float)
    return a / np.linalg.norm(a)


def bonded_host_atom(geometry, index: int, bond_cutoff: float = 1.3,
                     label: str = "") -> int:
    """Unique heavy atom (B, C, N, O, F) bonded to `index`, as a unit-cell index."""
    idx, rij = geometry.close(index, R=bond_cutoff, ret_rij=True)
    idx = np.asarray(geometry.asc2uc(idx), dtype=int)
    hosts = sorted({int(j) for j, r in zip(idx, rij)
                    if int(j) != index and geometry.atoms.Z[int(j)] in HEAVY_Z})
    if len(hosts) != 1:
        raise ValueError(
            f"{label}: atom {index} must bind exactly one heavy atom within "
            f"{bond_cutoff} Å, found {hosts}. Adjust bond_cutoff.")
    return hosts[0]


# ---------------------------------------------------------------------------
# Resolved defect: everything read from the DFT file
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class ResolvedDefect:
    label: str
    kind: str
    file: Path
    anchor: int                     # atom index (DFT file) used for placement
    defect_atoms: Tuple[int, ...]   # adatom, or substituted atoms
    Z: Tuple[int, ...]
    anchor_xyz: np.ndarray
    center: np.ndarray              # defect center (DFT file coordinates)

    @property
    def center_offset(self) -> np.ndarray:
        """Vector from anchor to center; used for defect-defect distances."""
        return self.center - self.anchor_xyz

    def __str__(self) -> str:
        atoms = ", ".join(f"{_symbol(z)}{a}" for a, z in zip(self.defect_atoms, self.Z))
        return (f"{self.label} [{self.kind}] atoms: {atoms}; anchor: {self.anchor}; "
                f"file: {self.file}")


# ---------------------------------------------------------------------------
# Shared behaviour of all defect specs
# ---------------------------------------------------------------------------
class _DefectBase:
    """Methods shared by Adatom and Heteroatom (no dataclass fields here)."""

    kind = "defect"

    def resolve(self, geometry=None) -> ResolvedDefect:  # pragma: no cover
        raise NotImplementedError

    def adatoms(self) -> Dict[int, str]:
        """{adatom index: retained orbitals} in this spec's file."""
        return {}

    # Host placement --------------------------------------------------------
    def locate_host_site(self, host_cell_geometry, axis: int = 0,
                         tol: float = 0.3) -> int:
        """Atom index in one PRUNED pristine cell matching the defect anchor.

        Valid when the defect and pristine files share the same frame up to
        integer translations along the transport axis (same ribbon width and
        vacuum placement). Only in-plane (x, y) positions are compared, so a
        puckered adatom carbon is still matched. Raises otherwise; then set
        host_site explicitly.
        """
        if self.host_site is not None:
            return int(self.host_site)
        r = self.resolve()
        cell = np.asarray(host_cell_geometry.cell, dtype=float)
        frac = r.anchor_xyz @ np.linalg.inv(cell)
        frac[axis] -= np.floor(frac[axis])
        target = frac @ cell
        xyz = np.asarray(host_cell_geometry.xyz, dtype=float)
        best = []
        for ia in range(host_cell_geometry.na):
            d = min(np.linalg.norm((xyz[ia] + s * cell[axis] - target)[:2])
                    for s in (-1, 0, 1))
            best.append((d, ia))
        best.sort()
        d0, ia0 = best[0]
        if d0 > tol:
            raise ValueError(
                f"{self.label}: no pristine site within {tol} Å of the anchor "
                f"(closest atom {ia0} at {d0:.3f} Å). The defect and pristine "
                "frames are probably shifted; set host_site explicitly.")
        if len(best) > 1 and best[1][0] <= tol:
            raise ValueError(f"{self.label}: ambiguous pristine site "
                             f"({best[0]}, {best[1]}); lower tol or set host_site.")
        return int(ia0)

    def host_anchor(self, host_cell_geometry, cell: int, axis: int = 0,
                    tol: float = 0.3) -> int:
        """Anchor index in a host built as H_0.tile(N, axis), placed in `cell`."""
        site = self.locate_host_site(host_cell_geometry, axis=axis, tol=tol)
        return int(cell) * host_cell_geometry.na + site

    # Arguments for extract_single_Delta -----------------------------------
    def delta_kwargs(self, method: str, companions: Iterable["_DefectBase"] = ()) -> dict:
        """Keyword arguments for extract_single_Delta.

        companions: other specs living in the SAME DFT file (two-defect
        device). Their adatoms are removed/kept together with this one.
        """
        method = _check_method(method)
        group = _same_file_group(self, companions)
        adatoms = keep_orbitals_for(group)
        kwargs = dict(defect_anchor=self.resolve().anchor)
        if method in PRISTINE_METHODS:
            # Pristine host: adatoms do not exist there, and a substituted
            # atom sits on a host carbon (species mismatch is expected).
            kwargs.update(adatom_atoms=tuple(adatoms), keep_orbitals={},
                          allow_species_mismatch=self.kind == "heteroatom")
        else:
            # Host contains the defects: keep adatom orbitals explicitly.
            kwargs.update(adatom_atoms=(), keep_orbitals=adatoms,
                          allow_species_mismatch=False)
        return kwargs


def _same_file_group(spec, companions) -> List[_DefectBase]:
    group = [spec] + [c for c in companions if c is not spec]
    files = {Path(s.file).expanduser().resolve() for s in group}
    if len(files) != 1:
        raise ValueError("Companion defects must come from the same DFT file: "
                         + ", ".join(str(f) for f in files))
    return group


def keep_orbitals_for(specs: Iterable[_DefectBase]) -> Dict[int, str]:
    """keep_orbitals dict for prepare_spin_basis from specs of one DFT file."""
    keep: Dict[int, str] = {}
    for spec in specs:
        for ia, orbs in spec.adatoms().items():
            if ia in keep and keep[ia] != orbs:
                raise ValueError(f"Adatom {ia} requested with orbitals "
                                 f"'{keep[ia]}' and '{orbs}'")
            keep[ia] = orbs
    return keep


# ---------------------------------------------------------------------------
# Adatom
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class Adatom(_DefectBase):
    """Adatom bonded to one heavy atom of the ribbon.

    label     : name used in folders/results.
    file      : full DFT file (fdf/TSHS) where `index` is defined.
    index     : adatom index in the FULL, unpruned DFT file.
    species   : element symbol or Z (checked against the file).
    host_site : optional pristine-cell site of the bonded atom (RKKY);
                found automatically if None.
    orbitals  : adatom orbitals kept for rkky_nm / liechtenstein.
    """
    label: str
    file: Union[str, Path]
    index: int
    species: Union[str, int] = "H"
    host_site: Optional[int] = None
    orbitals: str = "all"
    bond_cutoff: float = 1.3

    kind = "adatom"

    def __post_init__(self):
        object.__setattr__(self, "file", Path(self.file).expanduser())
        object.__setattr__(self, "index", int(self.index))
        object.__setattr__(self, "species", _as_Z(self.species))
        orbitals = str(self.orbitals).lower().replace(" ", "")
        if orbitals not in ADATOM_ORBITALS:
            raise ValueError(f"{self.label}: orbitals must be one of {ADATOM_ORBITALS}")
        object.__setattr__(self, "orbitals", orbitals)

    def resolve(self, geometry=None) -> ResolvedDefect:
        g = read_geometry(self.file) if geometry is None else geometry
        _check_index(g, self.index, self.label)
        Z = int(g.atoms.Z[self.index])
        if Z != self.species:
            raise ValueError(f"{self.label}: atom {self.index} is {_symbol(Z)}, "
                             f"expected {_symbol(self.species)}")
        host = bonded_host_atom(g, self.index, self.bond_cutoff, self.label)
        xyz = np.asarray(g.xyz[host], dtype=float)
        return ResolvedDefect(self.label, self.kind, self.file, host,
                              (self.index,), (Z,), xyz, xyz.copy())

    def adatoms(self) -> Dict[int, str]:
        return {self.index: self.orbitals}


# ---------------------------------------------------------------------------
# Heteroatom
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class Heteroatom(_DefectBase):
    """Substitutional defect made of one or several atoms (e.g. 2B).

    label     : name used in folders/results.
    file      : full DFT file where `indices` are defined.
    indices   : substituted atom index or list of indices (FULL DFT file).
    Z         : atomic number/symbol, one value or one per index (checked).
    anchor    : atom index (one of `indices`) used for placement;
                default indices[0].
    host_site : optional pristine-cell site of the anchor (RKKY);
                found automatically if None.
    """
    label: str
    file: Union[str, Path]
    indices: Union[int, Sequence[int]]
    Z: Optional[Union[int, str, Sequence[Union[int, str]]]] = None
    anchor: Optional[int] = None
    host_site: Optional[int] = None

    kind = "heteroatom"

    def __post_init__(self):
        object.__setattr__(self, "file", Path(self.file).expanduser())
        indices = tuple(int(i) for i in np.atleast_1d(self.indices))
        if not indices or len(set(indices)) != len(indices):
            raise ValueError(f"{self.label}: indices must be unique and nonempty")
        object.__setattr__(self, "indices", indices)
        if self.Z is not None:
            Z = [_as_Z(z) for z in np.atleast_1d(np.asarray(self.Z, dtype=object))]
            if len(Z) == 1:
                Z = Z * len(indices)
            if len(Z) != len(indices):
                raise ValueError(f"{self.label}: {len(Z)} Z values for {len(indices)} indices")
            object.__setattr__(self, "Z", tuple(Z))
        anchor = indices[0] if self.anchor is None else int(self.anchor)
        if anchor not in indices:
            raise ValueError(f"{self.label}: anchor {anchor} is not in indices {indices}")
        object.__setattr__(self, "anchor", anchor)

    def resolve(self, geometry=None) -> ResolvedDefect:
        g = read_geometry(self.file) if geometry is None else geometry
        actual = []
        for k, ia in enumerate(self.indices):
            _check_index(g, ia, self.label)
            z = int(g.atoms.Z[ia])
            if self.Z is not None and z != self.Z[k]:
                raise ValueError(f"{self.label}: atom {ia} is {_symbol(z)}, "
                                 f"expected {_symbol(self.Z[k])}")
            if z == 6:
                raise ValueError(f"{self.label}: atom {ia} is carbon, not a heteroatom")
            if z not in HEAVY_Z:
                raise ValueError(f"{self.label}: {_symbol(z)} is removed by the "
                                 f"basis pruning (kept Z: {HEAVY_Z})")
            actual.append(z)
        xyz = np.asarray(g.xyz[list(self.indices)], dtype=float)
        return ResolvedDefect(self.label, self.kind, self.file, self.anchor,
                              self.indices, tuple(actual),
                              np.asarray(g.xyz[self.anchor], dtype=float),
                              xyz.mean(axis=0))


DefectSpec = Union[Adatom, Heteroatom]


# ---------------------------------------------------------------------------
# Automatic detection (two-defect devices, first look at a new file)
# ---------------------------------------------------------------------------
def detect_adatoms(file, species="H", z_thresh=0.3, bond_cutoff=1.3, axis=0,
                   expect: Optional[int] = None, prefix: Optional[str] = None,
                   orbitals="all") -> List[Adatom]:
    """Out-of-plane adatoms of a DFT file, sorted along the transport axis.

    In-plane atoms of the same species (edge passivation) are excluded by
    requiring |z_adatom - z_host| > z_thresh.
    """
    g = read_geometry(file)
    Zs = _as_Z(species)
    found = []
    for ia in np.flatnonzero(g.atoms.Z == Zs):
        ia = int(ia)
        try:
            host = bonded_host_atom(g, ia, bond_cutoff, f"atom {ia}")
        except ValueError:
            continue
        if abs(g.xyz[ia, 2] - g.xyz[host, 2]) > z_thresh:
            found.append((ia, host))
    direction = _transport_direction(g, axis)
    found.sort(key=lambda p: float(g.xyz[p[1]] @ direction))
    if expect is not None and len(found) != expect:
        raise ValueError(f"Expected {expect} {_symbol(Zs)} adatoms in {file}, "
                         f"found {len(found)}: {[ia for ia, _ in found]}")
    prefix = prefix if prefix is not None else f"{_symbol(Zs)}"
    return [Adatom(f"{prefix}{k}", file, ia, species=Zs, orbitals=orbitals,
                   bond_cutoff=bond_cutoff) for k, (ia, _) in enumerate(found)]


def detect_heteroatoms(file, Z=5, group_cutoff=3.0, axis=0,
                       expect: Optional[int] = None,
                       prefix: Optional[str] = None) -> List[Heteroatom]:
    """Substituted atoms grouped into defects, sorted along the transport axis.

    Atoms with the given Z closer than `group_cutoff` (Å, periodic images
    included) form one defect, e.g. a boron pair -> one '2B' Heteroatom.
    The anchor of each group is its first atom along the transport axis.
    """
    g = read_geometry(file)
    Zlist = [_as_Z(z) for z in np.atleast_1d(np.asarray(Z, dtype=object))]
    atoms = [int(ia) for ia in np.flatnonzero(np.isin(g.atoms.Z, Zlist))]
    parent = {ia: ia for ia in atoms}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for ia in atoms:
        near = np.asarray(g.asc2uc(g.close(ia, R=group_cutoff)), dtype=int)
        for ja in near:
            if int(ja) in parent:
                parent[find(int(ja))] = find(ia)
    groups: Dict[int, List[int]] = {}
    for ia in atoms:
        groups.setdefault(find(ia), []).append(ia)

    direction = _transport_direction(g, axis)
    proj = lambda ia: float(g.xyz[ia] @ direction)
    ordered = sorted((sorted(grp, key=proj) for grp in groups.values()),
                     key=lambda grp: proj(grp[0]))
    if expect is not None and len(ordered) != expect:
        raise ValueError(f"Expected {expect} heteroatom defects in {file}, "
                         f"found {len(ordered)}: {ordered}")
    specs = []
    for k, grp in enumerate(ordered):
        tag = "".join(f"{n if n > 1 else ''}{_symbol(z)}" for z, n in
                      zip(*np.unique(g.atoms.Z[grp], return_counts=True)))
        label = f"{prefix if prefix is not None else tag}{k}"
        specs.append(Heteroatom(label, file, grp, Z=[int(g.atoms.Z[i]) for i in grp]))
    return specs


# ---------------------------------------------------------------------------
# Registry and legacy conversion
# ---------------------------------------------------------------------------
_TYPES = {"adatom": Adatom, "heteroatom": Heteroatom}


def load_defects(config: Dict[str, dict], base_dir=None) -> Dict[str, DefectSpec]:
    """Build {label: spec} from a dict (e.g. loaded from YAML/JSON).

    Example:
        {"H0":   {"type": "adatom", "file": "H/siesta.fdf", "index": 112},
         "2B-1": {"type": "heteroatom", "file": "2B/siesta.fdf",
                  "indices": [40, 51], "Z": "B", "anchor": 40, "host_site": 15}}
    Relative paths are resolved against base_dir.
    """
    specs = {}
    for label, entry in config.items():
        entry = dict(entry)
        kind = str(entry.pop("type", "")).lower()
        if kind not in _TYPES:
            raise ValueError(f"{label}: 'type' must be one of {tuple(_TYPES)}")
        path = Path(entry.pop("file")).expanduser()
        if base_dir is not None and not path.is_absolute():
            path = Path(base_dir) / path
        specs[label] = _TYPES[kind](label=label, file=path, **entry)
    return specs


def from_legacy_heteroatom(label: str, entry: dict, Z=5) -> Heteroatom:
    """Convert an old defect_dict entry {'idx', 'file', 'boron_ordinal'}.

    All atoms with atomic number Z in the file form the defect, the anchor is
    the `boron_ordinal`-th of them (old behaviour) and 'idx' is the host site.
    """
    Zn = _as_Z(Z)
    g = read_geometry(entry["file"])
    indices = [int(i) for i in np.flatnonzero(g.atoms.Z == Zn)]
    if not indices:
        raise ValueError(f"{label}: no {_symbol(Zn)} atoms in {entry['file']}")
    ordinal = int(entry.get("boron_ordinal", 0))
    if not 0 <= ordinal < len(indices):
        raise ValueError(f"{label}: boron_ordinal={ordinal} but {entry['file']} "
                         f"has {len(indices)} {_symbol(Zn)} atom(s): {indices}")
    anchor = indices[ordinal]
    return Heteroatom(label, entry["file"], indices, Z=Zn, anchor=anchor,
                      host_site=entry.get("idx"))


def auto_spec(label: str, file, kind: str = "adatom", host_site: Optional[int] = None,
              **detect_kwargs) -> DefectSpec:
    """Spec of the single defect in `file`, detected automatically (expect=1)."""
    from dataclasses import replace
    if kind == "adatom":
        spec = detect_adatoms(file, expect=1, **detect_kwargs)[0]
    elif kind == "heteroatom":
        spec = detect_heteroatoms(file, expect=1, **detect_kwargs)[0]
    else:
        raise ValueError("kind must be 'adatom' or 'heteroatom'")
    return replace(spec, label=label, host_site=host_site)


# ---------------------------------------------------------------------------
# Small utilities used by the drivers
# ---------------------------------------------------------------------------
def pruned_index(original_atoms, atom: int) -> int:
    """Index after pruning of an atom given in the full DFT file."""
    pos = np.flatnonzero(np.asarray(original_atoms) == int(atom))
    if len(pos) != 1:
        raise ValueError(f"Atom {atom} was removed by the basis pruning")
    return int(pos[0])


def defect_distance(host_geometry, host_anchors, resolved) -> float:
    """Center-to-center distance of two placed defects (Å).

    host_anchors : host atom indices where the two anchors are placed.
    resolved     : the two ResolvedDefect objects.
    """
    ri = host_geometry.xyz[host_anchors[0]] + resolved[0].center_offset
    rj = host_geometry.xyz[host_anchors[1]] + resolved[1].center_offset
    return float(np.linalg.norm(rj - ri))


def describe(file, axis=0, z_thresh=0.3) -> None:
    """Print adatom and heteroatom candidates of a DFT file."""
    g = read_geometry(file)
    print(f"{file}: {g.na} atoms, species "
          + ", ".join(f"{_symbol(z)}x{n}" for z, n in
                      zip(*np.unique(g.atoms.Z, return_counts=True))))
    for spec in detect_adatoms(file, z_thresh=z_thresh, axis=axis):
        print(" ", spec.resolve())
    others = [int(z) for z in np.unique(g.atoms.Z) if z in HEAVY_Z and z != 6]
    if others:
        for spec in detect_heteroatoms(file, Z=others, axis=axis):
            print(" ", spec.resolve())


if __name__ == "__main__":
    for f in sys.argv[1:]:
        describe(f)
