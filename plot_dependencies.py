# -*- coding: utf-8 -*-
"""Visualizations of exchange calculations (any method).

    from Exchange_Workflow import run_exchange
    from plot_dependencies import *

    run = run_exchange(method="liechtenstein", ...)
    plot_all(run)

From a results folder (also results copied from the cluster):
    plot_orbital_contributions(results_dir)   J from pairs of orbital classes
    plot_exchange_kernel(results_dir)         weighted kernel along the contour
    plot_atomic_contributions(results_dir)    map of the atom-resolved contributions
    atomic_contributions_table(results_dir)   the same as a table

From the ExchangeRun returned by run_exchange (they need the device or system):
    plot_setup(run)       device, vertex regions (+ Mulliken moments), electrodes, anchors
    plot_vertices(run)    spin splitting of the two vertices per atom, zoomed on the defects

The geometry plots also take a built Device / DysonSystem, or are called from it:
    dev = PristineTiled(...).build("rkky");  dev.plot_setup();  dev.plot_vertices()
    builder.plot_setup()  /  builder.plot_vertices()      (last build of the builder)
    plot_dos(run)         DOS of the reference of the method, with a comparison
    plot_all(run)         everything above

Only numpy, matplotlib and scipy are needed here (plus plot_atoms and
get_Delta_orbital_atoms from J_dependencies). Optional libraries for new plots
(sisl.viz, plotly, ase, ...) should be loaded with optional_import() inside the
function that needs them, so this module keeps loading without them.
"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import J_dependencies as JD
from Device_Builder import compute_dos
from Exchange_Workflow import ExchangeRun, load_results

__all__ = ["plot_setup", "plot_vertices", "plot_regions", "region_atoms", "draw_regions",
           "draw_electrodes", "draw_anchors",
           "plot_orbital_contributions", "plot_exchange_kernel",
           "plot_atomic_contributions", "atomic_contributions_table", "plot_dos", "plot_all",
           "optional_import"]

REGION_COLORS = ("red", "blue")       # vertex region of defect i / defect j


def _view(src, method=None):
    """Everything the geometry plots need, from an ExchangeRun, a Device or a DysonSystem.

    method: for a Device / DysonSystem, the method whose vertices are shown (default:
    the method it was built for; for a DysonSystem, 'liechtenstein_sup' vertices
    follow info['vertices']).
    """
    from types import SimpleNamespace
    from Device_Builder import canonical_method
    if isinstance(src, ExchangeRun):
        obj, cfg = src.obj, src.config
        Ds, orbs = src.vertices()
        meth, builder = src.method, cfg["builder"]
        region = cfg.get("region", "")
        if region == "disjoint":
            region += " (" + (cfg.get("region_assign") or
                              ("nearest" if builder == "DFTDevice" else "dominant")) + ")"
    else:
        obj = src
        info = getattr(obj, "info", {})
        if obj.__class__.__name__ == "DysonSystem":
            meth = canonical_method(method) if method else "liechtenstein_sup"
            total = meth == "liechtenstein_sup" and info.get("vertices", "own") == "region"
            Ds = list(obj.vertex_total if total else obj.vertex_own)
            orbs = list(obj.orbs)
            builder = "SuperpositionDyson"
        else:
            meth = canonical_method(method) if method else info.get("built_method", obj.methods[0])
            Ds, orbs = obj.vertices(meth) if method else (obj.Deltas, obj.orbs)
            builder = obj.kind
        region = info.get("region_text", "")
    return SimpleNamespace(obj=obj, H=obj.H, g=obj.H.geometry, Ds=Ds, orbs=orbs, method=meth,
                           builder=builder, region=region, anchors=list(obj.anchors),
                           left=obj.left_orbs, right=obj.right_orbs,
                           info=getattr(obj, "info", {}), distance=obj.anchor_distance)


def region_atoms(src, method=None):
    """Device atoms of the two vertex regions (static part + folded carbons)."""
    v = src if hasattr(src, "Ds") else _view(src, method)
    return [np.unique(v.g.o2a(np.asarray(o, dtype=int))) for o in v.orbs]


# Marker areas of plot_atoms (points^2), so overlays have the same size as the atoms
_ATOM_SIZE = {1: 10.0, 5: 70.0, 6: 50.0, 7: 70.0}
ELECTRODE_COLOR = "tab:green"


def _atom_sizes(geometry, atoms):
    Z = np.asarray(geometry.atoms.Z)[np.asarray(atoms, dtype=int)]
    return np.array([_ATOM_SIZE.get(int(z), 50.0) for z in Z])


def draw_regions(ax, geometry, regions, colors=REGION_COLORS, lw: float = 1.0,
                 alpha: float = 1.0, bond: float = 1.77, labels=("region i", "region j"),
                 style: str = "rings"):
    """Mark each vertex region in red (i) / blue (j).

    style 'rings' (default): an open ring (facecolor none) on every atom of the region,
          with the same size as the atom drawn by plot_atoms, so the atom colour
          underneath (Delta, or the plain atom) stays visible.
    style 'outline': thick lines along the perimeter of the region.
    style 'bonds'  : every bond inside the region as a thick line.
    Returns the legend handles.
    """
    from scipy.spatial import cKDTree
    from matplotlib.lines import Line2D
    if style not in ("rings", "outline", "bonds"):
        raise ValueError("style must be 'rings', 'outline' or 'bonds'")
    xyz = np.asarray(geometry.xyz)
    heavy = geometry.atoms.Z > 1
    handles = []
    if style != "rings":
        pairs = np.array(sorted(cKDTree(xyz).query_pairs(bond)), dtype=int).reshape(-1, 2)
        if len(pairs):
            pairs = pairs[heavy[pairs[:, 0]] & heavy[pairs[:, 1]]]
    for k, (atoms, col) in enumerate(zip(regions, colors)):
        atoms = np.asarray(atoms, dtype=int)
        if style == "rings":
            ax.scatter(xyz[atoms, 0], xyz[atoms, 1], s=_atom_sizes(geometry, atoms),
                       facecolors="none", edgecolors=col, linewidths=lw, alpha=alpha, zorder=6)
            handles.append(Line2D([0], [0], marker="o", ls="none", markerfacecolor="none",
                                  markeredgecolor=col, markeredgewidth=lw, markersize=7,
                                  label=f"{labels[k]} ({len(atoms)} atoms)"))
            continue
        inside = np.zeros(len(xyz), dtype=bool)
        inside[atoms] = True
        sel = pairs[inside[pairs[:, 0]] & inside[pairs[:, 1]]] if len(pairs) else pairs
        n_in = np.bincount(sel.ravel(), minlength=len(xyz)) if len(sel) else np.zeros(len(xyz), int)
        draw = sel
        if style == "outline":
            border = inside & (n_in < 3)
            draw = sel[border[sel[:, 0]] & border[sel[:, 1]]] if len(sel) else sel
        for p, q in draw:
            ax.plot(xyz[[p, q], 0], xyz[[p, q], 1], color=col, lw=2 * lw, alpha=alpha,
                    solid_capstyle="round", zorder=6)
        lone = np.flatnonzero(inside & heavy & (n_in == 0))
        if lone.size:
            ax.scatter(xyz[lone, 0], xyz[lone, 1], s=_atom_sizes(geometry, lone),
                       facecolors="none", edgecolors=col, linewidths=lw, zorder=6)
        handles.append(Line2D([0], [0], color=col, lw=2 * lw, alpha=alpha,
                              label=f"{labels[k]} ({int(inside.sum())} atoms)"))
    return handles


def draw_electrodes(ax, src, color: str = ELECTRODE_COLOR, lw: float = 1.0):
    """Open rings (atom size) on the left and right electrode cells of the device."""
    from matplotlib.lines import Line2D
    v = src if hasattr(src, "Ds") else _view(src)
    g = v.g
    at = np.unique(np.concatenate([g.o2a(np.asarray(o, dtype=int)) for o in (v.left, v.right)]))
    ax.scatter(g.xyz[at, 0], g.xyz[at, 1], s=_atom_sizes(g, at), facecolors="none",
               edgecolors=color, linewidths=lw, zorder=6)
    return [Line2D([0], [0], marker="o", ls="none", markerfacecolor="none", markeredgecolor=color,
                   markeredgewidth=lw, markersize=7, label="electrodes")]


def draw_anchors(ax, src, colors=REGION_COLORS, s: float = 220.0, lw: float = 1.8):
    """Dashed circles on the two defect anchors."""
    from matplotlib.lines import Line2D
    v = src if hasattr(src, "Ds") else _view(src)
    g = v.g
    handles = []
    for i, col in enumerate(colors):
        a = int(v.anchors[i])
        ax.scatter([g.xyz[a, 0]], [g.xyz[a, 1]], s=s, facecolors="none", edgecolors=col,
                   linewidths=lw, linestyles="--", zorder=7)
        handles.append(Line2D([0], [0], marker="o", ls="none", markerfacecolor="none",
                              markeredgecolor=col, markersize=10, markeredgewidth=lw,
                              label=f"defect {'ij'[i]} anchor"))
    return handles


def _legend_below(ax, handles):
    # below the x label, at a fixed distance in font units (independent of the aspect)
    ax.legend(handles=handles, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 0.0),
              borderaxespad=5.5, ncol=min(len(handles), 5), frameon=False)


def optional_import(module: str, purpose: str = "this plot"):
    """Import an optional plotting library, with a clear message if it is missing."""
    try:
        return importlib.import_module(module)
    except ImportError as err:
        raise ImportError(f"{purpose} needs the optional package '{module}' "
                          f"(pip install {module.split('.')[0]})") from err


def _dir(run_or_dir) -> Path:
    return run_or_dir.results_dir if isinstance(run_or_dir, ExchangeRun) else Path(run_or_dir)


# ===========================================================================
# Plots from the results folder (any method, also results from the cluster)
# ===========================================================================
def _orbital_classes(geo, side):
    """Masks of the vertex orbitals: carbon pz, adatom H, d (only nonempty classes)."""
    names = geo[f"orbital_names_{side}"]
    Z = geo["Z"][geo[f"atoms_{side}"]]
    h = Z == 1
    classes = {"C-pz": np.array(["pz" in n for n in names]) & ~h,
               "H": h,
               "d": np.array(["d" in n for n in names]) & ~h}
    return {k: v for k, v in classes.items() if v.any()}


def plot_orbital_contributions(run_or_dir, ax=None):
    """Bar chart of J from pairs of orbital classes (rows i, columns j)."""
    res, geo, cfg = load_results(_dir(run_or_dir))
    J_orb = res["J_orb_meV"]
    ci, cj = _orbital_classes(geo, "i"), _orbital_classes(geo, "j")
    labels, values = [], []
    for a, mi in ci.items():
        for b, mj in cj.items():
            labels.append(f"{a}-{b}")
            values.append(J_orb[np.ix_(mi, mj)].sum())
    J_total = float(res["J_meV"])
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(J_total, color="red", ls="--", label=rf"Total $J$ = {J_total:.4f} meV")
    ax.set_ylabel("Exchange contribution [meV]")
    ax.set_title(f"{str(res['method'])}: orbital-class contributions")
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=30)
    plt.tight_layout()
    return dict(zip(labels, values))


def plot_exchange_kernel(run_or_dir):
    """Weighted kernel along the contour: total, orbital classes, and the contour."""
    d = _dir(run_or_dir)
    res, geo, cfg = load_results(d)
    z = res["energies"]
    K = res["weighted_kernel_kE"][0]
    fig, ax = plt.subplots(3, 1, figsize=(6, 6), sharex=True)
    ax[0].scatter(z.real, 1000 * K.imag, s=15, c=z.imag, label="Im(K)")
    ax[0].scatter(z.real, 1000 * K.real, s=15, c=z.imag, marker="x", label="Re(K)")
    ax[0].set_ylabel("Total weighted [meV]")
    ax[0].legend()
    korb = d / "weighted_orbital_kernel.npy"
    if "weighted_class_kernel_kE" in res.files:          # per-energy kernel per class pair
        Kc = res["weighted_class_kernel_kE"][0]
        names_i, names_j = res["class_names_i"], res["class_names_j"]
        for a, na in enumerate(names_i):
            for b, nb in enumerate(names_j):
                if np.abs(Kc[:, a, b]).max() > 0:
                    ax[1].scatter(z.real, 1000 * Kc[:, a, b].imag, s=10, label=f"{na}-{nb}")
        ax[1].legend(fontsize=7)
    elif korb.is_file():                                 # full orbital kernel (large)
        K_orb = np.load(korb, mmap_mode="r")[0]
        ci, cj = _orbital_classes(geo, "i"), _orbital_classes(geo, "j")
        for a, mi in ci.items():
            for b, mj in cj.items():
                Kab = np.asarray(K_orb[:, mi][:, :, mj]).sum(axis=(1, 2))
                if np.abs(Kab).max() > 0:
                    ax[1].scatter(z.real, 1000 * Kab.imag, s=10, label=f"{a}-{b}")
        ax[1].legend(fontsize=7)
    ax[1].set_ylabel("Orbital weighted K [meV]")
    sc = ax[2].scatter(z.real, z.imag, c=z.imag, cmap="viridis", s=15)
    ax[2].set_xlabel("Re(E) [eV]")
    ax[2].set_ylabel("Im(E) [eV]")
    for a in ax:
        plt.colorbar(sc, ax=a, label="Im(E) [eV]")
    plt.tight_layout()
    return fig


def _atomic(res, geo):
    J_orb = res["J_orb_meV"]
    xyz = geo["xyz"]
    ai, aj = geo["atoms_i"].astype(int), geo["atoms_j"].astype(int)
    Ji, Jj = np.zeros(len(xyz)), np.zeros(len(xyz))
    np.add.at(Ji, ai, J_orb.sum(axis=1))
    np.add.at(Jj, aj, J_orb.sum(axis=0))
    selected = np.union1d(ai, aj)
    return Ji, Jj, 0.5 * (Ji + Jj), selected


def atomic_contributions_table(run_or_dir) -> None:
    """Print the atom-resolved contributions J_i, J_j and their symmetric average."""
    res, geo, cfg = load_results(_dir(run_or_dir))
    Ji, Jj, Ja, selected = _atomic(res, geo)
    xyz, Z = geo["xyz"], geo["Z"]
    print(f"{'Atom':>6} {'Z':>4} {'x [Å]':>10} {'y [Å]':>10} {'J_i [meV]':>14} {'J_j [meV]':>14} {'J_a [meV]':>14}")
    print("-" * 78)
    for a in selected:
        print(f"{a:6d} {Z[a]:4d} {xyz[a, 0]:10.3f} {xyz[a, 1]:10.3f} {Ji[a]:14.6f} {Jj[a]:14.6f} {Ja[a]:14.6f}")
    print("-" * 78)
    print(f"Total J = {float(res['J_meV']):.6f} meV")


def plot_atomic_contributions(run_or_dir, label_top: int = 5, cmap: str = "coolwarm"):
    """Map of the atom-resolved contributions (top view and side view)."""
    from scipy.spatial import cKDTree
    from matplotlib.collections import LineCollection
    res, geo, cfg = load_results(_dir(run_or_dir))
    Ji, Jj, Ja, selected = _atomic(res, geo)
    xyz = geo["xyz"]
    vals = Ja[selected]
    lim = np.max(np.abs(vals)) if np.any(vals) else 1.0
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 5), sharex=True,
                                  gridspec_kw=dict(height_ratios=(3, 1)))
    pairs = np.array(list(cKDTree(xyz).query_pairs(r=1.78)), dtype=int).reshape(-1, 2)
    if len(pairs):
        ax.add_collection(LineCollection(xyz[pairs, :2], colors="0.78", linewidths=0.8, zorder=1))
    ax.scatter(xyz[:, 0], xyz[:, 1], s=18, facecolor="0.82", edgecolor="0.6", linewidth=0.2, zorder=2)
    sizes = 45 + 140 * (np.abs(vals) / lim) ** 0.6
    sc = ax.scatter(xyz[selected, 0], xyz[selected, 1], c=vals, s=sizes, cmap=cmap,
                    vmin=-lim, vmax=lim, edgecolors="black", linewidths=0.35, zorder=3)
    defects = geo["defect_atoms"]
    ax.scatter(xyz[defects, 0], xyz[defects, 1], s=80, c="red", marker="X",
               edgecolors="black", linewidths=0.8, zorder=5, label="anchors")
    adatoms = geo["adatom_atoms"] if "adatom_atoms" in geo.files else []
    if len(adatoms):
        ax.scatter(xyz[adatoms, 0], xyz[adatoms, 1], s=60, c="cyan", edgecolors="black",
                   linewidths=0.8, zorder=6, label="adatoms")
        ax2.scatter(xyz[adatoms, 0], xyz[adatoms, 2], s=60, c="cyan", edgecolors="black", zorder=6)
    for a in selected[np.argsort(np.abs(vals))[::-1][:label_top]]:
        ax.annotate(f"{Ja[a]:.3f}", xyz[a, :2], fontsize=7, xytext=(3, 3), textcoords="offset points")
    plt.colorbar(sc, ax=ax, label=r"$J_a$ [meV]")
    ax.set_aspect("equal")
    ax.set_ylabel("y [Å]")
    ax.set_title(f"{str(res['method'])}: atomic contributions, J = {float(res['J_meV']):.4f} meV")
    ax.legend(fontsize=7, loc="upper right")
    ax2.scatter(xyz[:, 0], xyz[:, 2], s=10, c="0.7")
    ax2.scatter(xyz[selected, 0], xyz[selected, 2], c=vals, s=sizes / 3, cmap=cmap, vmin=-lim, vmax=lim)
    ax2.set_xlabel("x [Å]")
    ax2.set_ylabel("z [Å]")
    plt.tight_layout()
    return fig


# ===========================================================================
# Plots that need the run (device, vertices, DOS)
# ===========================================================================
def _zoom(ax, v, window, ylim=None):
    """Axis limits: anchors +/- window (Ang) in x and y; window None = whole device
    (y fitted to the atoms +/- 3 Ang instead of the vacuum of the cell)."""
    if window is not None:
        xy = v.g.xyz[v.anchors, :2]
        ax.set_xlim(xy[:, 0].min() - window, xy[:, 0].max() + window)
        ax.set_ylim(xy[:, 1].min() - window, xy[:, 1].max() + window)
    else:
        y = v.g.xyz[:, 1]
        ax.set_ylim(y.min() - 3.0, y.max() + 3.0)
    if ylim is not None:
        ax.set_ylim(*ylim)


def plot_setup(src, method: Optional[str] = None, ylim=None, show_regions: bool = True,
               region_lw: float = 1.0, region_style: str = "rings", window=None):
    """Device geometry with the vertex regions (red / blue rings, Mulliken moment of every
    defect cloud inside each region in the legend when available), the electrodes
    (green rings) and the anchors (dashed circles).

    src    : ExchangeRun, Device or DysonSystem (also device.plot_setup()).
    window : None (whole device, default) or half-width (Ang) around the anchors.
    """
    v = _view(src, method)
    g = v.g
    fig, ax = plot_atoms(g, annotate=False, size=12)
    handles = [] 
    if show_regions:
        labels = ["region i", "region j"]
        rep = v.info.get("region_moments_muB")
        if rep:
            for k in range(2):
                vals = [r["regions"][k] for r in rep.values()]
                labels[k] += ", m = " + " / ".join(f"{x:+.3f}" for x in vals) + " muB"
        handles += draw_regions(ax, g, region_atoms(v), lw=region_lw, labels=labels,
                                style=region_style)
    handles += draw_anchors(ax, v)
    handles += draw_electrodes(ax, v)
    _zoom(ax, v, window, ylim)
    _legend_below(ax, handles)
    ax.set_title(f"{v.method}: device ({v.builder}), region {v.region}, "
                 f"d = {v.distance:.2f} Å", fontsize=9)
    return fig


plot_regions = plot_setup        # merged into plot_setup (kept as an alias)


def plot_vertices(src, method: Optional[str] = None, ylim=None, vabs: float = 0.25,
                  show_regions: bool = True, region_lw: float = 1.0, both_regions: bool = True,
                  region_style: str = "rings", window: Optional[float] = 25.0):
    """Diagonal spin splitting of the two vertices per atom (pz, s and dz orbitals).

    src          : ExchangeRun, Device or DysonSystem (also device.plot_vertices()).
    window       : zoom, anchors +/- window Ang in x and y (default 10); None = whole device.
    show_regions : regions as open rings of the atom size, red = region of defect i,
                   blue = region of defect j, so the Delta colours stay visible
                   (region_style 'rings'; or 'outline' / 'bonds').
    both_regions : both regions on each figure (default) or only the one of the vertex.
    The electrodes (green rings) and the anchors (dashed circles) are always shown.
    For the Dyson methods the static part is shown; folded adatoms add an
    energy-dependent correction inside the exchange calculation.
    """
    v = _view(src, method)
    H, g = v.H, v.g
    regions = region_atoms(v) if show_regions else None
    figs = []
    for k, (D, o) in enumerate(zip(v.Ds, v.orbs)):
        color = (JD.get_Delta_orbital_atoms(H, D, "pz", orb_host=np.asarray(o))
                 + JD.get_Delta_orbital_atoms(H, D, "s", orb_host=np.asarray(o))
                 + JD.get_Delta_orbital_atoms(H, D, "dz", orb_host=np.asarray(o)))
        fig, ax = plot_atoms(g, color=color, size=8, vabs=vabs,
                             label=rf"$\Delta_{'ij'[k]}$ (eV)")
        handles = draw_electrodes(ax, v)
        if show_regions:
            show = [0, 1] if both_regions else [k]
            handles += draw_regions(ax, g, [regions[i] for i in show],
                                    colors=[REGION_COLORS[i] for i in show], lw=region_lw,
                                    labels=[("region i", "region j")[i] for i in show],
                                    style=region_style)
        handles += draw_anchors(ax, v)
        _zoom(ax, v, window, ylim)
        _legend_below(ax, handles)
        ax.set_title(rf"$\Delta_{'ij'[k]}$, {v.method}, region {v.region}", fontsize=9)
        figs.append(fig)
    return figs


def plot_dos(run_or_dir, energies=None, eta: float = 1e-2, orbitals="region"):
    """DOS of the reference of the method, with a comparison where it exists.

    rkky                         pristine device
    rkky_sup, liechtenstein_sup  defect region R, compared with the pristine host on R
    rkky_nm                      NM device, compared with the FM device (if available)
    liechtenstein                FM device, spin up and down

    Without `energies`, the curves computed by run_exchange(dos_energies=...) are
    plotted (from the run, or from results_name/dos.npz for a results folder);
    otherwise they are computed here (serially). orbitals: device builders, 'region'
    (defect regions, default), None (total device DOS) or device orbital indices.
    """
    from Exchange_Workflow import dos_curves, load_dos
    is_run = isinstance(run_or_dir, ExchangeRun)
    d = _dir(run_or_dir)
    curves = None
    if energies is None:
        curves = run_or_dir.dos if is_run and run_or_dir.dos else load_dos(d)
    if curves is None:
        if not is_run:
            raise ValueError(f"{d} has no dos.npz: run run_exchange(..., dos_energies=...) "
                             "or pass the ExchangeRun and energies")
        E = np.linspace(-1, 1, 100) if energies is None else np.asarray(energies)
        curves = dos_curves(run_or_dir.obj, run_or_dir.method, run_or_dir.config["builder"], E,
                            eta=eta, orbitals=orbitals)
    meta = curves.pop("_meta", None) if isinstance(curves, dict) else None
    method = run_or_dir.method if is_run else (meta or {}).get("method", "")
    builder = run_or_dir.config["builder"] if is_run else ""
    E = next(iter(curves.values()))["energies"]
    fig, ax = plt.subplots(figsize=(4.5, 4))
    for label, c in curves.items():
        for k, sp in enumerate(c["spins"]):
            ax.plot(E, c["dos"][k], label=f"{label} {sp}" if len(c["spins"]) > 1 else label, lw=1)
    ax.set_xlabel("Energy [eV]")
    region = builder == "SuperpositionDyson" or (meta or {}).get("orbitals", orbitals) == "region"
    ax.set_ylabel("DOS [states/eV]" + (" of the defect region" if region else ""))
    ax.set_ylim(0, None)
    ax.set_xlim(np.min(E), np.max(E))
    ax.legend(fontsize=7)
    ax.set_title(f"{method}: DOS")
    plt.tight_layout()
    return fig, curves


def plot_all(run: ExchangeRun, dos: bool = True, ylim=None):
    """Every visualization of the tutorials for this run."""
    print(run.summary())
    plot_setup(run, ylim=ylim)
    plot_vertices(run)
    plot_orbital_contributions(run)
    plot_exchange_kernel(run)
    plot_atomic_contributions(run)
    atomic_contributions_table(run)
    if dos:
        plot_dos(run)
    plt.show()

def plot_contour(E_list,eta_list,W_real,W_imag,main_folder_local=None):
    vabs=np.max(np.abs(W_imag))
    fig, ax = plt.subplots(1, 1)
    sc=ax.scatter(E_list,eta_list,c=W_imag,s=50*np.array(W_real),cmap='viridis',vmin=-vabs,vmax=vabs)
    ax.plot(E_list, eta_list, color='k')
    ax.set_ylabel('Im$(E)$',size=16)
    ax.set_xlabel('Re$(E)$',size=16)
    divider = make_axes_locatable(ax)
   
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar=plt.colorbar(sc, cax=cax)
    cbar.set_label(r'Im(w($E$))', size=14)
    ax.set_aspect('equal')
    if main_folder_local!= None:
        fig.savefig(os.path.join(main_folder_local, "plots", f"CC.png"), bbox_inches='tight')
    return fig,ax

def plot_pz_bands(H_0,H_0_DFT,path,sym_names,main_folder_local=None):
    # Plot bands 
    fig = plt.figure('Bands',figsize=(2,6))
    ax = fig.add_subplot(111)
    
    for ham_idx, ham in enumerate([H_0_DFT,H_0]):
        band = sisl.BandStructure(ham,path,
                               30, sym_names)
        lk, kt, kl = band.lineark(True)
        ax.set_ylabel(r'$E-E_F$')
        bs = []
        for ik, eigh in enumerate(band.apply.eigh()):
            bs.append(eigh)
        bs= np.array(bs)
        
        ax.set_ylim(-2,2)
        firsttime=True
        if ham_idx==0:
            c='k'
            label='DFT'
        else:
            c='r'
            label=r'$p_z$TB'
        
        for bk in bs.T:
            if(firsttime):
                ax.plot(lk, bk, c=c, label=label)
                firsttime=False
            else:
                ax.plot(lk, bk, c=c)
                
    xcoords = kt
    for k in xcoords:
        plt.axvline(x=k,c='k')        
    plt.xticks(kt, kl)        
    ax.legend(loc='center right')
    ax.set_xlim(kt[0],kt[-1])
    if main_folder_local!=None:
        fig.savefig(os.path.join(main_folder_local, "plots", f"Bands.png"), bbox_inches='tight')
    return fig


def plot_atoms(device,color=[],label=None,annotate=None,size=8,vabs=0.25):
    fig = plt.figure(figsize=(size, size))
    ax = plt.gca()
    ax.set_aspect('equal')# Plot bonds
    bonds=True
    if bonds==True:
        for i, Z in enumerate(device.atoms.Z):
            for j in range(i + 1, len((device.xyz))):
                dist_3d = np.linalg.norm((device.xyz[i] - (device.xyz[j])))
                if dist_3d < 1.77:
                    x_vals = [device.xyz[i][0], device.xyz[j][0]]
                    z_vals = [device.xyz[i][1], device.xyz[j][1]]
                    ax.plot(x_vals, z_vals, color='gray', linewidth=1.2,zorder=1)
            if len(color) < 1:
                if Z==5:        
                    ax.scatter(device.xyz[i,0],device.xyz[i,1],c='pink',s=70,alpha=1,edgecolors='k',linewidths=0.1) 
                if Z==1:
                    ax.scatter(device.xyz[i,0],device.xyz[i,1],c='white',s=10,alpha=1,edgecolors='k',linewidths=0.1)
                if Z==6:        
                    ax.scatter(device.xyz[i,0],device.xyz[i,1],c='gray',s=50,alpha=1,edgecolors='k',linewidths=0.1)
                if Z==7:        
                    ax.scatter(device.xyz[i,0],device.xyz[i,1],c='blue',s=70,alpha=1,edgecolors='k',linewidths=0.1) 
            else:
                if Z==5:        
                    ax.scatter(device.xyz[i,0],device.xyz[i,1],c='pink',s=70,alpha=1,edgecolors='k',linewidths=0.1) 
                if Z==1:
                    pcm1=ax.scatter(device.xyz[i,0],device.xyz[i,1],c=color[i],cmap='RdBu',vmin=-vabs,vmax=vabs,s=10,alpha=1,edgecolors='k',linewidths=0.1)
                if Z==6:        
                    pcm1=ax.scatter(device.xyz[i,0],device.xyz[i,1],c=color[i],cmap='RdBu',vmin=-vabs,vmax=vabs,s=50,alpha=1,edgecolors='k',linewidths=0.1)
                if Z==7:        
                    pcm1=ax.scatter(device.xyz[i,0],device.xyz[i,1],c=color[i],cmap='RdBu',vmin=-vabs, vmax=vabs,s=70,alpha=1,edgecolors='k',linewidths=0.1)   
            if annotate==True:
                ax.text(device.xyz[i,0],device.xyz[i,1],i,size=10)
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_xlim(0,device.cell[0,0])
    ax.set_ylim(0,device.cell[1,1])
    if label != None:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(pcm1, cax=cax).set_label(label=label,size=15)
    return fig, ax
