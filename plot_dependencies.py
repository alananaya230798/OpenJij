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
               region_lw: float = 0.25, region_style: str = "rings", window=None):
    """Device geometry with the vertex regions (red / blue rings, Mulliken moment of every
    defect cloud inside each region in the legend when available), the electrodes
    (green rings) and the anchors (dashed circles).

    src    : ExchangeRun, Device or DysonSystem (also device.plot_setup()).
    window : None (whole device, default) or half-width (Ang) around the anchors.
    """
    v = _view(src, method)
    g = v.g
    fig, ax = plot_atoms(g, annotate=False, size=12)
    handles=[]
    if show_regions:
        labels = ["region i", "region j"]
        rep = v.info.get("region_moments_muB")
        if rep:
            for k in range(2):
                vals = [r["regions"][k] for r in rep.values()]
                labels[k] += ", m = " + " / ".join(f"{x:+.3f}" for x in vals) + " muB"
        handles += draw_regions(ax, g, region_atoms(v), lw=region_lw, labels=labels,
                                style=region_style)
    handles += draw_electrodes(ax, v)
    handles += draw_anchors(ax, v)
    _zoom(ax, v, window, ylim)
    _legend_below(ax, handles)
    ax.set_title(f"{v.method}: device ({v.builder}), region {v.region}, "
                 f"d = {v.distance:.2f} Å", fontsize=9)
    return fig


plot_regions = plot_setup        # merged into plot_setup (kept as an alias)


def plot_vertices(src, method: Optional[str] = None, ylim=None, vabs: float = 0.25,
                  show_regions: bool = True, region_lw: float = 0.25, both_regions: bool = True,
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
