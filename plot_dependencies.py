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
    plot_setup(run)       device, vertex regions, electrodes, adatoms, defect region
    plot_vertices(run)    spin splitting of the two vertices per atom
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

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import J_dependencies as JD
from Device_Builder import compute_dos
from Exchange_Workflow import ExchangeRun, load_results

__all__ = ["plot_setup", "plot_vertices", "plot_orbital_contributions", "plot_exchange_kernel",
           "plot_atomic_contributions", "atomic_contributions_table", "plot_dos", "plot_all",
           "optional_import"]


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
    if korb.is_file():
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
def plot_setup(run: ExchangeRun, ylim=None):
    """Device geometry: anchors, vertex regions, electrodes, adatoms (and region R)."""
    H = run.H
    g = H.geometry
    fig, ax = plot_atoms(g, annotate=False, size=12)
    _, orbs = run.vertices()
    colors = ("red", "blue")

    for i, color in enumerate(colors):
        anchor = int(run.obj.anchors[i])    
        ax.scatter(
            [g.xyz[anchor, 0]],
            [g.xyz[anchor, 1]],
            s=150,
            marker="o",
            edgecolors=color,facecolors="none",
            zorder=10,label=fr"Defect {i}")
    if run.config["builder"] == "SuperpositionDyson":
        R_at = g.o2a(run.obj.R, unique=True)
        ax.scatter(g.xyz[R_at, 0], g.xyz[R_at, 1], s=70, facecolors="none",edgecolors="k", label="region R")
    for orbs_e, col in ((run.obj.left_orbs, "tab:green"), (run.obj.right_orbs, "tab:green")):
        at = g.o2a(np.asarray(orbs_e), unique=True)
        ax.scatter(g.xyz[at, 0], g.xyz[at, 1], s=70, c=col,edgecolors='k',linewidths=0.1)
    h = np.flatnonzero(g.atoms.Z == 1)
    h = [a for a in h if min(np.linalg.norm(g.xyz[a] - g.xyz[list(run.obj.anchors)], axis=1)) < 1.6]
    #if h:
    #    ax.scatter(g.xyz[h, 0], g.xyz[h, 1], s=60, c="cyan", edgecolors="black", label="adatoms")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=7)
    ax.set_title(f"{run.method}: device ({run.config['builder']}), "
                 f"d = {run.obj.anchor_distance:.2f} Å")
    return fig


def plot_vertices(run: ExchangeRun, ylim=None, vabs: float = 0.5):
    """Diagonal spin splitting of the two vertices per atom (pz and s orbitals).

    For the Dyson methods the static part is shown; folded adatoms add an
    energy-dependent correction inside the exchange calculation.
    """
    H = run.H
    g = H.geometry
    Ds, orbs = run.vertices()
    figs = []
    colors = ("red", "blue")
    for k, (D, o) in enumerate(zip(Ds, orbs)):
        color = (JD.get_Delta_orbital_atoms(H, D, "pz", orb_host=np.asarray(o))
                 + JD.get_Delta_orbital_atoms(H, D, "s", orb_host=np.asarray(o))+ JD.get_Delta_orbital_atoms(H, D, "dz", orb_host=np.asarray(o)))
        fig, ax = plot_atoms(g, color=color, size=8, vabs=vabs,
                                label=rf"$\Delta_{'ij'[k]}$ (eV)")
        for i, color in enumerate(colors):
          ax.scatter(g.xyz[run.obj.anchors[i], 0],g.xyz[run.obj.anchors[i], 1], s=150, facecolors="none",edgecolors=color)
        if ylim is not None:
            ax.set_ylim(*ylim)
        figs.append(fig)
    return figs


def plot_dos(run: ExchangeRun, energies=None, eta: float = 1e-2):
    """DOS of the reference of the method, with a comparison where it exists.

    rkky                         pristine device
    rkky_sup, liechtenstein_sup  defect region R, compared with the pristine host on R
    rkky_nm                      NM device, compared with the FM device (if available)
    liechtenstein                FM device, spin up and down
    """
    E = np.linspace(-1, 1, 100) if energies is None else np.asarray(energies)
    curves = {run.method: compute_dos(run.obj, run.method, E, eta=eta)}
    if run.config["builder"] == "SuperpositionDyson":
        curves["pristine host (same region)"] = compute_dos(run.obj, "rkky", E, eta=eta)
    if run.method == "rkky_nm" and "liechtenstein" in run.obj.methods:
        curves["FM device"] = compute_dos(run.obj, "liechtenstein", E, eta=eta)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    for label, d in curves.items():
        for k, s in enumerate(d["spins"]):
            ax.plot(E, d["dos"][k], label=f"{label} {s}" if len(d["spins"]) > 1 else label, lw=1)
    ax.set_xlabel("Energy [eV]")
    ax.set_ylabel("DOS [states/eV]" + (" of R" if run.config["builder"] == "SuperpositionDyson" else ""))
    ax.set_ylim(0, None)
    ax.set_xlim(min(energies), max(energies))
    ax.legend(fontsize=7)
    ax.set_title(f"{run.method}: DOS")
    plt.tight_layout()
    return fig, curves


def plot_all(run: ExchangeRun, dos: bool = True, ylim=None):
    """Every visualization of the tutorials for this run."""
    print(run.summary())
    plot_setup(run, ylim=ylim)
    plot_vertices(run, ylim=ylim)
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


def plot_atoms(device,color=[],label=None,annotate=None,size=8,vabs=0.1):
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
