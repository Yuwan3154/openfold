"""Figures for the T2 template deep-dive: SSE/drift decomposition vs TM, and compatibility vs TM.

Run: python plot_deepdive.py --dir <deepdive_dir>   (reads out/compat_drift_joined.csv)
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODEL_C = {"cc89": "#E07A2E", "cc91": "#2A78D6", "cc94": "#1BAF7A"}
BINS = np.round(np.arange(0.15, 1.0001, 0.05), 2)


def binned(ax, d, y, ylabel, ref=None):
    for m, g in d.groupby("model"):
        ax.scatter(g.tm, g[y], s=6, alpha=0.25, color=MODEL_C[m], linewidths=0)
        b = pd.cut(g.tm, BINS)
        mm = g.groupby(b, observed=True)[y].mean()
        x = [iv.mid for iv in mm.index]
        ax.plot(x, mm.values, color=MODEL_C[m], lw=2, label=m)
    ax.axvspan(0.4, 0.9, color="0.92", zorder=-1)
    if ref is not None:
        ax.axhline(ref, color="0.4", ls="--", lw=1)
    ax.set_xlabel("TM to native (template)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0.15, 1.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    a = p.parse_args()
    d = pd.read_csv(os.path.join(a.dir, "out", "compat_drift_joined.csv"))
    # differences, not ratios: a ratio explodes on a native with ~no strand (4hto_A E = 0.02)
    d["dH"] = d.fH - d.fH_native
    d["dE"] = d.fE - d.fE_native
    out = os.path.join(a.dir, "figs")

    fig, ax = plt.subplots(2, 3, figsize=(13, 7.2))
    binned(ax[0, 0], d, "q3", "Q3 agreement with native DSSP", 1.0)
    binned(ax[0, 1], d, "dH", "helix fraction − native", 0.0)
    binned(ax[0, 2], d, "dE", "strand fraction − native", 0.0)
    binned(ax[1, 0], d, "elem_local_rmsd", "element CA RMSD, fit ALONE (Å)")
    binned(ax[1, 1], d, "elem_global_rmsd", "element CA RMSD, whole-chain fit (Å)")
    binned(ax[1, 2], d, "q_contacts", "native contacts kept (CASP 8 Å)", 1.0)
    ax[0, 0].legend(frameon=False)
    for x in (ax[0, 0], ax[1, 0], ax[1, 1], ax[1, 2]):
        x.set_ylim(bottom=0)
    fig.suptitle("How Protpardelle-1c templates depart from the native (5 chains × 16 rewinds × 3 seeds per model; "
                 "grey band = TM 0.4-0.9)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "sse_drift_vs_tm.png"), dpi=140)
    plt.close(fig)

    comp = [("d_mpnn_nll", "MPNN NLL(query) − native (nat/res)")]
    if "d_ebm_ptm" in d:
        comp += [("d_ebm_energy_per_res", "ProteinEBM energy/res − native"), ("d_ebm_ptm", "ProteinEBM pTM − native")]
    fig, ax = plt.subplots(1, len(comp), figsize=(4.4 * len(comp), 3.8), squeeze=False)
    for i, (k, lab) in enumerate(comp):
        binned(ax[0, i], d, k, lab, 0.0)
    ax[0, 0].legend(frameon=False)
    fig.suptitle("Query-sequence compatibility vs how far the template moved", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "compat_vs_tm.png"), dpi=140)
    plt.close(fig)
    print("wrote figures")


if __name__ == "__main__":
    main()
