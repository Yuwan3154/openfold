"""Grid of Protpardelle-1c synthetic templates superposed on their native, colored by the template's
OWN DSSP 3-state call (helix / strand / coil); the native is a thin grey trace underneath.
Rows = model (cc89, cc91, cc94), columns = the sample whose TM-to-native is closest to each target
(0.9 ... 0.4, the band asked for), plus a native-only reference panel.

Run (headless, env with pymol + mdtraj):  pymol -cq render_sse_overlays.py -- <deepdive_dir> <chain> [<chain> ...]
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
import pandas as pd
from pymol import cmd

D = sys.argv[1]
CHAINS = sys.argv[2:]
MODELS = ["cc89", "cc91", "cc94"]
TARGETS = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
SS_RGB = {"H": (0xD1, 0x49, 0x5B), "E": (0x2E, 0x86, 0xAB), "C": (0xB8, 0xB8, 0xB8)}
PYMOL_SS = {"H": "H", "E": "S", "C": "L"}
PX = 700


def dssp(pdb):
    t = md.load_pdb(pdb)
    prot = [i for i, r in enumerate(t.topology.residues) if r.is_protein]
    return "".join(np.array(md.compute_dssp(t, simplified=True)[0])[prot])


def style():
    cmd.bg_color("white")
    cmd.set("ray_opaque_background", 1)
    cmd.set("ray_trace_mode", 1)
    cmd.set("ray_trace_color", "grey30")
    cmd.set("antialias", 2)
    cmd.set("light_count", 2)
    cmd.set("cartoon_fancy_helices", 1)
    cmd.set("cartoon_transparency", 0.0)
    for k, rgb in SS_RGB.items():
        cmd.set_color(f"ss_{k}", [c / 255 for c in rgb])


def apply_ss(obj, ss):
    resi = [a.resi for a in cmd.get_model(f"{obj} and name CA").atom]
    assert len(resi) == len(ss), (obj, len(resi), len(ss))
    for r, s in zip(resi, ss):
        cmd.alter(f"{obj} and resi \\{r}", f"ss='{PYMOL_SS[s]}'")
        cmd.color(f"ss_{s}", f"{obj} and resi \\{r}")


def render(native, template, out, view=None):
    cmd.reinitialize()
    style()
    cmd.load(native, "native")
    cmd.hide("everything")
    ss_n = dssp(native)
    if template is None:
        apply_ss("native", ss_n)
        cmd.show("cartoon", "native")
    else:
        cmd.load(template, "tpl")
        cmd.hide("everything")
        cmd.pair_fit("tpl and name CA", "native and name CA")  # same residues in the same order
        apply_ss("tpl", dssp(template))
        cmd.show("cartoon", "tpl")
        # thin CA ribbon: cartoon_trace_atoms on an all-atom native traces through every atom
        cmd.set("ribbon_trace_atoms", 0)
        cmd.set("ribbon_width", 2.5)
        cmd.show("ribbon", "native and name CA")
        cmd.color("grey70", "native")
        cmd.set("ribbon_transparency", 0.3)
    cmd.rebuild()
    if view is None:
        cmd.orient("native")
        cmd.zoom("native", 4)
    else:
        cmd.set_view(view)
    cmd.ray(PX, PX)
    cmd.png(out, dpi=150)
    return cmd.get_view()


def main():
    sd = pd.read_csv(os.path.join(D, "out", "bakeoff_sse_drift.csv"))
    tmt = pd.read_csv(os.path.join(D, "pp1c_tm_local.csv"))
    tmt["chain_id"] = tmt.pdb_id + "_" + tmt.chain
    sd = sd.merge(tmt[["model", "chain_id", "rewind_steps", "sample", "pdb_file"]],
                  left_on=["model", "chain", "rewind", "sample"],
                  right_on=["model", "chain_id", "rewind_steps", "sample"], validate="one_to_one")
    outdir = os.path.join(D, "figs", "overlays")
    os.makedirs(outdir, exist_ok=True)
    picks = []
    for chain in CHAINS:
        native = os.path.join(D, "natives", f"{chain}.pdb")
        view = render(native, None, os.path.join(outdir, f"{chain}_native.png"))
        fig, axes = plt.subplots(len(MODELS), len(TARGETS) + 1,
                                 figsize=(2.3 * (len(TARGETS) + 1), 2.55 * len(MODELS)))
        for i, m in enumerate(MODELS):
            sub = sd[(sd.chain == chain) & (sd.model == m)]
            ax = axes[i, 0]
            ax.imshow(plt.imread(os.path.join(outdir, f"{chain}_native.png")))
            ax.set_title("native" if i == 0 else "", fontsize=9)
            ax.set_ylabel(m, fontsize=11, fontweight="bold")
            for j, t in enumerate(TARGETS):
                r = sub.iloc[(sub.tm - t).abs().argmin()]
                png = os.path.join(outdir, f"{chain}_{m}_tm{t:.1f}.png")
                render(native, r.pdb_file, png, view)
                ax = axes[i, j + 1]
                ax.imshow(plt.imread(png))
                ax.set_title(f"TM {r.tm:.2f}  Q3 {r.q3:.2f}\nrewind {int(r.rewind)}  s{int(r['sample'])}",
                             fontsize=8)
                picks.append(dict(chain=chain, model=m, target=t, tm=r.tm, q3=r.q3, rewind=int(r.rewind),
                                  sample=int(r["sample"]), elem_local_rmsd=r.elem_local_rmsd,
                                  elem_global_rmsd=r.elem_global_rmsd, pdb_file=r.pdb_file))
        for ax in axes.flat:
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
        handles = [plt.Rectangle((0, 0), 1, 1, color=[c / 255 for c in SS_RGB[k]]) for k in "HEC"]
        fig.legend(handles, ["helix (DSSP H/G/I)", "strand (E/B)", "coil"], loc="lower center",
                   ncol=3, frameon=False, fontsize=9)
        fig.suptitle(f"{chain}: templates closest to each target TM, colored by their own DSSP; "
                     f"grey trace = native", fontsize=10)
        fig.tight_layout(rect=(0, 0.04, 1, 0.96))
        fig.savefig(os.path.join(D, "figs", f"overlay_grid_{chain}.png"), dpi=130)
        plt.close(fig)
    pd.DataFrame(picks).to_csv(os.path.join(D, "figs", "overlay_picks.csv"), index=False)
    print("wrote", len(picks), "panels")


main()
