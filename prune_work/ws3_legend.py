"""WS3 legend overlay (run in cue_openfold_gated): load ws3_overlay_raw.png + summary.json, add a
color legend + TM-score title -> ws3_overlay_final.png.
  python ws3_legend.py <out_dir>
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Patch

out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
img = mpimg.imread(f"{out_dir}/ws3_overlay_raw.png")
m = json.load(open(f"{out_dir}/summary.json"))


def fmt(x):
    return "n/a" if x is None else f"{x:.3f}"


h, w = img.shape[0], img.shape[1]
fig, ax = plt.subplots(figsize=(w / 150, h / 150 + 1.4))  # extra height for legend/title
ax.imshow(img)
ax.axis("off")
handles = [
    Patch(facecolor="grey", edgecolor="black", label=f"Target natural  ({m['target_name']}, L={m['L']})"),
    Patch(facecolor="royalblue", edgecolor="black", label=f"Slim design prediction    TM->target = {fmt(m['tm_slim_target'])}"),
    Patch(facecolor="orange", edgecolor="black", label=f"Full-AF2 refold of design  TM->target = {fmt(m['tm_full_target'])}"),
]
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
          ncol=1, fontsize=12, frameon=True)
ax.set_title(
    "WS3: slim-model monomer hallucination (distogram loss) vs target\n"
    f"designed seq folded by slim vs full AF2 -- cross-model TM(full,slim) = {fmt(m['tm_full_slim'])}",
    fontsize=13)
plt.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.16)
plt.savefig(f"{out_dir}/ws3_overlay_final.png", dpi=150, transparent=True)
print(f"wrote {out_dir}/ws3_overlay_final.png")
