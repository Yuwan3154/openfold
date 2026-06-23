"""WS3 render (run in the pymol env). Superpose target + slim_pred + full_pred by CA residue
index (exact sequence match-up), pick ONE shared camera, then emit (ALL transparent bg):
  - ws3_overlay_raw.png   : all three superposed (legend added later by ws3_legend.py)
  - target_only.png       : target alone     (for slides)
  - slim_only.png         : slim_pred alone
  - full_only.png         : full_pred alone
Individual images share the overlay camera so they overlay cleanly in slides.

Style = user's preferred publication look (Display>Quality>Maximum Quality + light_count 1 +
ray_trace_mode 1 + antialias 0), transparent background, ray-traced export. See memory
reference_pymol_rendering.
  pymol -cq ws3_render_pymol.py -- <out_dir>
"""
import sys
from pymol import cmd, util

out_dir = sys.argv[1] if len(sys.argv) > 1 else "."

cmd.load(f"{out_dir}/target.pdb", "target")
cmd.load(f"{out_dir}/slim_pred.pdb", "slim")
cmd.load(f"{out_dir}/full_pred.pdb", "full")

# exact sequence match-up: pair CA_i <-> CA_i by residue order (all three equal length L)
cmd.pair_fit("slim and name CA", "target and name CA")
cmd.pair_fit("full and name CA", "target and name CA")

cmd.hide("everything")
cmd.show("cartoon")
cmd.color("grey70", "target")
cmd.color("marine", "slim")
cmd.color("orange", "full")

# --- user's preferred publication style ---
util.performance(0)                  # Display > Quality > Maximum Quality
cmd.set("light_count", 1)
cmd.set("ray_trace_mode", 1)         # black-outline cartoon look
cmd.set("antialias", 0)
cmd.set("ray_opaque_background", 0)  # TRANSPARENT background = default for all pngs

# one shared camera, tight framing
cmd.orient()
cmd.zoom("all", 2, complete=1)

W, H = 1800, 1400

# --- overlay (transparent) ---
cmd.enable("all")
cmd.ray(W, H)
cmd.png(f"{out_dir}/ws3_overlay_raw.png", dpi=150)
print(f"wrote {out_dir}/ws3_overlay_raw.png", flush=True)

# --- individual figures, TRANSPARENT bg, SAME camera ---
for name in ("target", "slim", "full"):
    cmd.disable("all")
    cmd.enable(name)
    cmd.ray(W, H)  # ray preserves the current camera (no re-orient)
    cmd.png(f"{out_dir}/{name}_only.png", dpi=150)
    print(f"wrote {out_dir}/{name}_only.png", flush=True)
