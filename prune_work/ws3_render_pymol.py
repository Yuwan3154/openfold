"""WS3 render (run in the pymol env): superpose target + slim_pred + full_pred by CA residue
index (exact sequence match-up), cartoon, color each, ray-trace -> ws3_raw.png.
  pymol -cq ws3_render_pymol.py -- <out_dir>
"""
import sys
from pymol import cmd

out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
cmd.load(f"{out_dir}/target.pdb", "target")
cmd.load(f"{out_dir}/slim_pred.pdb", "slim")
cmd.load(f"{out_dir}/full_pred.pdb", "full")

# exact sequence match-up: pair CA_i <-> CA_i by residue order (all three have equal length L)
cmd.pair_fit("slim and name CA", "target and name CA")
cmd.pair_fit("full and name CA", "target and name CA")

cmd.hide("everything")
cmd.show("cartoon")
cmd.color("grey70", "target")
cmd.color("marine", "slim")
cmd.color("orange", "full")
cmd.set("cartoon_transparency", 0.1)
cmd.bg_color("white")
cmd.set("ray_opaque_background", 1)
cmd.orient()
cmd.ray(1500, 1200)
cmd.png(f"{out_dir}/ws3_raw.png", dpi=150)
print(f"wrote {out_dir}/ws3_raw.png")
