"""Extract a NATURAL deposited structure's chain as an all-atom PDB target for WS3 hallucination.

USE A DEPOSITED/EXPERIMENTAL STRUCTURE, never a diffusion-sampled or cg2all-reconstructed one
(those are model outputs, not natural folds). Default: 7ad5 chain A (effector AvrLm5-9), 124 aa,
from the box mmcif store -> ~/data/7ad5_natural/7ad5_A.pdb.
  python ws3_prep_natural_target.py [--cif ...] [--chain A] [--out ...]
"""
import argparse
import os

from Bio.PDB import MMCIFParser, PDBIO, Select
from Bio.PDB.Polypeptide import is_aa

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--cif", default="/home/jupyter-chenxi/data/pdb_mmcif/mmcif_files/7ad5.cif")
ap.add_argument("--chain", default="A")
ap.add_argument("--out", default="/home/jupyter-chenxi/data/7ad5_natural/7ad5_A.pdb")
args = ap.parse_args()

s = MMCIFParser(QUIET=True).get_structure("x", args.cif)


class Sel(Select):
    def accept_chain(self, c):
        return c.id == args.chain

    def accept_residue(self, r):
        return is_aa(r, standard=True)

    def accept_atom(self, a):
        return a.element != "H"


os.makedirs(os.path.dirname(args.out), exist_ok=True)
io = PDBIO()
io.set_structure(s)
io.save(args.out, Sel())
n = sum(1 for ln in open(args.out) if ln.startswith("ATOM") and ln[12:16].strip() == "CA")
print(f"wrote {args.out} ({n} residues, chain {args.chain})")
