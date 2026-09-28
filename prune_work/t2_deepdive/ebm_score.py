"""ProteinEBM (pae.ckpt) energy + pTM of structures under their OWN sequence (read from each PDB).

Same checkpoint, eval time (t=0.05, pae_config.yaml's eval_time) and feature convention as
~/score_synthetic_templates.py (the 2026-08-12 Proteina analysis), with one change taken from the
repo's own protein_ebm/scripts/score_decoys.py: CA coordinates are CENTERED on their centroid
before scoring (score_decoys calls center_random_augmentation, which centers and randomly rotates).
The 08-12 script fed raw PDB coordinates; --no-center reproduces it for comparison.
--rotations K scores K random rotations (seeded) and reports their mean and spread; K=0 means the
centered input as-is.

Run: <proteinebm env>/bin/python ebm_score.py --list pdbs.txt --out-csv out.csv [--rotations K]
"""
import argparse
import csv
import sys

import numpy as np
import torch
import yaml
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa, three_to_index, index_to_one
from ml_collections import ConfigDict

sys.path.insert(0, "/home/jupyter-chenxi/ProteinEBM")
from protein_ebm.data.protein_utils import restype_order, restype_num  # noqa: E402
from protein_ebm.model.heads import ProteinRegressionTrainer  # noqa: E402
from protein_ebm.model.loss import compute_ptm  # noqa: E402

PAE_CONFIG = "/home/jupyter-chenxi/ProteinEBM/protein_ebm/config/pae_config.yaml"
PAE_CKPT = "/home/jupyter-chenxi/ProteinEBM/weights/pae.ckpt"
SCORING_T = 0.05


def read(pdb):
    s = PDBParser(QUIET=True).get_structure("s", pdb)
    ca, seq = [], []
    for r in s.get_residues():
        if not is_aa(r) or not r.has_id("CA"):
            continue
        ca.append(r["CA"].get_coord())
        seq.append(index_to_one(three_to_index(r.get_resname())))
    return np.array(ca, np.float32), "".join(seq)


def random_rotation(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], np.float32)


def score(model, ca, seq):
    n = len(seq)
    aat = torch.tensor([restype_order.get(c, restype_num) for c in seq], dtype=torch.long)
    batch = {
        "r_noisy": torch.tensor(ca).unsqueeze(0),
        "aatype": aat.unsqueeze(0),
        "mask": torch.ones(1, n),
        "residue_idx": torch.arange(n).unsqueeze(0),
        "t": torch.full((1,), SCORING_T),
        "chain_encoding": torch.zeros(1, n, dtype=torch.long),
        "external_contacts": torch.ones(1, n, dtype=torch.long),  # num_contact_embeddings=3
        "atom_mask": torch.ones(1, n),
    }
    with torch.no_grad():
        out, _, pae_logits, _, _ = model.forward(batch)
    return out["energy"].flatten()[0].item(), compute_ptm(pae_logits, batch["mask"]).flatten()[0].item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--list", required=True, help="file with one PDB path per line")
    p.add_argument("--out-csv", required=True)
    p.add_argument("--rotations", type=int, default=0)
    p.add_argument("--no-center", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=8)
    a = p.parse_args()
    torch.set_num_threads(a.threads)

    with open(PAE_CONFIG) as f:
        config = ConfigDict(yaml.safe_load(f))
    model = ProteinRegressionTrainer.load_from_checkpoint(
        PAE_CKPT, config=config, ebm_checkpoint_path=None, map_location="cpu").eval()

    paths = [l.strip() for l in open(a.list) if l.strip()]
    rng = np.random.default_rng(a.seed)
    with open(a.out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pdb_file", "L", "energy", "ptm", "energy_sd", "ptm_sd", "n_rot", "centered"])
        for i, pdb in enumerate(paths):
            ca, seq = read(pdb)
            if not a.no_center:
                ca = ca - ca.mean(0)
            if a.rotations > 0:
                es, ps = zip(*[score(model, ca @ random_rotation(rng).T, seq) for _ in range(a.rotations)])
            else:
                es, ps = zip(*[score(model, ca, seq)])
            w.writerow([pdb, len(seq), np.mean(es), np.mean(ps), np.std(es), np.std(ps),
                        a.rotations, not a.no_center])
            fh.flush()
            if i % 50 == 0:
                print(f"{i}/{len(paths)} {pdb.split('/')[-1]} E={np.mean(es):.1f} pTM={np.mean(ps):.3f}",
                      flush=True)
    print(f"wrote {len(paths)} rows to {a.out_csv}")


if __name__ == "__main__":
    main()
