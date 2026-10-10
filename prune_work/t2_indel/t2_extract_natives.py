"""Extract the natives of the production chain list from proteina's processed .pt files (env proteina: torch_geometric; CPU). Keys are proteina LABEL chain IDs
<pdbid>_<label_asym_id> (T8 handoff 10-10): never author IDs. Each .pt holds coords (L, 37, 3) and coord_mask (L, 37) in proteina's on-disk atom order (N, CA, C, O, CB, ...,
proteinfoundation.utils.constants.ATOM_NUMBERING); they are reordered to the AF/OpenFold atom37 order the pipeline (Protpardelle) uses with PDB_TO_OPENFOLD_INDEX_TENSOR.
Writes <out>/<id[1:3]>/<id>.npz: coords (L,37,3) f32 atom37, mask (L,37) bool, sequence str, residue_pdb_idx (L,), n_incomplete (residues missing any of N/CA/C/O).
Residues with a missing N/C/O (CA present) are KEPT: their mask stays False, stage A fills the missing backbone atoms with the CA only to run the edit geometry, and stage B marks those atoms
unknown so the sampler dummy-fills and noises them (T8 DSSP rule: such residues are -1 in DSSP). Chains that are missing, have a sequence/length mismatch or a residue without CA are NOT written
and are recorded in <out>/skipped.tsv. The processed files are not all under <id[1:3]>/ (the tree is sharded), so --index (TSV id<TAB>path, made once with `find -L`) is used when given.
Resumable (existing npz skipped). With --stats-only nothing is written: prints the skip statistics.
Run: python t2_extract_natives.py --chains-file ids.txt --processed-dir P --out-dir O --proteina-src /path/to/proteina [--index idx.tsv] [--stats-only]
"""
import argparse
import os
import sys

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains-file", required=True)
    ap.add_argument("--processed-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--proteina-src", required=True, help="proteina checkout (read-only) providing proteinfoundation.utils.constants")
    ap.add_argument("--index", default=None, help="TSV id<TAB>path of the processed .pt files")
    ap.add_argument("--stats-only", action="store_true")
    a = ap.parse_args()
    sys.path.insert(0, a.proteina_src)
    from proteinfoundation.utils.constants import PDB_TO_OPENFOLD_INDEX_TENSOR as reorder

    ids = [ln.strip() for ln in open(a.chains_file) if ln.strip()]
    path_of = dict(ln.rstrip("\n").split("\t") for ln in open(a.index)) if a.index else {}
    skipped, n_ok, lens = [], 0, []
    os.makedirs(a.out_dir, exist_ok=True)
    for cid in ids:
        out = os.path.join(a.out_dir, cid[1:3], cid + ".npz")
        if os.path.isfile(out) and not a.stats_only:
            n_ok += 1
            continue
        pt = path_of.get(cid, os.path.join(a.processed_dir, cid[1:3], cid + ".pt"))
        if not os.path.isfile(pt):
            skipped.append((cid, "no processed file", ""))
            continue
        d = torch.load(pt, weights_only=False)
        L = int(d.coords.shape[0])
        coords, mask = d.coords[:, reorder].numpy().astype(np.float32), d.coord_mask[:, reorder].numpy().astype(bool)
        if len(d.sequence) != L:
            skipped.append((cid, "sequence length differs from coords", f"{len(d.sequence)} vs {L}"))
            continue
        if not mask[:, 1].all():
            skipped.append((cid, "residue without CA", f"{int((~mask[:, 1]).sum())} of {L} residues"))
            continue
        n_inc = int((~mask[:, [0, 1, 2, 4]].all(1)).sum())   # AF order: N 0, CA 1, C 2, CB 3, O 4: residues kept with missing backbone atoms
        n_ok += 1
        lens.append(L)
        if not a.stats_only:
            os.makedirs(os.path.dirname(out), exist_ok=True)
            np.savez(out + ".tmp.npz", coords=coords, mask=mask, sequence=np.array(d.sequence), residue_pdb_idx=d.residue_pdb_idx.numpy(), n_incomplete=np.int32(n_inc))
            os.replace(out + ".tmp.npz", out)
    with open(os.path.join(a.out_dir, "skipped.tsv" if not a.stats_only else "skipped_stats.tsv"), "w") as f:
        f.writelines("\t".join(r) + "\n" for r in skipped)
    why = {}
    for r in skipped:
        why[r[1]] = why.get(r[1], 0) + 1
    print(f"{len(ids)} chains: {n_ok} usable, {len(skipped)} skipped {why}")
    if lens:
        print(f"usable lengths: min {min(lens)}, median {int(np.median(lens))}, max {max(lens)}")


if __name__ == "__main__":
    main()
