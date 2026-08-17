"""Fast version of the residue-mapping check: mmCIF seqres only, no template featurization.

The slow version went through OpenFoldSingleDataset.__getitem__, which also runs the hhsearch
template pipeline (4 mmCIF parses per chain) that this question does not need. The query sequence
the T2 hook sees is `data["sequence"]`, which process_mmcif takes from the mmCIF's seqres -- so
parsing seqres directly is the same ground truth at a fraction of the cost.

Verdict criterion: at query position `ridx[j] - 1` the seqres residue must equal the npz aatype[j].
An off-by-one stays in bounds, so identity is the only test that distinguishes them.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from openfold.data import mmcif_parsing
from openfold.data.synthetic_templates import SyntheticTemplatePool
from openfold.np import residue_constants as rc

ap = argparse.ArgumentParser()
ap.add_argument("--index", default="/home/jupyter-chenxi/pp1c_work/index_band.npz")
ap.add_argument("--root", default="/home/jupyter-chenxi/pp1c_work/templates_band")
ap.add_argument("--mmcif-dir", default="/home/jupyter-chenxi/data/pdb_mmcif/mmcif_files")
ap.add_argument("--n", type=int, default=1500)
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--num-shards", type=int, default=1)
a = ap.parse_args()

pool = SyntheticTemplatePool(a.index, a.root, min_tm=0.3, max_tm=0.9)
chains = [c for c in pool.row_of if c in pool]
rng = np.random.default_rng(0)
sample = [chains[i] for i in rng.permutation(len(chains))[: a.n]]
sample = sample[a.shard :: a.num_shards]

exact = oob = seqfail = parsefail = equal_len = 0
best_offset = {0: 0, -1: 0, 1: 0}
failures = []

for n, chain in enumerate(sample):
    file_id, chain_id = chain.rsplit("_", 1)
    path = Path(a.mmcif_dir) / f"{file_id}.cif"
    if not path.is_file():
        parsefail += 1
        continue
    parsed = mmcif_parsing.parse(file_id=file_id, mmcif_string=path.read_text())
    if parsed.mmcif_object is None:
        parsefail += 1
        continue
    seqres = parsed.mmcif_object.chain_to_seqres.get(chain_id)
    if not seqres:
        parsefail += 1
        continue
    qL = len(seqres)
    d = np.load(pool.npz_path(chain), allow_pickle=False)
    ridx = d["residue_index"].astype(int)
    aat = d["aatype"].astype(int)
    letters = np.array([rc.restypes[x] if x < len(rc.restypes) else "X" for x in aat])
    equal_len += (len(ridx) == qL)

    sq = np.array(list(seqres))
    for off in (0, -1, 1):
        p = ridx + off
        if p.min() < 0 or p.max() >= qL:
            continue
        m = (letters != sq[p]) & (letters != "X") & (sq[p] != "X")
        if not m.any():
            best_offset[off] += 1

    p = ridx - 1
    if p.min() < 0 or p.max() >= qL:
        oob += 1
        failures.append((chain, "oob", f"ridx {ridx.min()}-{ridx.max()} vs seqres {qL}"))
        continue
    m = (letters != sq[p]) & (letters != "X") & (sq[p] != "X")
    if m.any():
        seqfail += 1
        failures.append((chain, f"{int(m.sum())}/{len(p)} differ",
                         f"first at q{int(p[m][0])}: npz {letters[m][0]} vs seqres {sq[p][m][0]}"))
    else:
        exact += 1
    if (n + 1) % 200 == 0:
        print(f"  {n+1}/{len(sample)}  exact={exact} oob={oob} seqfail={seqfail}", flush=True)

checked = exact + oob + seqfail
print(f"\nchains checked             : {checked}  (parse failures {parsefail})")
print(f"  npz L == seqres L        : {equal_len}  ({100*equal_len/max(checked,1):.1f}%)")
print(f"  ridx-1 EXACT seq match   : {exact}  ({100*exact/max(checked,1):.2f}%)")
print(f"  ridx-1 out of bounds     : {oob}")
print(f"  ridx-1 sequence mismatch : {seqfail}")
print(f"\nchains matching by offset  : ridx+0 -> {best_offset[0]}, "
      f"ridx-1 -> {best_offset[-1]}, ridx+1 -> {best_offset[1]}")
if failures:
    print(f"\nfirst {min(10, len(failures))} failures:")
    for f in failures[:10]:
        print("  ", f)
print("\nVERDICT:", "ridx-1 IS the query index" if seqfail == 0 and oob == 0
      else f"NOT uniform: {oob + seqfail}/{checked} chains fail")
sys.exit(1 if (oob or seqfail) else 0)
