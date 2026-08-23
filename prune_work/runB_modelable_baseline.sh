#!/bin/bash
. /home/jupyter-chenxi/miniconda3/etc/profile.d/conda.sh
conda activate cue_openfold_gated
python3 - <<'PY'
import csv, json
import statistics as st
from collections import defaultdict

E = "/home/jupyter-chenxi/prune_work/eval_out"
full = json.load(open(f"{E}/pda_cluster_representatives.json"))
mdl = json.load(open(f"{E}/pda_cluster_representatives_modelable.json"))

def key(e): return (e["pdb"].lower(), e["chain_id"])
full_order = [key(e) for e in full]
mdl_set = {key(e) for e in mdl}
print(f"full manifest {len(full_order)}, modelable {len(mdl_set)}")
overlap = sum(1 for k in full_order if k in mdl_set)
print(f"modelable entries found in the 425 order: {overlap}")

idx_in_mdl = {i for i, k in enumerate(full_order) if k in mdl_set}

p = "/home/jupyter-chenxi/runs/runB_full_stack_pda_eval/lightning_logs/version_1/per_entry_val_history.csv"
# dedupe the DDP padding: keep the FIRST row per (epoch, batch_idx)
seen, per = set(), defaultdict(lambda: defaultdict(list))
for r in csv.DictReader(open(p)):
    ep, bi = int(r["epoch"]), int(r["batch_idx"])
    if (ep, bi) in seen:
        continue
    seen.add((ep, bi))
    for m in ("lddt_ca", "recall_2A", "gdt_ts", "alignment_rmsd"):
        per[ep][m].append((bi, float(r[m])))

print(f"\n{'ep':>3} {'metric':16s} {'ALL-425':>10} {'MODELABLE-306':>14} {'n_mdl':>6}")
for ep in sorted(per):
    if ep not in (10, 15, 16):
        continue
    for m in ("lddt_ca", "recall_2A", "gdt_ts", "alignment_rmsd"):
        vals = per[ep][m]
        allv = [v for _, v in vals]
        mdlv = [v for bi, v in vals if bi in idx_in_mdl]
        print(f"{ep:>3} {m:16s} {st.fmean(allv):10.4f} {st.fmean(mdlv):14.4f} {len(mdlv):6d}")
    print()
PY
