"""Pick the noise-sweep target subset and stage exactly the cif files it needs.

Stratified over (length bin) x (stock-AF2 pass/fail), 8 per cell = 64. Length is the PDA set's
dominant covariate (A4) and the pass/fail axis is there so the sweep can answer "do hot rungs help
where the model currently fails" and not only "what is the average diversity".

⭐ The monomer/denovo annotations ride along in the manifest so the ANALYSIS can subset (well-posed
monomers only, de novo only) without re-running a single forward. Measure broadly, subset later.
"""
import csv
import json
import os
import shutil

ROOT = "/home/jupyter-chenxi/prune_work"
PER_CELL = 8

ann = {}
with open(f"{ROOT}/pda_a4_final_table.csv") as fh:
    for r in csv.DictReader(fh):
        ann[(r["pdb"].lower(), r["cid"])] = r
base = {}
with open(f"{ROOT}/eval_out/pda_baseline_full/pda_baseline_full.csv") as fh:
    for r in csv.DictReader(fh):
        base[(r["pdb"].lower(), r["chain_id"])] = r
manifest = json.load(open(f"{ROOT}/eval_out/pda_cluster_representatives.json"))
by_key = {(e["pdb"].lower(), e["chain_id"]): e for e in manifest}
print(f"manifest {len(manifest)}  annotated {len(ann)}  baseline {len(base)}")


def lbin(L):
    L = int(L)
    return "L<=30" if L <= 30 else "L31-50" if L <= 50 else "L51-150" if L <= 150 else "L>150"


cells = {}
for k in sorted(by_key):
    if k not in ann or k not in base:
        continue
    cell = (lbin(ann[k]["L"]), "FAIL" if base[k]["success_2A_stock"] == "False" else "PASS")
    cells.setdefault(cell, []).append(k)

picked, sub = [], []
for cell in sorted(cells):
    # deterministic: shortest-first within the cell, no RNG, so the subset is reproducible
    pool = sorted(cells[cell], key=lambda k: (int(ann[k]["L"]), k))
    step = max(len(pool) // PER_CELL, 1)
    take = [pool[i * step] for i in range(min(PER_CELL, len(pool)))]
    print(f"  {cell[0]:<8} {cell[1]}  pool {len(pool):>3} -> took {len(take)}")
    for k in take:
        picked.append(k)
        e = dict(by_key[k])
        e.update(length=int(ann[k]["L"]), denovo=ann[k]["denovo"], oligo=ann[k]["oligo"],
                 stock_fail=base[k]["success_2A_stock"] == "False",
                 stock_lddt=round(float(base[k]["lddt_ca_stock"]), 4))
        sub.append(e)

out = "/home/jupyter-chenxi/sweep_stage"
os.makedirs(f"{out}/cif", exist_ok=True)
json.dump(sub, open(f"{out}/sweep_manifest.json", "w"), indent=1)
n = 0
for k in picked:
    src = f"{ROOT}/eval_out/pda_mmcif_cache/{k[0]}.cif"
    if os.path.isfile(src):
        shutil.copy(src, f"{out}/cif/{k[0]}.cif")
        n += 1
    else:
        print(f"  ⛔ MISSING cif: {src}")
print(f"\n{len(sub)} targets, {n} cif files staged -> {out}")
print(f"  lengths {min(e['length'] for e in sub)}-{max(e['length'] for e in sub)}, "
      f"fail {sum(e['stock_fail'] for e in sub)}/{len(sub)}, "
      f"monomer {sum(e['oligo'] == 'MONOMER' for e in sub)}")
print(f"  cif dir size: {sum(os.path.getsize(f'{out}/cif/{f}') for f in os.listdir(f'{out}/cif'))/1e6:.0f} MB")
