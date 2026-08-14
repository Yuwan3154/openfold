"""How many NATURAL (hhsearch/PDB) templates does a training chain actually get?

Answers the user's question ahead of choosing the synthetic:natural mixing ratio for T2.

⛔ Measured through the REAL reader -- `OpenFoldSingleDataset` in train mode with T1's exact config
preset, paths and date cutoff -- not by counting raw .hhr hits, which are filtered afterwards by
release date, resolution and the max_template_date cutoff.

Two numbers matter and only the first needs measuring:

  AVAILABLE  = hits surviving the featurizer, capped at `max_template_hits` (4). Measured here.
  DELIVERED  = what reaches the model on a given step. In train mode `subsample_templates=True`, so
               `random_crop_to_size` draws `templates_crop_start ~ Uniform{0..num_templates}`
               (INCLUSIVE) and takes `min(num_templates - start, max_templates)`. Since the
               featurizer already caps at 4 = max_templates, that reduces exactly to
                   delivered ~ Uniform{0, 1, ..., available}
               -- so the mean delivered is available/2 and the model is handed ZERO templates with
               probability 1/(available+1). Derived, not simulated: running the feature pipeline per
               draw costs ~10 s and adds nothing.

CPU-only, nice'd, multi-process -- must not disturb T1 on the GPUs.
"""

import argparse
import json
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

HOME = "/home/jupyter-chenxi"
COMMON = dict(
    data_dir=f"{HOME}/data/pdb_mmcif/mmcif_files",
    alignment_dir=f"{HOME}/data/openproteinset_aln",
    template_mmcif_dir=f"{HOME}/data/pdb_mmcif/mmcif_files",
    max_template_date="2018-04-30",                        # T1's cutoff
    kalign_binary_path=f"{HOME}/miniconda3/envs/cue_openfold_gated/bin/kalign",
    obsolete_pdbs_file_path=f"{HOME}/data/pdb_mmcif/obsolete.dat",
    template_release_dates_cache_path=f"{HOME}/data/pdb_mmcif/mmcif_cache.json",
    chain_list_path=f"{HOME}/prune_work/lists_pdb/slim_struct_train.list",
    max_template_hits=4,
    mode="train",
)
_DS = None


def _dataset():
    global _DS
    if _DS is None:
        from openfold.config import model_config
        from openfold.data import data_modules as dm
        # ⛔ chain_list_path is honoured ONLY when block_replacement_scripts.enhanced_data_utils
        # imports (needs <repo>/openfold on PYTHONPATH). Without it the dataset silently becomes
        # os.listdir(alignment_dir) = 133,019 chains and every number here describes the wrong set.
        assert dm.ENHANCED_UTILS_AVAILABLE, "enhanced_data_utils did not import -- list IGNORED"
        cfg = model_config("finetuning_ptm", train=True)    # T1's preset
        _DS = dm.OpenFoldSingleDataset(config=cfg.data, _output_raw=True, **COMMON)
        assert len(_DS) == 88155, f"expected the 88155-chain training list, got {len(_DS)}"
    return _DS


def _worker(idx):
    ds = _dataset()
    raw = ds[idx]
    n = int(raw["template_aatype"].shape[0]) if "template_aatype" in raw else 0
    return ds.idx_to_chain_id(idx), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-chains", type=int, default=400)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=f"{HOME}/prune_work/natural_template_counts.json")
    a = ap.parse_args()

    n_total = len(_dataset())
    random.seed(a.seed)
    idxs = random.sample(range(n_total), min(a.n_chains, n_total))
    print(f"{n_total} training chains; sampling {len(idxs)} with {a.workers} workers", flush=True)

    rows, avail = [], Counter()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, (chain, n) in enumerate(ex.map(_worker, idxs, chunksize=4)):
            rows.append({"chain": chain, "available": n})
            avail[n] += 1
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(idxs)}", flush=True)

    n_c = sum(avail.values())
    print(f"\nAVAILABLE natural templates per chain (cap {COMMON['max_template_hits']}), n={n_c}")
    for k in sorted(avail):
        print(f"  {k}: {avail[k]:5d}  {100*avail[k]/n_c:5.1f}%")
    mean_a = sum(k * v for k, v in avail.items()) / n_c
    print(f"  mean {mean_a:.2f}")

    # delivered ~ Uniform{0..available}
    deliv = Counter()
    for k, v in avail.items():
        for d in range(k + 1):
            deliv[d] += v / (k + 1)
    n_d = sum(deliv.values())
    print(f"\nDELIVERED per training step (derived: Uniform{{0..available}})")
    for k in sorted(deliv):
        print(f"  {k}: {100*deliv[k]/n_d:5.1f}%")
    print(f"  mean {sum(k*v for k, v in deliv.items())/n_d:.2f}   "
          f"steps with ZERO template {100*deliv[0]/n_d:.1f}%")

    json.dump({"available": dict(avail), "delivered": {str(k): v for k, v in deliv.items()},
               "per_chain": rows}, open(a.out, "w"))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
