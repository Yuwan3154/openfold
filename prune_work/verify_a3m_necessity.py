"""Does the no-MSA recipe actually USE the a3m files, or only `pdb70_hits.hhr`?

⛔ WHY THIS DECIDES A DEPLOYMENT, NOT JUST AN OPTIMISATION. The a3m files are 89% of the 601 GB
alignment tree (measured: uniref90 348 GB + mgnify 121 GB + bfd_uniclust 113 GB vs pdb70_hits.hhr
69 GB). Engaging's SCRATCH has ~361 GB free, so shipping the a3m does not fit and shipping hhr-only
does. But `--enable_single_seq_mode` only clamps `max_msa_clusters=1` / `max_extra_msa=1`; it does NOT
set `config.model.extra_msa.enabled=False`. So one real homolog row may still reach the ExtraMSA
stack, in which case dropping the a3m changes the model's input and breaks comparability with T1/T2 --
which is the whole point of the run.

WHAT IS COMPARED, and why it is not a blanket tensor diff. `sample_msa` calls `randperm(num_seq)`, and
num_seq differs between the two conditions, so the torch RNG stream DIVERGES and a naive diff would
flag everything downstream (crop offsets, template subsampling) as "different" while telling us nothing
about information content. Instead this asks the two questions that actually decide the matter:
  1. RAW (deterministic, no RNG): what does `process_mmcif` produce differently? Only the `msa` block
     should change; template features come from the hhr and coordinates from the mmCIF.
  2. DELIVERED: with the clamps applied, is the row that lands in `extra_msa` a REAL homolog, or just
     the query / padding? If it is never a real homolog, the a3m is genuinely dead weight here.
"""

import argparse
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch

from openfold.config import model_config
from openfold.data import data_transforms  # noqa: F401  (import side effects match training)
from openfold.data.data_modules import OpenFoldSingleDataset

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", required=True)
ap.add_argument("--aln-dir", required=True)
ap.add_argument("--chain-list", required=True)
ap.add_argument("--mmcif-cache", required=True)
ap.add_argument("--obsolete", required=True)
ap.add_argument("--kalign", required=True)
ap.add_argument("--template-cache", required=True)
ap.add_argument("--work-dir", required=True, help="where the hhr-only mirror is built")
ap.add_argument("--n-chains", type=int, default=12)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

# ---- config: exactly what run_stock_af2_nomsa_t2.sh produces -------------------------------------
cfg = model_config("finetuning_ptm", train=True, low_prec=True)
cfg.data.common.max_extra_msa = 1
cfg.data.common.max_msa_clusters = 1
cfg.data.train.max_extra_msa = 1
cfg.data.train.max_msa_clusters = 1
cfg.loss.masked_msa.weight = 0.0            # --enable_single_seq_mode, templates KEPT
cfg.data.train.crop_size = min(cfg.data.train.crop_size, 256)
print(f"config: max_msa_clusters={cfg.data.train.max_msa_clusters} "
      f"max_extra_msa={cfg.data.train.max_extra_msa} "
      f"template.enabled={cfg.model.template.enabled} "
      f"extra_msa.enabled={cfg.model.extra_msa.enabled}")
assert cfg.model.extra_msa.enabled, "if this were False the a3m could not matter at all"

chains = [l.strip() for l in open(a.chain_list) if l.strip()]
random.seed(a.seed)
pick = random.sample(chains, a.n_chains)

# ---- build an hhr-only mirror of the alignment dir ------------------------------------------------
work = Path(a.work_dir)
if work.exists():
    shutil.rmtree(work)
work.mkdir(parents=True)
for c in pick:
    src, dst = Path(a.aln_dir) / c, work / c
    dst.mkdir()
    hhr = src / "pdb70_hits.hhr"
    if hhr.is_file():
        os.symlink(hhr, dst / "pdb70_hits.hhr")
    n_a3m = len(list(src.glob("*.a3m")))
    print(f"  {c}: hhr={hhr.is_file()} a3m_files_left_behind={n_a3m}")


def make_ds(aln, raw):
    return OpenFoldSingleDataset(
        data_dir=a.data_dir, alignment_dir=str(aln), template_mmcif_dir=a.data_dir,
        max_template_date="2018-04-30", config=cfg.data, chain_data_cache_path=None,
        kalign_binary_path=a.kalign, max_template_hits=cfg.data.train.max_template_hits,
        shuffle_top_k_prefiltered=cfg.data.train.shuffle_top_k_prefiltered,
        template_release_dates_cache_path=a.template_cache,
        obsolete_pdbs_file_path=a.obsolete, mode="train", _output_raw=raw,
        chain_list_path=a.chain_list,
    )


full_raw, hhr_raw = make_ds(a.aln_dir, True), make_ds(work, True)
full_fp, hhr_fp = make_ds(a.aln_dir, False), make_ds(work, False)
idx_of = {full_raw.idx_to_chain_id(i): i for i in range(len(full_raw))}

print("\n================ PART 1: RAW features (deterministic apart from the hit shuffle) ============")
depths = []
for c in pick:
    i = idx_of[c]
    np.random.seed(a.seed); torch.manual_seed(a.seed)
    A = full_raw[i]
    np.random.seed(a.seed); torch.manual_seed(a.seed)
    B = hhr_raw[i]
    diff = []
    for k in sorted(set(A) | set(B)):
        if k not in A or k not in B:
            diff.append(f"{k}:MISSING")
            continue
        x, y = np.asarray(A[k]), np.asarray(B[k])
        if x.shape != y.shape:
            diff.append(f"{k}:shape {x.shape}->{y.shape}")
        elif x.dtype == object:
            if not (x == y).all():
                diff.append(f"{k}:objdiff")
        elif not np.array_equal(x, y):
            diff.append(f"{k}:values")
    depths.append((c, int(np.asarray(A["msa"]).shape[0]), int(np.asarray(B["msa"]).shape[0])))
    print(f"  {c}: msa depth {depths[-1][1]} -> {depths[-1][2]}; differing keys: "
          f"{diff if diff else 'NONE'}")

print(f"\n  a3m MSA depth: median {int(np.median([d[1] for d in depths]))}, "
      f"min {min(d[1] for d in depths)}, max {max(d[1] for d in depths)}  ->  hhr-only always "
      f"{sorted({d[2] for d in depths})}")

print("\n================ PART 2: what actually reaches the model =====================================")
print("  extra_msa row identity: is it a REAL homolog, or the query itself?")
n_real, n_query, n_total = 0, 0, 0
for c in pick:
    i = idx_of[c]
    np.random.seed(a.seed); torch.manual_seed(a.seed)
    FA = full_fp[i]
    np.random.seed(a.seed); torch.manual_seed(a.seed)
    HB = hhr_fp[i]
    row = {}
    for tag, F in (("a3m", FA), ("hhr", HB)):
        ex = F.get("extra_msa")
        aat = F["aatype"]
        if ex is None:
            row[tag] = "extra_msa ABSENT"
            continue
        e0 = ex[..., 0] if ex.dim() > 2 else ex          # strip the recycling dim
        e0 = e0[0] if e0.dim() > 1 else e0
        q = aat[..., 0] if aat.dim() > 1 else aat
        same = bool((e0[: q.shape[0]] == q).all())
        row[tag] = (f"shape {tuple(ex.shape)} identical_to_query={same} "
                    f"n_distinct_res={len(torch.unique(e0))}")
        if tag == "a3m":
            n_total += 1
            n_real += (not same)
            n_query += same
    print(f"  {c}\n     with a3m : {row['a3m']}\n     hhr only : {row['hhr']}")
    for k in ("msa_feat", "extra_msa", "msa_mask", "true_msa"):
        if k in FA and k in HB:
            x, y = FA[k].float(), HB[k].float()
            tag = ("shape " + str(tuple(x.shape)) + " vs " + str(tuple(y.shape))
                   if x.shape != y.shape else
                   ("IDENTICAL" if torch.equal(x, y) else f"max|d|={(x - y).abs().max():.4g}"))
            print(f"     {k:12s} {tag}")

print(f"\nVERDICT INPUT: with the a3m present, extra_msa carried a REAL homolog in {n_real}/{n_total} "
      f"chains and merely the query in {n_query}/{n_total}.")
print("  -> if n_real > 0 the a3m DOES change the model input, so hhr-only is NOT comparable to T1/T2.")
shutil.rmtree(work)
