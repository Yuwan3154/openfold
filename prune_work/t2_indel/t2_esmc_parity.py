"""Parity of reduced-precision ESMC against fp32 on the T2 edit plans (env esmfold2_env, GPU). For each chain: the SAME plans and the SAME per-draw sampling rng go through the fp32 and the
reduced-precision model; reports (rows = chain; n = draws) the share of draws whose sampled sequence is identical and the max / mean |logit difference| over the MASKED positions
(the only ones that are sampled). Run: python t2_esmc_parity.py --chains-file ids.txt --natives-dir N --esmc-pth P --dtype bf16 [--n-draws 64]
"""
import argparse
import time

import numpy as np
import torch

from esmc_fill_pilot import AA
from t2_native import load_npz, native_ref
from t2_stage_a import DTYPES, esmc_logits, load_esmc, make_plans, sample_sequences


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains-file", required=True)
    ap.add_argument("--natives-dir", required=True)
    ap.add_argument("--esmc-pth", required=True)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--n-draws", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=64)
    a = ap.parse_args()
    ref, red = load_esmc("esmc_300m", "cuda", a.esmc_pth, "fp32"), load_esmc("esmc_300m", "cuda", a.esmc_pth, a.dtype)
    tok = ref.tokenizer
    ids_of = {c: tok.convert_tokens_to_ids(c) for c in AA}
    tot_same = tot_n = 0
    t_ref = t_red = 0.0
    print(f"rows = chain (n = {a.n_draws} draws); same = share of draws with an identical sampled sequence fp32 vs {a.dtype}; dlogit = |fp32 - {a.dtype}| at the masked positions")
    for ln in open(a.chains_file):
        chain, path = native_ref(ln, a.natives_dir)
        d = load_npz(path)
        plans = make_plans(chain, d["bb"], a.n_draws, 0, 0.05, 0.10, 0.05, 0.10)
        if plans is None:
            continue
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        lr = esmc_logits(ref, tok, ids_of, d["names"], plans, "cuda", a.batch_size)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        ld = esmc_logits(red, tok, ids_of, d["names"], plans, "cuda", a.batch_size)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        t_ref, t_red = t_ref + t1 - t0, t_red + t2 - t1
        sr, sd = sample_sequences(d["names"], chain, plans, lr, 0.7, 1.0), sample_sequences(d["names"], chain, plans, ld, 0.7, 1.0)
        dl = []
        for p, x, y in zip(plans, lr, ld):
            orig = np.asarray(p["orig_idx"])
            m = np.union1d(np.flatnonzero(orig < 0), np.asarray(p["mut_idx"], dtype=int))
            dl.append((x[m] - y[m]).abs().flatten())
        dl = torch.cat(dl)
        same = float(np.mean([u == v for u, v in zip(sr, sd)]))
        tot_same += same * len(plans)
        tot_n += len(plans)
        print(f"{chain} L={len(d['bb'])} same {same:.3f} dlogit max {float(dl.max()):.3f} mean {float(dl.mean()):.4f} fwd fp32 {t1 - t0:.2f}s {a.dtype} {t2 - t1:.2f}s")
    assert tot_n > 0
    print(f"all: {tot_same / tot_n:.3f} of {tot_n} draws identical; forward time fp32 {t_ref:.1f}s vs {a.dtype} {t_red:.1f}s")


if __name__ == "__main__":
    main()
