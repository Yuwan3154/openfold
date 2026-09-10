"""How much template diversity does a SEED buy, versus a RUNG?

Round 2 of the T2 pool must split a fixed per-chain budget between DISTINCT rewind values (rungs)
and REPEATED samples at the same rewind (seeds). Round 1 already ran it both ways, so the answer is
measurable from existing output with no new generation:

  tiered path  (L <= 300, 62,922 chains)  64 distinct rewinds x 1 sample   -- rung axis only
  grouped path (L >  300, 19,811 chains)  rewind_ladder(8, 90, 375)
                                          = [375,334,294,253,212,171,131,90] x 8 samples each
                                                                            -- BOTH axes

This scores every chain's full 64x64 template-vs-template TM matrix and stores the upper triangle.
The reduction that answers the question (done by t2_seed_vs_rung_report.py) is

  within        mean pairwise TM among samples SHARING a rewind      = what a seed buys
  across[d]     mean pairwise TM between rungs d apart on the ladder = what a rung buys

The smallest rung separation worth paying for is the one where across[d] first drops meaningfully
below `within`; any finer spacing buys no more than another seed would, at the same cost.

TM here is template-vs-template on ONE chain, so both structures carry the identical residue set and
`mask` == `norm_mask` == that chain's resolved-CA mask. That differs from build_template_index.py,
which scores template-vs-NATIVE and normalizes by the native's coverage; the numbers are therefore
NOT comparable to index_all.npz's `tm` and are not meant to be.

⛔ This reports the curve. It does NOT pick the interval or the seed count.
"""

from __future__ import annotations

import argparse
import time
import zlib
from pathlib import Path

import numpy as np
import torch

from openfold.utils.tm_score import FAST_KWARGS, REFERENCE_KWARGS, tm_score

PRESETS = {"reference": REFERENCE_KWARGS, "fast": FAST_KWARGS}

CA = 1
# same threshold as build_template_index.py: below 15 residues TM's d0 = 1.24*(L-15)^(1/3) - 1.8 is
# not even real, and 5 is where that file already draws the line
MIN_CA = 5


def npz_path(root: Path, chain: str) -> Path:
    # ⛔ zlib.crc32, NOT builtin hash(): PYTHONHASHSEED randomizes string hashing per process, which
    # silently scattered round-1 output across shard dirs until it was caught by the smoke gate
    return root / f"shard{zlib.crc32(chain.encode()) % 1000:04d}" / f"{chain}.npz"


def pair_tms(ca: torch.Tensor, mask: torch.Tensor, ii: np.ndarray, jj: np.ndarray,
             chunk: int, device: str, kwargs: dict) -> np.ndarray:
    """Upper-triangle pairwise TM over the N templates in `ca` (N,L,3)."""
    out = []
    L = ca.shape[1]
    for s in range(0, len(ii), chunk):
        a = torch.from_numpy(ii[s:s + chunk].astype(np.int64))
        b = torch.from_numpy(jj[s:s + chunk].astype(np.int64))
        n = len(a)
        m = mask[None].expand(n, L).to(device)
        out.append(tm_score(
            ca[a].to(device), ca[b].to(device), mask=m, norm_mask=m, **kwargs,
        ).cpu())
    return torch.cat(out).numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="index_all.npz (the FULL 64-rung index)")
    ap.add_argument("--templates-root", required=True, help="the FULL 64-rung tree, not the band one")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-distinct", type=int, default=8,
                    help="select chains whose ladder has exactly this many distinct rewind values. "
                         "8 = the grouped path (the only population with a seed axis); 64 = tiered.")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-chains", type=int, default=0, help="0 = no cap; >0 for a smoke run")
    ap.add_argument("--tm-preset", choices=sorted(PRESETS), default="reference",
                    help="tm_score accuracy-vs-speed knobs. reference = what every other TM number "
                         "in this project uses. fast drops the sub-32-residue Zhang-Skolnick seeds, "
                         "which on an L>300 chain are ~90%% of the seeds and cannot win anyway. "
                         "⛔ Only use fast after --validate-presets shows the two agree HERE.")
    ap.add_argument("--validate-presets", action="store_true",
                    help="score the same pairs under BOTH presets and report the agreement, "
                         "including whether the within-rung vs across-rung contrast is preserved. "
                         "Writes nothing; this is a gate, not a measurement.")
    a = ap.parse_args()

    z = np.load(a.index, allow_pickle=False)
    chains, rewind, length = [str(c) for c in z["chains"]], z["rewind"], z["length"]
    keep = [i for i in range(len(chains)) if len(np.unique(rewind[i])) == a.n_distinct]
    keep = keep[a.shard::a.num_shards]
    if a.max_chains:
        keep = keep[:a.max_chains]
    print(f"shard {a.shard}/{a.num_shards}: {len(keep)} chains with {a.n_distinct} distinct rewinds",
          flush=True)

    root = Path(a.templates_root)
    n_t = rewind.shape[1]
    ii, jj = (x.astype(np.int32) for x in np.triu_indices(n_t, 1))

    names, tms, rws, lens, ncas, skipped = [], [], [], [], [], []
    for c, i in enumerate(keep):
        chain = chains[i]
        p = npz_path(root, chain)
        if not p.exists():
            skipped.append(f"{chain}:missing_npz")
            continue
        d = np.load(p, allow_pickle=False)
        atom_mask = d["atom_mask"]
        L = atom_mask.shape[0]
        ca_mask = atom_mask[:, CA]
        # skip-and-record, never assert: one degenerate chain must not cost the whole shard
        if int(ca_mask.sum()) < MIN_CA:
            skipped.append(f"{chain}:only_{int(ca_mask.sum())}_ca")
            continue
        full = np.zeros((n_t, L, 37, 3), np.float32)
        full[:, atom_mask] = d["coords"]
        ca = torch.from_numpy(full[:, :, CA, :])
        mask = torch.from_numpy(ca_mask.astype(np.float32))

        if a.validate_presets:
            rw_c = d["rewind_steps"]
            same = rw_c[ii] == rw_c[jj]
            t0 = time.perf_counter()
            ref = pair_tms(ca, mask, ii, jj, a.chunk, a.device, PRESETS["reference"])
            t_ref = time.perf_counter() - t0
            t0 = time.perf_counter()
            fst = pair_tms(ca, mask, ii, jj, a.chunk, a.device, PRESETS["fast"])
            t_fst = time.perf_counter() - t0
            print(f"{chain} L={L} n_ca={int(ca_mask.sum())}  "
                  f"reference {t_ref:.1f}s  fast {t_fst:.1f}s  speedup {t_ref / t_fst:.1f}x")
            print(f"   per-pair  max|diff| {np.abs(ref - fst).max():.4f}  "
                  f"mean|diff| {np.abs(ref - fst).mean():.4f}  "
                  f"pearson {np.corrcoef(ref, fst)[0, 1]:.6f}")
            print(f"   WITHIN-rung mean  reference {ref[same].mean():.4f}  fast {fst[same].mean():.4f}"
                  f"   (diff {fst[same].mean() - ref[same].mean():+.4f})")
            print(f"   ACROSS-rung mean  reference {ref[~same].mean():.4f}  fast {fst[~same].mean():.4f}"
                  f"   (diff {fst[~same].mean() - ref[~same].mean():+.4f})")
            print(f"   ⭐ THE CONTRAST (within - across)  reference "
                  f"{ref[same].mean() - ref[~same].mean():+.4f}  fast "
                  f"{fst[same].mean() - fst[~same].mean():+.4f}", flush=True)
            continue

        names.append(chain)
        tms.append(pair_tms(ca, mask, ii, jj, a.chunk, a.device, PRESETS[a.tm_preset]))
        rws.append(d["rewind_steps"].astype(np.int16))
        lens.append(np.int32(length[i]))
        ncas.append(np.int32(ca_mask.sum()))
        if (c + 1) % 50 == 0:
            print(f"  {c + 1}/{len(keep)}", flush=True)

    if a.validate_presets:                 # a gate, not a measurement: there is nothing to write
        return

    np.savez(
        a.out,
        chains=np.array(names, dtype="<U8"),
        tm_pairs=np.stack(tms) if tms else np.zeros((0, len(ii)), np.float32),
        rewind=np.stack(rws) if rws else np.zeros((0, n_t), np.int16),
        length=np.array(lens, np.int32),
        n_ca=np.array(ncas, np.int32),
        pair_i=ii, pair_j=jj,
        skipped=np.array(skipped, dtype="<U64"),
    )
    print(f"wrote {a.out}: {len(names)} chains, {len(skipped)} skipped", flush=True)


if __name__ == "__main__":
    main()
