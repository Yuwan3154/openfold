"""TOTAL natural template hits per training chain -- how much room the synthetic ones take up.

The earlier measurement reported the count AFTER `max_template_hits=4`, which is just the cap.
This reports the whole funnel, because three different numbers matter for the mixing question:

  RAW        every hit in pdb70_hits.hhr.
  PREFILTERED  hits surviving the real prefilter (`_prefilter_hit`: release-date cutoff,
               align-ratio, exact-subsequence, length). This is the honest "how many natural
               templates does this chain HAVE".
  VARIETY    min(PREFILTERED, shuffle_top_k_prefiltered=20). The featurizer sorts by sum_probs and
             shuffles only the TOP 20 before taking hits, so across epochs a chain can ever show at
             most its top 20 -- hit 21+ is unreachable no matter how long training runs.
  PER STEP   <= max_template_hits = 4.

⭐ No mmCIF and no kalign here: `_prefilter_hit` needs only the hit, the release-date cache and the
query sequence (read from the first record of uniref90_hits.a3m), so this is ~1000x cheaper than
running the featurizer and can cover a big sample.
"""

import argparse
import datetime
import json
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HOME = Path("/home/jupyter-chenxi")
ALN = HOME / "data/openproteinset_aln"
CUTOFF = datetime.datetime(2018, 4, 30)          # T1's max_template_date
TOP_K = 20                                       # config.train.shuffle_top_k_prefiltered
MAX_HITS = 4                                     # config.train.max_template_hits
_STATE = {}


def _init():
    if _STATE:
        return _STATE
    from openfold.data import templates
    from openfold.data.templates import _parse_release_dates, _parse_obsolete
    _STATE["templates"] = templates
    _STATE["release"] = _parse_release_dates(str(HOME / "data/pdb_mmcif/mmcif_cache.json"))
    _STATE["obsolete"] = _parse_obsolete(str(HOME / "data/pdb_mmcif/obsolete.dat"))
    return _STATE


def _query_seq(d: Path) -> str:
    """First record of the uniref90 a3m is the query itself."""
    a3m = d / "uniref90_hits.a3m"
    if not a3m.is_file():
        return ""
    seq = []
    with open(a3m) as fh:
        first = fh.readline()
        if not first.startswith(">"):
            return ""
        for line in fh:
            if line.startswith(">"):
                break
            seq.append(line.strip())
    return "".join(seq)


def _worker(chain):
    st = _init()
    from openfold.data import parsers
    d = ALN / chain
    hhr = d / "pdb70_hits.hhr"
    if not hhr.is_file():
        return chain, 0, 0
    hits = parsers.parse_hhr(hhr.read_text(errors="ignore"))
    q = _query_seq(d)
    if not q:
        return chain, len(hits), -1
    n_ok = 0
    for h in hits:
        r = st["templates"]._prefilter_hit(
            query_sequence=q, hit=h, max_template_date=CUTOFF,
            release_dates=st["release"], obsolete_pdbs=st["obsolete"],
        )
        n_ok += bool(r.valid)
    return chain, len(hits), n_ok


def _pct(a, p):
    return int(np.percentile(a, p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-chains", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HOME / "prune_work/raw_template_hits.json"))
    a = ap.parse_args()

    chains = [l.strip() for l in
              open(HOME / "prune_work/lists_pdb/slim_struct_train.list") if l.strip()]
    random.seed(a.seed)
    pick = random.sample(chains, min(a.n_chains, len(chains)))
    print(f"{len(chains)} training chains; sampling {len(pick)} with {a.workers} workers",
          flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(_worker, pick, chunksize=8)):
            rows.append(r)
            if (i + 1) % 250 == 0:
                print(f"  {i+1}/{len(pick)}", flush=True)

    raw = np.array([r[1] for r in rows])
    filt = np.array([r[2] for r in rows if r[2] >= 0])
    print(f"\nRAW hits in pdb70_hits.hhr   (n={len(raw)})")
    print(f"  min {raw.min()}  p25 {_pct(raw,25)}  median {_pct(raw,50)}  "
          f"p75 {_pct(raw,75)}  p95 {_pct(raw,95)}  max {raw.max()}  mean {raw.mean():.0f}")
    print(f"\nPREFILTERED (date cutoff {CUTOFF:%Y-%m-%d} + align/length/dup)   (n={len(filt)})")
    print(f"  min {filt.min()}  p25 {_pct(filt,25)}  median {_pct(filt,50)}  "
          f"p75 {_pct(filt,75)}  p95 {_pct(filt,95)}  max {filt.max()}  mean {filt.mean():.0f}")
    band = Counter()
    for n in filt:
        band["0" if n == 0 else "1-3" if n < 4 else "4-19" if n < 20 else ">=20"] += 1
    print("\n  distribution:")
    for k in ("0", "1-3", "4-19", ">=20"):
        print(f"    {k:>5}: {band[k]:5d}  {100*band[k]/len(filt):5.1f}%")

    variety = np.minimum(filt, TOP_K)
    print(f"\nREACHABLE VARIETY = min(prefiltered, shuffle_top_k={TOP_K})")
    print(f"  median {_pct(variety,50)}  mean {variety.mean():.1f}   "
          f"(chains capped by the top-{TOP_K} shuffle: "
          f"{100*(filt > TOP_K).mean():.1f}%)")
    print(f"\nPER STEP the model still sees at most max_template_hits={MAX_HITS}.")

    json.dump({"chains": [r[0] for r in rows], "raw": raw.tolist(),
               "prefiltered": [int(x) for x in filt]}, open(a.out, "w"))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
