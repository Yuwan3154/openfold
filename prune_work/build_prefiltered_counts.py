"""Per-chain count of PREFILTERED natural template hits, for every training chain.

⛔⛔ WHY AN OFFLINE TABLE AND NOT A CODE CHANGE. The user's top-up rule needs the number of hits that
survive `_prefilter_hit`, because that is the pool `shuffle_top_k_prefiltered=20` truncates. That
number is computed deep inside `HhsearchHitFeaturizer.get_templates` and is NOT in the features it
returns, so the T2 hook (which runs on the featurizer's OUTPUT) cannot see it. The alternatives were
to thread it out through `TemplateSearchResult` or to stash it on the featurizer instance -- both edit
a core path that every run in this project shares, for a number that is completely static: the
prefilter depends only on the hit, the fixed 2018-04-30 release-date cutoff and the query sequence.
So it is computed ONCE here and looked up, exactly as the qmap is.

⭐ ~1000x cheaper than the featurizer: `_prefilter_hit` needs no mmCIF and no kalign (see
`count_raw_template_hits.py`, which established this), so the whole 88155-chain list is CPU-hours, not
GPU-days, and is shardable.

⚠️ The count is only valid for the cutoff it was built with. It is written into the npz and the
consumer asserts on it, rather than trusting the filename.
"""

import argparse
import datetime
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

_STATE = {}


def _init(mmcif_cache, obsolete):
    if _STATE:
        return _STATE
    from openfold.data import templates
    from openfold.data.templates import _parse_release_dates, _parse_obsolete
    _STATE["templates"] = templates
    _STATE["release"] = _parse_release_dates(mmcif_cache)
    _STATE["obsolete"] = _parse_obsolete(obsolete)
    return _STATE


def _query_seq(d: Path) -> str:
    """First record of the uniref90 a3m is the query itself."""
    a3m = d / "uniref90_hits.a3m"
    if not a3m.is_file():
        return ""
    seq = []
    with open(a3m) as fh:
        if not fh.readline().startswith(">"):
            return ""
        for line in fh:
            if line.startswith(">"):
                break
            seq.append(line.strip())
    return "".join(seq)


_CFG = {}


def _worker(chain):
    st = _init(_CFG["mmcif_cache"], _CFG["obsolete"])
    from openfold.data import parsers
    d = Path(_CFG["aln"]) / chain
    hhr = d / "pdb70_hits.hhr"
    if not hhr.is_file():
        return chain, -1                       # no hhsearch output at all: distinct from "0 passed"
    q = _query_seq(d)
    if not q:
        return chain, -2                       # no query sequence: cannot prefilter, distinct again
    n = 0
    for h in parsers.parse_hhr(hhr.read_text(errors="ignore")):
        if st["templates"]._prefilter_hit(
            query_sequence=q, hit=h, max_template_date=_CFG["cutoff"],
            release_dates=st["release"], obsolete_pdbs=st["obsolete"],
        ).valid:
            n += 1
    return chain, n


def _set_cfg(cfg):
    _CFG.update(cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain-list", required=True)
    ap.add_argument("--aln-dir", required=True)
    ap.add_argument("--mmcif-cache", required=True)
    ap.add_argument("--obsolete", required=True)
    ap.add_argument("--cutoff", default="2018-04-30", help="must match --max_template_date")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cutoff = datetime.datetime.strptime(a.cutoff, "%Y-%m-%d")
    cfg = {"aln": a.aln_dir, "mmcif_cache": a.mmcif_cache, "obsolete": a.obsolete,
           "cutoff": cutoff}
    _set_cfg(cfg)
    chains = [l.strip() for l in open(a.chain_list) if l.strip()]
    print(f"{len(chains)} chains, {a.workers} workers, cutoff {a.cutoff}", flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_set_cfg, initargs=(cfg,)) as ex:
        for i, r in enumerate(ex.map(_worker, chains, chunksize=16)):
            rows.append(r)
            if (i + 1) % 2000 == 0:
                ok = [x[1] for x in rows if x[1] >= 0]
                print(f"  {i+1}/{len(chains)}  median so far "
                      f"{int(np.median(ok)) if ok else 0}", flush=True)

    names = np.array([r[0] for r in rows])
    counts = np.array([r[1] for r in rows], np.int32)
    good = counts[counts >= 0]
    print(f"\nprefiltered counts (n={len(good)} usable, "
          f"{int((counts == -1).sum())} without hhr, {int((counts == -2).sum())} without a query seq)")
    for lo, hi, lab in [(0, 1, "0"), (1, 4, "1-3"), (4, 20, "4-19"), (20, 10 ** 9, ">=20")]:
        n = int(((good >= lo) & (good < hi)).sum())
        print(f"  {lab:>5}: {n:6d}  {100*n/max(1,len(good)):5.2f}%")
    print(f"  median {int(np.median(good))}  mean {good.mean():.1f}  max {int(good.max())}")
    np.savez(a.out, chains=names, n_prefiltered=counts,
             cutoff=np.array(a.cutoff), chain_list=np.array(a.chain_list))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
