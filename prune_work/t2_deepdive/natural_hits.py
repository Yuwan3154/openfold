"""Which natural (hhsearch/PDB) templates training delivers for a chain sample, via the REAL featurizer.

Reuses count_natural_templates.py's dataset (T1's preset, paths, 2018-04-30 cutoff, cap 4) and its
seed-0 400-chain sample, and records each chain's template_domain_names (the pdbid_chain of every hit
that survived the featurizer). Env: cue_openfold_gated with <repo> and <repo>/openfold on PYTHONPATH.

Run: python natural_hits.py --out natural_hits.json [--workers 6]
"""
import argparse
import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import count_natural_templates as cnt  # noqa: E402


def _worker(idx):
    ds = cnt._dataset()
    raw = ds[idx]
    names = [n.decode() if isinstance(n, bytes) else str(n) for n in raw.get("template_domain_names", [])]
    return ds.idx_to_chain_id(idx), names


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--n-chains", type=int, default=400)  # count_natural_templates.py's sample
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    n_total = len(cnt._dataset())
    random.seed(a.seed)
    idxs = random.sample(range(n_total), min(a.n_chains, n_total))  # identical draw to count_natural_templates
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, (chain, names) in enumerate(ex.map(_worker, idxs, chunksize=4)):
            rows.append({"chain": chain, "hits": names})
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(idxs)}", flush=True)
    json.dump(rows, open(a.out, "w"))
    print(f"wrote {len(rows)} chains, {sum(len(r['hits']) for r in rows)} hits to {a.out}")


if __name__ == "__main__":
    main()
