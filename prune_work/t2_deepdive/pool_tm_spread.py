"""What TM-to-native distribution does training actually draw synthetic templates from?

SyntheticTemplatePool.sample_features picks uniformly (without replacement) among a chain's rungs
with min_tm < tm < max_tm, so the per-draw TM distribution is the chain-weighted mix of each
chain's in-band rungs. Reports that mix, the per-chain in-band span, and the rung-count per bin.

Run: <proteinebm env>/bin/python pool_tm_spread.py --index ~/pp1c_work/index_band.npz --out-json out.json
"""
import argparse
import json

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--index", required=True)
p.add_argument("--out-json", required=True)
a = p.parse_args()

z = np.load(a.index, allow_pickle=False)
print("keys:", z.files, {k: z[k].shape for k in z.files})
tm = z["tm"].astype(np.float64)
lo, hi = float(z["min_tm"]), float(z["max_tm"])  # the band the pool was pruned to
band = (tm > lo) & (tm < hi)
n_in = band.sum(1)
has = n_in > 0
print(f"band ({lo},{hi}); chains {len(tm)}, with >=1 in-band rung {has.sum()}; "
      f"in-band rungs/chain mean {n_in[has].mean():.1f} median {np.median(n_in[has]):.0f}")

# per-draw distribution: each chain equally likely, each of its in-band rungs equally likely
edges = np.round(np.arange(lo, hi + 1e-9, 0.1), 2)
w = np.where(band, 1.0 / np.maximum(n_in, 1)[:, None], 0.0)
draw_hist, _ = np.histogram(tm[band], bins=edges, weights=w[band])
draw_hist = draw_hist / draw_hist.sum()
rung_hist, _ = np.histogram(tm[band], bins=edges)

tm_in = np.where(band, tm, np.nan)
cmin, cmax = np.nanmin(tm_in[has], 1), np.nanmax(tm_in[has], 1)
span = cmax - cmin
q = [0.05, 0.25, 0.5, 0.75, 0.95]
out = {
    "index": a.index, "band": [lo, hi], "n_chains": int(len(tm)), "n_with_band": int(has.sum()),
    "bin_edges": edges.tolist(),
    "per_draw_fraction": draw_hist.round(4).tolist(),
    "rungs_per_bin": rung_hist.tolist(),
    "per_draw_mean_tm": float((tm[band] * w[band]).sum() / w[band].sum()),
    "chain_min_tm_q": dict(zip(map(str, q), np.quantile(cmin, q).round(3).tolist())),
    "chain_max_tm_q": dict(zip(map(str, q), np.quantile(cmax, q).round(3).tolist())),
    "chain_span_q": dict(zip(map(str, q), np.quantile(span, q).round(3).tolist())),
    "frac_chains_reaching_below_0.5": float((cmin < 0.5).mean()),
    "frac_chains_reaching_below_0.4": float((cmin < 0.4).mean()),
}
print(json.dumps(out, indent=1))
with open(a.out_json, "w") as fh:
    json.dump(out, fh, indent=1)
