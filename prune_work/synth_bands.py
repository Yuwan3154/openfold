"""Band the synthetic-template TM sample on the SAME edges as the natural one, for comparison."""

import numpy as np

z = np.load("/home/gridsan/cou/pp1c_work/template_index_sample/index_all.npz", allow_pickle=False)
tm = z["tm"].reshape(-1)
L = z["length"]
print(f"synthetic: {len(tm)} templates over {len(L)} chains, lengths {L.min()}-{L.max()}")
q = [5, 25, 50, 75, 95]
v = np.percentile(tm, q)
print(f"  min {tm.min():.3f}  " + "  ".join(f"p{a:02d} {b:.3f}" for a, b in zip(q, v)) +
      f"  max {tm.max():.3f}  mean {tm.mean():.3f}")
print("\n  banded:")
edges = [0, .2, .3, .4, .5, .6, .7, .8, .9, 1.01]
for lo, hi in zip(edges, edges[1:]):
    n = int(((tm >= lo) & (tm < hi)).sum())
    print(f"    {lo:.1f}-{min(hi,1.0):.1f}: {n:7d}  {100*n/len(tm):5.1f}%")
print(f"\n  TM < 0.9: {100*(tm < 0.9).mean():.1f}%    TM < 0.5: {100*(tm < 0.5).mean():.1f}%")
per = (z["tm"] < 0.9).sum(axis=1)
print(f"  per chain with TM < 0.9: median {int(np.median(per))}  mean {per.mean():.1f}")
