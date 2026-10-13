"""Unit test of the greedy min-max selection (numpy only)."""
import numpy as np

from t2_select_diverse import select_greedy, stats

rng = np.random.default_rng(0)
n = 80
M = rng.uniform(0.2, 0.5, (n, n)); M = (M + M.T) / 2; np.fill_diagonal(M, 0)
dup = np.arange(14)                      # 14 near-duplicates of each other (pair TM 0.95): 80 - 64 = 16 drops are enough to keep only one of them
M[np.ix_(dup, dup)] = 0.95; np.fill_diagonal(M, 0)
idx = select_greedy(M, 64)
assert len(idx) == 64 and len(set(idx)) == 64
mx, _, _ = stats(M, idx)
rand = [stats(M, sorted(rng.choice(n, 64, replace=False)))[0] for _ in range(20)]
assert mx < 0.95 <= min(rand), (mx, min(rand))      # the near-duplicate block is thinned to one member (random subsets keep >= 2 of them)
assert sum(i in set(dup) for i in idx) == 1, [i for i in idx if i in set(dup)]
assert select_greedy(M[:30, :30], 64) == list(range(30))   # a pool <= k keeps everything
print("test_t2_select: all checks passed")
