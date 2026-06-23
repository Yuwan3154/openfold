"""Pick the best WS3 multi-start replica by TM(slim_pred, target); copy its artifacts to <root>/best/.
  python ws3_pick_best.py <root>
"""
import json
import shutil
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for s in sorted(root.glob("rep*/summary.json")):
    rows.append((s.parent.name, json.load(open(s))))
if not rows:
    raise RuntimeError(f"no rep*/summary.json under {root} (all replicas failed? check rep*.log)")


def f(x):
    return "n/a " if x is None else f"{x:.3f}"


print(f"{'rep':6}{'init':10}{'seed':6}{'loss_last':11}{'TM_slim':9}{'TM_full':9}{'TM_xmodel':9}")
for name, m in rows:
    print(f"{name:6}{str(m['init_seq']):10}{str(m['seed']):6}{m['loss_last']:<11.4f}"
          f"{f(m['tm_slim_target']):9}{f(m['tm_full_target']):9}{f(m['tm_full_slim']):9}")

best_name, best_m = max(rows, key=lambda r: (r[1]["tm_slim_target"] if r[1]["tm_slim_target"] is not None else -1.0))
best_dir = root / best_name
out = root / "best"
out.mkdir(exist_ok=True)
for fn in ("target.pdb", "slim_pred.pdb", "full_pred.pdb", "summary.json", "loss.png"):
    if (best_dir / fn).exists():
        shutil.copy(best_dir / fn, out / fn)
print(f"BEST = {best_name}  TM(slim,target)={best_m['tm_slim_target']}  -> copied to {out}")
