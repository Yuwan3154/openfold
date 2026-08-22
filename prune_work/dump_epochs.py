"""Epoch-level TB scalars for a run, walking EVERY version_* dir.

The predecessor watcher globbed a hard-coded version_0 and silently went quiet after a restart while
its PID stayed alive. Takes the run dir as argv[1] so it cannot be pinned to one run either.
"""
import glob
import sys
from collections import defaultdict

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RUN = sys.argv[1] if len(sys.argv) > 1 else "/home/jupyter-chenxi/runs/runC_replica_exchange"
PREFIX = tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 else ("val/", "explore/", "t4/")

rows = defaultdict(dict)
for ver in sorted(glob.glob(f"{RUN}/lightning_logs/version_*")):
    v = ver.rsplit("_", 1)[-1]
    for ev in sorted(glob.glob(f"{ver}/events.out.tfevents.*")):
        acc = EventAccumulator(ev, size_guidance={"scalars": 0})
        acc.Reload()
        for tag in acc.Tags()["scalars"]:
            if not tag.startswith(PREFIX):
                continue
            for e in acc.Scalars(tag):
                rows[(v, e.step)][tag] = e.value

if not rows:
    print(f"no scalars yet under {RUN}")
    sys.exit(0)

# an epoch boundary is where val/ scalars appear
val_steps = sorted([k for k, d in rows.items() if any(t.startswith("val/") for t in d)],
                   key=lambda k: (k[0], k[1]))
print(f"run: {RUN}")
print(f"epoch-boundary (validation) events: {len(val_steps)}\n")
HEAD = ["val/lddt_ca", "val/lddt_ca_src_pda", "val/lddt_ca_src_easy", "val/lddt_ca_src_hard",
        "val/lddt_ca_nonneural", "val/recall_2A", "val/ptm_calibration_spearman", "val/loss"]
print(f"{'ver':>3} {'step':>7}  " + "  ".join(f"{t.split('/')[-1][:16]:>16}" for t in HEAD))
for k in val_steps:
    d = rows[k]
    print(f"{k[0]:>3} {k[1]:>7}  " + "  ".join(
        f"{d[t]:16.4f}" if t in d else f"{'-':>16}" for t in HEAD))

# latest step-level explore/t4 values, which is where promotion health shows up
print("\nlatest explore/ and t4/ step scalars:")
last = {}
for (v, s), d in sorted(rows.items(), key=lambda x: (x[0][0], x[0][1])):
    for t, val in d.items():
        if t.startswith(("explore/", "t4/")):
            last[t] = (s, val)
for t in sorted(last):
    s, val = last[t]
    print(f"  {t:44s} step {s:>7}  {val:12.4f}")
