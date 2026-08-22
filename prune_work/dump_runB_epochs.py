import glob
import sys
from collections import defaultdict

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

WANT = [
    "val/lddt_ca", "val/lddt_ca_held_out", "val/recall_2A", "val/gdt_ts",
    "val/alignment_rmsd", "val/ptm_calibration_spearman", "val/loss",
    "explore/using_true_loss_epoch", "explore/conf_picks_loss_argmin_epoch",
    "explore/regret_vs_best_epoch", "explore/loss_spread_epoch",
    "explore/loss_gain_vs_mean_epoch",
    "t4/promote_rate_epoch", "t4/tm_pred_epoch", "t4/tm_template_epoch",
    "epoch",
]

root = "/home/jupyter-chenxi/runs/runB_full_stack_pda_eval/lightning_logs"
rows = defaultdict(dict)   # (version, step) -> {tag: val}
step_epoch = {}            # (version, step) -> epoch

for ver in sorted(glob.glob(f"{root}/version_*")):
    vname = ver.rsplit("_", 1)[-1]
    for ev in sorted(glob.glob(f"{ver}/events.out.tfevents.*")):
        acc = EventAccumulator(ev, size_guidance={"scalars": 0})
        acc.Reload()
        tags = set(acc.Tags()["scalars"])
        for tag in WANT:
            if tag not in tags:
                continue
            for e in acc.Scalars(tag):
                if tag == "epoch":
                    step_epoch[(vname, e.step)] = e.value
                else:
                    rows[(vname, e.step)][tag] = e.value

print(f"{'ver':>3} {'step':>7} {'ep':>4}  " + "  ".join(f"{t.split('/')[-1][:14]:>14}" for t in WANT[:-1]))
for (vname, step) in sorted(rows, key=lambda k: (k[0], k[1])):
    d = rows[(vname, step)]
    if "val/lddt_ca" not in d and "explore/using_true_loss_epoch" not in d:
        continue
    ep = step_epoch.get((vname, step))
    eps = f"{ep:.0f}" if ep is not None else "-"
    cells = []
    for t in WANT[:-1]:
        v = d.get(t)
        cells.append(f"{v:14.4f}" if v is not None else f"{'-':>14}")
    print(f"{vname:>3} {step:>7} {eps:>4}  " + "  ".join(cells))
