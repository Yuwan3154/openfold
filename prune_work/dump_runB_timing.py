import glob
import datetime

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

root = "/home/jupyter-chenxi/runs/runB_full_stack_pda_eval/lightning_logs"
pts = []
for ver in sorted(glob.glob(f"{root}/version_*")):
    vname = ver.rsplit("_", 1)[-1]
    for ev in sorted(glob.glob(f"{ver}/events.out.tfevents.*")):
        acc = EventAccumulator(ev, size_guidance={"scalars": 0})
        acc.Reload()
        if "val/lddt_ca" not in acc.Tags()["scalars"]:
            continue
        for e in acc.Scalars("val/lddt_ca"):
            pts.append((vname, e.step, e.wall_time))

pts.sort(key=lambda p: p[2])
prev = None
print(f"{'ver':>3} {'step':>7} {'val ended (EDT)':>20} {'gap_h':>7}")
for vname, step, wt in pts:
    ts = datetime.datetime.fromtimestamp(wt).strftime("%Y-%m-%d %H:%M:%S")
    gap = "" if prev is None else f"{(wt - prev) / 3600:7.2f}"
    print(f"{vname:>3} {step:>7} {ts:>20} {gap:>7}")
    prev = wt
