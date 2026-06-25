"""Conf-weight tuning sweep: for each multiplier M, run a ~100-step conf+structure smoke
(run_conf_smoke.sh, evoformer-only, conf weights = AF2 ratios x M), then read per-step TRAIN losses
from TensorBoard and report, per M:
  - conf-loss drop      : sum(plddt_loss, tm, experimentally_resolved), mean(first 15 steps) -> mean(last 15)
  - still-dropping?     : mean(last 15) < mean(steps 70..85)  (no plateau)
  - structure change    : sum(fape, distogram), mean(last 15) - mean(first 15)  (must be <= ~0; no harm)
Recommend the largest M with the biggest conf drop that is still-dropping and does NOT raise structure loss.

GPU-blocked by the 12-block run -> run on the box when a GPU frees (single GPU per M, sequential):
  python conf_sweep.py --mults 1,3,10,30
Builds conf_smoke_init.ckpt first if missing. Per-step train losses are noisy (batch 1) -> heavy
smoothing; if signal is ambiguous, raise --steps or add a fixed-set before/after eval.
"""
import argparse
import glob
import os
import subprocess

REPO = "/home/jupyter-chenxi/openfold"
PY = "/home/jupyter-chenxi/miniconda3/envs/cue_openfold_gated/bin/python"

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--mults", default="1,3,10,30", help="comma list of conf-weight multipliers to sweep")
ap.add_argument("--steps", type=int, default=100)
ap.add_argument("--lr", default="5e-4")
ap.add_argument("--gpus", default="0,1,2,3", help="comma list of GPUs; one M per GPU, in waves")
ap.add_argument("--root", default="/tmp/conf_sweep")
args = ap.parse_args()


def build_init():
    init = "/home/jupyter-chenxi/runs/conf_smoke_init.ckpt"
    if not os.path.exists(init):
        subprocess.run([PY, f"{REPO}/prune_work/build_conf_smoke_init.py"], check=True,
                       env={**os.environ, "PYTHONPATH": f"{REPO}/openfold"})
    return init


def read_train_scalars(out_dir):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    vdirs = sorted(glob.glob(f"{out_dir}/lightning_logs/version_*"))
    if not vdirs:
        return {}
    ea = EventAccumulator(vdirs[-1]); ea.Reload()
    tags = [t for t in ea.Tags().get("scalars", []) if t.startswith("train/") and not t.endswith("_epoch")]
    return {t: [s.value for s in ea.Scalars(t)] for t in tags}


def classify(tags):
    conf = [t for t in tags if any(s in t.lower() for s in ("plddt", "/tm", "experimentally_resolved"))]
    struct = [t for t in tags if any(s in t.lower() for s in ("fape", "distogram"))]
    return conf, struct


def smean(xs, a, b):
    seg = xs[a:b]
    return sum(seg) / len(seg) if seg else float("nan")


def summarize(series):
    tags = list(series.keys())
    conf_t, struct_t = classify(tags)
    n = max((len(v) for v in series.values()), default=0)
    if n < 30:
        return None, conf_t, struct_t
    def total(ts):
        return [sum(series[t][i] for t in ts if i < len(series[t])) for i in range(n)]
    conf = total(conf_t); struct = total(struct_t)
    conf_first, conf_last = smean(conf, 0, 15), smean(conf, n - 15, n)
    conf_mid = smean(conf, max(0, n - 30), n - 15)
    struct_first, struct_last = smean(struct, 0, 15), smean(struct, n - 15, n)
    return {
        "conf_drop": conf_first - conf_last,                 # >0 = improved
        "conf_still_dropping": conf_last < conf_mid,          # no plateau
        "struct_change": struct_last - struct_first,          # <=0 = no harm
        "conf_first": conf_first, "conf_last": conf_last,
        "struct_first": struct_first, "struct_last": struct_last,
    }, conf_t, struct_t


def main():
    init = build_init()
    mults = [m.strip() for m in args.mults.split(",") if m.strip()]
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    os.makedirs(args.root, exist_ok=True)
    outs = {m: f"{args.root}/M{m}" for m in mults}
    # run in parallel waves: one M per GPU
    for w in range(0, len(mults), len(gpus)):
        wave = mults[w:w + len(gpus)]
        procs = []
        for m, gpu in zip(wave, gpus):
            env = {**os.environ, "CONF_MULT": m, "STEPS": str(args.steps), "LR": args.lr,
                   "OUT_DIR": outs[m], "INIT_CKPT": init, "CUDA_VISIBLE_DEVICES": gpu}
            logf = open(f"{args.root}/M{m}.log", "w")
            print(f"launch M={m} on GPU {gpu}", flush=True)
            procs.append(subprocess.Popen(["bash", f"{REPO}/prune_work/run_conf_smoke.sh"],
                                          env=env, stdout=logf, stderr=subprocess.STDOUT))
        for p in procs:
            p.wait()
    rows = []
    for i, m in enumerate(mults):
        s, conf_t, struct_t = summarize(read_train_scalars(outs[m]))
        rows.append((m, s))
        if i == 0 and s is not None:
            print("conf tags:", conf_t, "| struct tags:", struct_t, flush=True)

    print("\n=== CONF-WEIGHT SWEEP ===")
    print(f"{'M':>5} {'conf_drop':>10} {'still_drop':>11} {'struct_chg':>11} {'conf(f->l)':>16} {'struct(f->l)':>16}")
    best = None
    for m, s in rows:
        if s is None:
            print(f"{m:>5}  (insufficient steps logged)"); continue
        print(f"{m:>5} {s['conf_drop']:>10.4f} {str(s['conf_still_dropping']):>11} {s['struct_change']:>11.4f} "
              f"{s['conf_first']:>7.3f}->{s['conf_last']:<7.3f} {s['struct_first']:>7.3f}->{s['struct_last']:<7.3f}")
        ok = s["conf_still_dropping"] and s["struct_change"] <= 0.02  # tolerate tiny structure noise
        if ok and (best is None or s["conf_drop"] > best[1]["conf_drop"]):
            best = (m, s)
    if best:
        print(f"\nRECOMMEND CONF_MULT={best[0]} (max conf drop with no plateau and no structure harm)")
    else:
        print("\nNo M satisfied (no-plateau AND no-structure-harm); inspect the table / widen the sweep.")


if __name__ == "__main__":
    main()
