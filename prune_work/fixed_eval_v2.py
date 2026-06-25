"""Fixed-set structure-stability sweep, FAITHFUL path (user choice: full-MSA via train_openfold).
For each conf-weight M:
  before = validate_only(conf_smoke_init) on a FIXED val set   [val/lddt_ca, val/fape, val/plddt_loss, val/tm]
  train N steps at conf-weight M via run_conf_smoke.sh (full MSA, efficient attn) -> save raw last.ckpt
  after  = validate_only(last.ckpt_M) on the SAME fixed val set  (--validate_only refreshes EMA FROM loaded weights -> evals raw weights)
struct: val/lddt_ca higher=better (delta<0 = DEGRADED), val/fape lower=better. conf: val/plddt_loss+tm lower=better (delta<0 = IMPROVED).
Phases run 4-way (one M per GPU). before is computed once and shared.
"""
import argparse
import os
import re
import subprocess

PY = "/home/jupyter-chenxi/miniconda3/envs/cue_openfold_gated/bin/python"
REPO = "/home/jupyter-chenxi/openfold"
INIT = "/home/jupyter-chenxi/runs/conf_smoke_init.ckpt"
L = "/home/jupyter-chenxi/prune_work/lists_pdb"
VAL_TAGS = ["val/lddt_ca", "val/fape", "val/distogram", "val/plddt_loss", "val/tm", "val/experimentally_resolved"]

ap = argparse.ArgumentParser()
ap.add_argument("--mults", default="0.003,0.01,0.03,0.1,0.3,1,3")
ap.add_argument("--steps", type=int, default=100)
ap.add_argument("--lr", default="5e-4")
ap.add_argument("--warmup", default="10")
ap.add_argument("--gpus", default="0,1,2,3")
ap.add_argument("--val_list", default=f"{L}/fe_eval16.list")  # pre-cutoff train proteins (post-cutoff val lacks alignments -> stalls)
ap.add_argument("--root", default="/tmp/fe2")
args = ap.parse_args()

mults = [m.strip() for m in args.mults.split(",") if m.strip()]
gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
os.makedirs(args.root, exist_ok=True)


def read_val(logf):
    # validate_only prints the val summary to the log (no TB events file); parse "val/<name>  <value>".
    if not os.path.exists(logf):
        return {}
    txt = open(logf).read()
    out = {}
    for name, val in re.findall(r"val/(\w+)\s+([-\d.]+)", txt):
        out["val/" + name] = float(val)
    return out


def run_validate(ckpt, outdir, gpu, logf):
    env = {**os.environ, "CKPT": ckpt, "OUT_DIR": outdir, "VAL_LIST": args.val_list,
           "CUDA_VISIBLE_DEVICES": gpu}
    return subprocess.Popen(["bash", f"{REPO}/prune_work/run_fixed_validate.sh"], env=env,
                            stdout=open(logf, "w"), stderr=subprocess.STDOUT)


def run_train(m, outdir, gpu, logf):
    env = {**os.environ, "CONF_MULT": m, "STEPS": str(args.steps), "LR": args.lr, "WARMUP": args.warmup,
           "CKPT_EVERY": str(args.steps), "VAL_LIST": f"{L}/fe_test4.list",  # training end-val (ignored) kept tiny
           "OUT_DIR": outdir, "INIT_CKPT": INIT, "CUDA_VISIBLE_DEVICES": gpu}
    return subprocess.Popen(["bash", f"{REPO}/prune_work/run_conf_smoke.sh"], env=env,
                            stdout=open(logf, "w"), stderr=subprocess.STDOUT)


def wave(items, fn):
    for w in range(0, len(items), len(gpus)):
        procs = [fn(it, gpus[i]) for i, it in enumerate(items[w:w + len(gpus)])]
        for p in procs:
            p.wait()


# Phase 0: before = validate(init)
print("=== Phase 0: validate(init) ===", flush=True)
run_validate(INIT, f"{args.root}/before", gpus[0], f"{args.root}/before.log").wait()
before = read_val(f"{args.root}/before.log")
print("before:", {k: round(v, 4) for k, v in before.items()}, flush=True)

# Phase 1: train each M -> raw ckpt
print("=== Phase 1: train each M (4-way) ===", flush=True)
wave(mults, lambda m, g: run_train(m, f"{args.root}/train_M{m}", g, f"{args.root}/train_M{m}.log"))
ckpts = {m: f"{args.root}/train_M{m}/lightning_logs/version_0/checkpoints/last.ckpt" for m in mults}

# Phase 2: validate each M's raw ckpt
print("=== Phase 2: validate each M ckpt (4-way) ===", flush=True)
have = [m for m in mults if os.path.exists(ckpts[m])]
for m in mults:
    if m not in have:
        print(f"WARN: no ckpt for M={m} (train failed?)", flush=True)
wave(have, lambda m, g: run_validate(ckpts[m], f"{args.root}/after_M{m}", g, f"{args.root}/after_M{m}.log"))

# Phase 3: table
print("\n=== FIXED-EVAL v2 (full-MSA, validate_only) — struct: lddt_ca higher=better / fape lower=better; conf: plddt+tm lower=better ===")
hdr = f"{'M':>7} {'lddt_before':>11} {'lddt_after':>10} {'d_lddt':>8} {'fape_b':>7} {'fape_a':>7} {'plddt_b':>8} {'plddt_a':>8} {'tm_b':>6} {'tm_a':>6}"
print(hdr)
bl = before.get("val/lddt_ca", float("nan"))
for m in mults:
    a = read_val(f"{args.root}/after_M{m}.log")
    if not a:
        print(f"{m:>7}  (no after metrics)"); continue
    la = a.get("val/lddt_ca", float("nan"))
    print(f"{m:>7} {bl:>11.4f} {la:>10.4f} {la-bl:>+8.4f} {before.get('val/fape',float('nan')):>7.3f} "
          f"{a.get('val/fape',float('nan')):>7.3f} {before.get('val/plddt_loss',float('nan')):>8.3f} "
          f"{a.get('val/plddt_loss',float('nan')):>8.3f} {before.get('val/tm',float('nan')):>6.3f} {a.get('val/tm',float('nan')):>6.3f}")
print("\nd_lddt < 0 = structure DEGRADED at that M. Largest M with d_lddt ~>= 0 = structure-stable ceiling.")
