"""Driver for the fixed-set structure-stability sweep (user test 2): run fixed_eval_smoke.py for each
conf-weight M, 4-way (one M per GPU per wave), then collect the RESULT lines into a table.
Controlled: same train proteins (seed 42) + same fixed eval set for every M -> struct_delta vs M is the
clean conf-weight effect on structure (no data-order artifact). struct_delta>0 = structure DEGRADED.
"""
import argparse
import os
import re
import subprocess

PY = "/home/jupyter-chenxi/miniconda3/envs/cue_openfold_gated/bin/python"
REPO = "/home/jupyter-chenxi/openfold"

ap = argparse.ArgumentParser()
ap.add_argument("--mults", default="0.003,0.01,0.03,0.1,0.3,1,3")
ap.add_argument("--steps", type=int, default=100)
ap.add_argument("--lr", type=float, default=5e-4)
ap.add_argument("--gpus", default="0,1,2,3")
ap.add_argument("--root", default="/tmp/fixed_eval")
args = ap.parse_args()

mults = [m.strip() for m in args.mults.split(",") if m.strip()]
gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
os.makedirs(args.root, exist_ok=True)
logs = {m: f"{args.root}/M{m}.log" for m in mults}

for w in range(0, len(mults), len(gpus)):
    wave = mults[w:w + len(gpus)]
    procs = []
    for m, gpu in zip(wave, gpus):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "PYTHONPATH": f"{REPO}/openfold",
               "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
        lf = open(logs[m], "w")
        print(f"launch M={m} on GPU {gpu}", flush=True)
        procs.append(subprocess.Popen(
            [PY, f"{REPO}/prune_work/fixed_eval_smoke.py", "--mult", m,
             "--steps", str(args.steps), "--lr", str(args.lr)], env=env, stdout=lf, stderr=subprocess.STDOUT))
    for p in procs:
        p.wait()

print("\n=== FIXED-SET STRUCTURE-STABILITY SWEEP (lower=better; struct_delta>0 = structure DEGRADED, conf_delta<0 = conf IMPROVED) ===")
print(f"{'M':>7} {'struct_before':>13} {'struct_after':>12} {'struct_delta':>12} {'conf_before':>11} {'conf_after':>10} {'conf_delta':>10}")
rows = []
for m in mults:
    txt = open(logs[m]).read() if os.path.exists(logs[m]) else ""
    mt = re.search(r"RESULT M=\S+ struct_before=(\S+) struct_after=(\S+) struct_delta=(\S+) "
                   r"conf_before=(\S+) conf_after=(\S+) conf_delta=(\S+)", txt)
    if mt:
        sb, sa, sd, cb, ca, cd = mt.groups()
        print(f"{m:>7} {sb:>13} {sa:>12} {sd:>12} {cb:>11} {ca:>10} {cd:>10}")
        rows.append((m, float(sd), float(cd)))
    else:
        err = "OOM" if "out of memory" in txt.lower() else ("ERR/traceback" if "Traceback" in txt else "no RESULT")
        print(f"{m:>7}  ({err})")

stable = [m for m, sd, cd in rows if sd <= 0.02]
print("\nStructure-stable M (struct_delta <= +0.02):", stable or "NONE")
improved = [m for m, sd, cd in rows if cd < 0]
print("Conf-improved M (conf_delta < 0):", improved or "NONE")
