import re
import subprocess
import sys

import os
LAUNCHER = os.environ.get("LAUNCHER", "/home/jupyter-chenxi/prune_work/run_C_replica_exchange.sh")
TRAIN = "/home/jupyter-chenxi/openfold-esmfold2-recycling/train_openfold.py"

src = open(LAUNCHER).read()
flags = re.findall(r"^\s+(--[A-Za-z0-9_]+)\s*\\?$", src, re.M)
known = set(re.findall(r'add_argument\(\s*\n?\s*"(--[A-Za-z0-9_]+)"', open(TRAIN).read()))

print(f"flags in launcher : {len(flags)}")
print(f"argparse flags    : {len(known)}")

unknown = [f for f in flags if f not in known]
dupes = sorted({f for f in flags if flags.count(f) > 1})
print(f"UNKNOWN flags     : {unknown if unknown else 'none'}")
print(f"DUPLICATED flags  : {dupes if dupes else 'none'}")

# guards that must hold for this specific run
def val(flag):
    m = re.search(re.escape(flag) + r"\s*\\\n\s+(\S+?)\s*\\?\n", src)
    return m.group(1) if m else None

checks = {
    "monitor UNCHANGED at val/lddt_ca": val("--checkpoint_monitor") == "val/lddt_ca",
    "save_top_k is 5": val("--checkpoint_save_top_k") == "5",
    "resume_model_weights_only is true": val("--resume_model_weights_only") == "true",
    "resume_from_ema is true": val("--resume_from_ema") == "true",
    "explore_k == len(noise_ladder)": (
        val("--explore_k") is not None
        and len(val("--explore_noise_ladder").split(",")) == int(val("--explore_k"))),
    "explore_switch_epoch is 0": val("--explore_switch_epoch") == "0",
    "t4_promote_after_epoch is 0": val("--t4_promote_after_epoch") == "0",
    "t4_promote_all present": "--t4_promote_all" in flags,
    "fresh t4 pool dir": "/t4_pool" in src,
    "run dir is a runC variant": "/home/jupyter-chenxi/runs/runC" in src,
    "resume ckpt is runB best-010": "best-010-008250.ckpt" in src,
    "contractive + gaussian on": "--contractive_recycling" in flags and "--gaussian_pair_init" in flags,
    "expanded val easy+hard wired": ("--expanded_val_easy" in flags and "--expanded_val_hard" in flags),
    "nonneural ids wired": "--pda_nonneural_ids" in flags,
}
print()
npass = 0
for k, v in checks.items():
    print(f"  [{'OK ' if v else 'FAIL'}] {k}")
    npass += bool(v)
print(f"\nguards: {npass}/{len(checks)}")

ok = not unknown and not dupes and npass == len(checks)
# bash syntax check
sy = subprocess.run(["bash", "-n", LAUNCHER], capture_output=True, text=True)
print(f"bash -n: {'OK' if sy.returncode == 0 else sy.stderr}")
ok = ok and sy.returncode == 0
print(f"\nVERDICT: {'READY' if ok else 'NOT READY'}")
sys.exit(0 if ok else 1)
