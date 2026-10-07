#!/bin/bash
# Pre-fetch the ESMFold2 weights into the HF cache (compute nodes have no internet). Runs INSIDE a download-partition job
# after the env build; uses snapshot_download (no model load, so no big RAM) for biohub/ESMFold2 and the ESMC repo its config names.
source activate esmfold2
python - <<'PY'
import json, os
from huggingface_hub import snapshot_download
p = snapshot_download("biohub/ESMFold2")
print("ESMFold2 snapshot:", p, os.listdir(p))
cfg = None
for f in os.listdir(p):
    if f.endswith("config.json"):
        cfg = json.load(open(os.path.join(p, f))); print("config keys:", list(cfg)[:20]); break
txt = json.dumps(cfg) if cfg else ""
import re
repos = sorted(set(re.findall(r"biohub/[A-Za-z0-9_.-]+|EvolutionaryScale/[A-Za-z0-9_.-]+", txt)))
print("repos named in the config:", repos)
for r in repos:
    if r != "biohub/ESMFold2":
        print("fetching", r, snapshot_download(r))
PY
