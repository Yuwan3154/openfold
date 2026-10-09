#!/bin/bash
#SBATCH -J esmc_dl
#SBATCH -p download
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/gridsan/cou/logs/esmc_dl-%j.out
#SBATCH --error=/home/gridsan/cou/logs/esmc_dl-%j.err
# usage: sbatch [--export=ALL,SIZE=600M] esmc_download.sbatch
# Fetch biohub/ESMC-<SIZE> (default 300M) (the repo esm's ESMC.from_pretrained resolves; the cached EvolutionaryScale/esmc-300m-2024-12 is the legacy .pth layout it cannot load).
# Same allow_patterns as esm/models/hub.py resolve_model_dir. The download partition has the only internet.
source /etc/profile
module load conda/Python-ML-2025b-pytorch
source activate esmfold2
python - <<'PY'
import os
from huggingface_hub import snapshot_download
import esm.models.esmc.config as esmc_config
from esm.models.hub import resolve_model_dir
REPO = getattr(esmc_config, f"ESMC_{os.environ.get('SIZE', '300M')}_HF_REPO")  # SIZE = 300M (default) | 600M
print("repo:", REPO)
p = snapshot_download(REPO, allow_patterns=["*.json", "*.safetensors"])
fs = sorted(os.listdir(p))
print("snapshot:", p, fs)
assert "config.json" in fs and any(f.endswith(".safetensors") for f in fs), fs
assert all(os.path.getsize(os.path.join(p, f)) > 0 for f in fs if f.endswith(".safetensors")), fs
q = resolve_model_dir(REPO, local_files_only=True)
assert os.path.samefile(p, q), (p, q)
PY
rc=$?
echo rc=$rc
exit $rc
