#!/bin/bash
#SBATCH -J raygun_envbuild
#SBATCH -p download
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/gridsan/cou/logs/raygun_envbuild-%j.out
#SBATCH --error=/home/gridsan/cou/logs/raygun_envbuild-%j.err
# Builds the `raygun` env (download partition = the only compute place with internet) and CACHES the weights
# (ESM-2 650M + raygun_8_8mil_800M, both fetched by torch.hub into ~/.cache/torch/hub; the home dir is shared
# with GPU nodes). Pins follow the Raygun README's verified set: py3.11, torch 2.1.1+cu121 (ships sm_70),
# numpy 1.26.4, pandas 2.1.4, fair-esm 2.0.0. Recipe is recorded in the PLAN.
source /etc/profile
module load conda/Python-ML-2025b-pytorch
rc=0
conda create -y -n raygun python=3.11 pip || rc=$?
source activate raygun || rc=$?
python -m pip install --no-cache-dir torch==2.1.1 --index-url https://download.pytorch.org/whl/cu121 || rc=$?
python -m pip install --no-cache-dir numpy==1.26.4 pandas==2.1.4 fair-esm==2.0.0 || rc=$?
python -m pip install --no-cache-dir git+https://github.com/rohitsinghlab/raygun.git || rc=$?
python -m pip list 2>/dev/null | grep -i -E "^(torch|numpy|pandas|fair-esm|raygun|lightning|einops|h5py|hydra-core) " || true
# cache both weight files and prove the model constructs (CPU; no GPU on this partition)
CUDA_VISIBLE_DEVICES= python - <<'PY' || rc=$?
import torch
from raygun.pretrained import raygun_8_8mil_800M
m = raygun_8_8mil_800M()
print("raygun loaded:", type(m).__name__, "params(M)=%.1f" % (sum(p.numel() for p in m.parameters()) / 1e6))
PY
ls -la ~/.cache/torch/hub/checkpoints 2>&1 | head
echo "rc=$rc"
exit $rc
