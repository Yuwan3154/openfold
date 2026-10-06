#!/bin/bash
#SBATCH -J t2indel_af3env
#SBATCH -p download
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/gridsan/cou/logs/t2indel_af3env-%j.out
#SBATCH --error=/home/gridsan/cou/logs/t2indel_af3env-%j.err
# RECORDED RECIPE (user 10-06: "Yes set it up on SuperCloud"). Official install route (google-deepmind/alphafold3 README):
#   python 3.11; pip install -r dev-requirements.txt; pip install --no-deps .; build_data
# download partition = internet. gcc 13.3 + cmake are system packages on SuperCloud (the C++ extension builds here).
# Weights: ~/params/af3.bin.zst (already on SuperCloud). Inference on V100 (compute 7.0) needs
# --flash_attention_implementation=xla and is TESTED by the shakedown, not assumed.
source /etc/profile
module load conda/Python-ML-2025b-pytorch
rc=0
conda create -y -n af3 python=3.11 pip || rc=$?
source activate af3 || rc=$?
python --version
cd /home/gridsan/cou
[ -d alphafold3_sc ] || git clone --depth 1 https://github.com/google-deepmind/alphafold3.git alphafold3_sc || rc=$?
cd alphafold3_sc
git log --oneline | head -1
python -m pip install --no-cache-dir -r dev-requirements.txt || rc=$?
python -m pip install --no-cache-dir --no-deps . || rc=$?
build_data || rc=$?
python -m pip list 2>/dev/null | grep -i -E "^(jax|jaxlib|jax-triton|triton|dm-haiku|alphafold3|numpy|rdkit|tensorflow) " || true
python - <<'PY' || rc=$?
import jax, alphafold3
print("jax", jax.__version__, "alphafold3", getattr(alphafold3, "__file__", "?"))
from alphafold3.model import model
print("alphafold3.model import ok")
PY
echo "rc_install=$rc"
exit $rc
