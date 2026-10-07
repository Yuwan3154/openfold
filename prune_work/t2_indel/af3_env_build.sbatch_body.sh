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
