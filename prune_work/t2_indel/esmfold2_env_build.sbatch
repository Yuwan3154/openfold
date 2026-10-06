#!/bin/bash
#SBATCH -J t2indel_esm2env
#SBATCH -p download
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/gridsan/cou/logs/t2indel_esm2env-%j.out
#SBATCH --error=/home/gridsan/cou/logs/t2indel_esm2env-%j.err
# RECORDED RECIPE (user 10-06: "ESMFold2 should also be able to run on SuperCloud. Just set up the environment").
# download partition = the only compute place with internet. Official route (github.com/Biohub/esm README): `pip install esm`
# (PyPI 3.4.1.post1: python >= 3.12, torch >=2.11,<2.12, transformers >=4.57.6,<5). torch from the cu126 index
# because SuperCloud is V100 (sm_70); NO flash-attn (Volta is unsupported by flash-attn 2) - whether inference is correct
# without it is TESTED by the refold pilot, not assumed. Weights biohub/ESMFold2 (+ its ESMC backbone) are pre-fetched
# into the HF cache because compute nodes have no internet.
source /etc/profile
module load conda/Python-ML-2025b-pytorch
rc=0
conda create -y -n esmfold2 python=3.12 pip || rc=$?
source activate esmfold2 || rc=$?
python --version
python -m pip install --no-cache-dir "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu126 || rc=$?
python -m pip install --no-cache-dir esm==3.4.1.post1 || rc=$?
python -m pip list 2>/dev/null | grep -i -E "^(torch|esm|transformers|numpy|biotite|huggingface-hub|safetensors|triton|cuequivariance[a-z0-9-]*|flash[-_]attn) " || true
python - <<'PY' || rc=$?
import importlib, pkgutil, esm
print("esm", esm.__version__ if hasattr(esm, "__version__") else "?", esm.__file__)
import esm.models
print("esm.models:", [m.name for m in pkgutil.iter_modules(esm.models.__path__)])
try:
    import esm.models.esmfold2 as e2
    print("esmfold2 module:", [m.name for m in pkgutil.iter_modules(e2.__path__)])
    print("has InputBuilder:", hasattr(e2, "ESMFold2InputBuilder"))
except Exception as ex:
    print("esm.models.esmfold2 import failed:", repr(ex))
try:
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
    print("transformers ESMFold2Model: ok")
except Exception as ex:
    print("transformers ESMFold2Model import failed:", repr(ex))
PY
echo "rc_install=$rc"
exit $rc
