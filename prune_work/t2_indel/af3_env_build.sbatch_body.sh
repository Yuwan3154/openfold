# RECORDED RECIPE v2 (the repo HEAD changed: python >= 3.12, jax 0.10.2, uv.lock; docs/installation.md: uv venv --python 3.12; uv sync; uv run build_data).
source /etc/profile
module load conda/Python-ML-2025b-pytorch
rc=0
conda env remove -y -n af3 >/dev/null 2>&1
python -m pip install --user --no-cache-dir uv || rc=$?
export PATH=$HOME/.local/bin:$PATH
cd /home/gridsan/cou/alphafold3_sc || exit 1
git log --oneline | head -1
export UV_CACHE_DIR=/home/gridsan/cou/.cache/uv
# the download node has no zlib dev package (CMake: "Could NOT find ZLIB (missing: ZLIB_LIBRARY)"): use the module conda's own
P=$(python -c 'import sys; print(sys.prefix)')
ls $P/lib/libz.so* $P/include/zlib.h
export CMAKE_PREFIX_PATH=$P ZLIB_ROOT=$P
export CMAKE_ARGS="-DZLIB_LIBRARY=$P/lib/libz.so -DZLIB_INCLUDE_DIR=$P/include"
uv venv --python 3.12 .venv || rc=$?
uv sync || rc=$?
uv run build_data || rc=$?
.venv/bin/python - <<'PY' || rc=$?
import jax, alphafold3
print("jax", jax.__version__, "devices:", jax.devices())
from alphafold3.model import model
print("alphafold3.model import ok")
PY
echo "rc_install=$rc"
exit $rc
