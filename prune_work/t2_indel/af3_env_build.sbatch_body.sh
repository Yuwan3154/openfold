# RECORDED RECIPE v3 (repo HEAD: python >= 3.12, jax 0.10.2, uv.lock; docs/installation.md: uv venv --python 3.12; uv sync; uv run build_data).
# The download node has no libz.so (CMake: "Could NOT find ZLIB (missing: ZLIB_LIBRARY)"): a local zlib 1.3.1 is built and CMake pointed at it.
source /etc/profile
module load conda/Python-ML-2025b-pytorch
rc=0
python -m pip install --user --no-cache-dir uv || rc=$?
export PATH=$HOME/.local/bin:$PATH
if [ ! -e $HOME/zlib_local/lib/libz.so ]; then
  (cd /tmp && curl -fsSL https://github.com/madler/zlib/releases/download/v1.3.1/zlib-1.3.1.tar.gz | tar xz && cd zlib-1.3.1 && ./configure --prefix=$HOME/zlib_local >/dev/null && make -j8 >/dev/null && make install >/dev/null) || rc=$?
fi
ls $HOME/zlib_local/lib/libz.so* $HOME/zlib_local/include/zlib.h
export CMAKE_PREFIX_PATH=$HOME/zlib_local ZLIB_ROOT=$HOME/zlib_local
export CMAKE_ARGS="-DZLIB_LIBRARY=$HOME/zlib_local/lib/libz.so -DZLIB_INCLUDE_DIR=$HOME/zlib_local/include"
export LD_LIBRARY_PATH=$HOME/zlib_local/lib:${LD_LIBRARY_PATH:-}
export UV_CACHE_DIR=/home/gridsan/cou/.cache/uv
cd /home/gridsan/cou/alphafold3_sc || exit 1
git log --oneline | head -1
rm -rf .venv; uv venv --python 3.12 .venv || rc=$?
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
