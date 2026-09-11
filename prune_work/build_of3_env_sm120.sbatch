#!/bin/bash
#SBATCH -p mit_normal
#SBATCH -J of3_sm120_build
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH -o /orcd/pool/006/chenxiou/logs/of3_sm120_build-%j.out
#SBATCH -e /orcd/pool/006/chenxiou/logs/of3_sm120_build-%j.out
#
# Build of3_env_sm120: of3_env with the SAME package versions but the cu128 torch build, so the
# RTX Pro 6000 (Blackwell, sm_120) nodes of pi_so3 become usable by Run D.
#
# WHY THIS EXISTS: 22526403 died in 7 s on node4007 -- of3_env is torch 2.7.1+cu126, whose
# libtorch_cuda.so carries sm_50..sm_90 only. compute_90 PTX does NOT JIT to sm_120.
#
# NOTHING HERE IS AN INVENTED VERSION. Every pin is the version ALREADY installed in of3_env, taken
# from the cu128 index instead of cu126:
#     torch       2.7.1+cu126  -> 2.7.1+cu128
#     torchvision 0.22.1+cu126 -> 0.22.1+cu128
#     torchaudio  2.7.1+cu126  -> 2.7.1+cu128
# The two compute capabilities are the two that physically exist in pi_so3, read off the hardware:
#     9.0  = H200          (node4009)
#     12.0 = RTX Pro 6000  (node4007/4008, named by torch's own sm_120 error message)
#
# of3_env is NOT modified. The existing repo-root .so is backed up and relocated INTO of3_env's
# site-packages so the cu126/H200 path keeps working byte-identically; the repo root is then left
# without a .so so each env supplies its own arch-appropriate copy (sys.path[0] is the repo, which
# would otherwise shadow both).
set -eo pipefail

S=/orcd/scratch/orcd/011/chenxiou
SRC=$S/envs/of3_env
DST=$S/envs/of3_env_sm120
REPO=$S/openfold-esmfold2-recycling
SITE_SRC=$SRC/lib/python3.11/site-packages
SITE_DST=$DST/lib/python3.11/site-packages
SO=attn_core_inplace_cuda.cpython-311-x86_64-linux-gnu.so
ARCHES="9.0;12.0"
STAMP=$(date +%Y%m%dT%H%M%S)
REC=$S/envs/RECIPE_of3_env_sm120.txt

echo "START $(date -Is) host=$(hostname) job=$SLURM_JOB_ID"
echo "SRC=$SRC"
echo "DST=$DST"
echo "ARCHES=$ARCHES"

# ---------------------------------------------------------------- gate 0: toolchain
module load cuda/12.9.1
NVCC_VER=$(nvcc --version | grep -oE "release [0-9]+\.[0-9]+" | awk '{print $2}')
NVCC_MAJ=${NVCC_VER%%.*}; NVCC_MIN=${NVCC_VER##*.}
echo "nvcc=$NVCC_VER  gcc=$(gcc -dumpversion)  CUDA_HOME=${CUDA_HOME:-unset}"
# sm_120 codegen needs nvcc >= 12.8. nvcc 13.x would be a MAJOR mismatch against a cu128 torch.
[ "$NVCC_MAJ" -eq 12 ] || { echo "ABORT: need nvcc 12.x (cu128 torch); got $NVCC_VER"; exit 1; }
[ "$NVCC_MIN" -ge 8 ] || { echo "ABORT: need nvcc >= 12.8 for sm_120; got $NVCC_VER"; exit 1; }

# ---------------------------------------------------------------- gate 1: network
curl -sfI --max-time 30 https://download.pytorch.org/whl/cu128/torch/ >/dev/null \
  || { echo "ABORT: no route to download.pytorch.org from this node -- wheels must be staged first"; exit 1; }
echo "network to download.pytorch.org OK"

# ---------------------------------------------------------------- gate 2: destination is free
[ -e "$DST" ] && { echo "ABORT: $DST already exists -- refusing to overwrite an env"; exit 1; }
[ -x "$SRC/bin/python" ] || { echo "ABORT: $SRC/bin/python missing"; exit 1; }
[ -f "$REPO/$SO" ] || { echo "ABORT: $REPO/$SO missing -- nothing to preserve for the cu126 path"; exit 1; }

# ---------------------------------------------------------------- 3. inventory the source
# ⛔ `du -sb` is NOT a completeness measure. It is --apparent-size, which INCLUDES directory inode
#    sizes, and a freshly created directory is more compact than one that has grown over months.
#    Gating on it aborted 22590939 with a 303,009-byte (0.004%) delta while all 60,898 entries had
#    copied. The real measure is the summed size of REGULAR FILES, plus counts per entry type.
inv() {  # one walk -> "<n_regular> <n_dirs> <n_symlinks> <regular_file_bytes>"
  # ⛔ The \n is load-bearing: without it `read` hits EOF, returns 1, and `set -e` kills the job
  #    before a single line of output -- the same silent-abort shape as the pgrep no-match bug.
  find "$1" -printf '%y %s\n' \
    | awk '{t[$1]++; if($1=="f") b+=$2} END{printf "%d %d %d %d\n", t["f"]+0, t["d"]+0, t["l"]+0, b+0}'
}
"$SRC/bin/pip" freeze > "$S/envs/of3_env.freeze.$STAMP.txt"
read -r SRC_F SRC_D SRC_L SRC_FB < <(inv "$SRC")
SRC_DU=$(du -sb "$SRC" | awk '{print $1}')
echo "SRC inventory: regular=$SRC_F dirs=$SRC_D symlinks=$SRC_L file_bytes=$SRC_FB ($(numfmt --to=iec "$SRC_FB"))  du_apparent=$SRC_DU"
echo "SRC freeze -> $S/envs/of3_env.freeze.$STAMP.txt ($(wc -l < "$S/envs/of3_env.freeze.$STAMP.txt") pkgs)"

# ---------------------------------------------------------------- 4. clone the env
# A copy, not a fresh resolve: ~200 packages must stay at BYTE-identical versions so Run D's
# numbers are not silently moved by a resolver upgrade. Only the torch trio is then swapped.
# NOTE: this copy IS the quota test-write. There is no client-side quota tool on Engaging and df is
#       meaningless here, so a quota wall shows up as cp failing -- which aborts the job.
echo "cp -a starting $(date -Is)"
cp -a "$SRC" "$DST"
echo "cp -a done $(date -Is)"
read -r DST_F DST_D DST_L DST_FB < <(inv "$DST")
DST_DU=$(du -sb "$DST" | awk '{print $1}')
echo "DST inventory: regular=$DST_F dirs=$DST_D symlinks=$DST_L file_bytes=$DST_FB  du_apparent=$DST_DU"
echo "du_apparent delta = $((DST_DU - SRC_DU)) bytes (directory inode sizes; informational, NOT a gate)"
[ "$SRC_F"  -eq "$DST_F"  ] || { echo "ABORT: regular-file count $SRC_F != $DST_F -- copy is short (quota?)"; exit 1; }
[ "$SRC_D"  -eq "$DST_D"  ] || { echo "ABORT: directory count $SRC_D != $DST_D -- copy is short"; exit 1; }
[ "$SRC_L"  -eq "$DST_L"  ] || { echo "ABORT: symlink count $SRC_L != $DST_L -- copy is short"; exit 1; }
[ "$SRC_FB" -eq "$DST_FB" ] || { echo "ABORT: regular-file bytes $SRC_FB != $DST_FB -- copy is short (quota?)"; exit 1; }
echo "copy gate PASSED: every entry type and every content byte accounted for"

# venv console-script shebangs are absolute and still point at SRC. We invoke bin/python directly,
# but leaving them wrong would silently run the OLD env from any entry point.
# grep exits 1 on no-match, which under `set -e` would kill the script -> || true.
NSH=$(grep -rl "^#\!$SRC/bin/" "$DST/bin" 2>/dev/null | wc -l || true)
echo "shebangs pointing at SRC: $NSH"
if [ "$NSH" -gt 0 ]; then
  grep -rl "^#\!$SRC/bin/" "$DST/bin" 2>/dev/null | xargs sed -i "s|^#\!$SRC/bin/|#\!$DST/bin/|" || true
fi
NSH_AFTER=$(grep -rl "^#\!$SRC/bin/" "$DST/bin" 2>/dev/null | wc -l || true)
[ "$NSH_AFTER" -eq 0 ] || { echo "ABORT: $NSH_AFTER shebangs still point at SRC"; exit 1; }
sed -i "s|$SRC|$DST|g" "$DST/pyvenv.cfg" || true

# ---------------------------------------------------------------- 5. swap the torch trio to cu128
echo "=== installing cu128 trio $(date -Is) ==="
"$DST/bin/pip" install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 \
    "torch==2.7.1+cu128" "torchvision==0.22.1+cu128" "torchaudio==2.7.1+cu128"

# ---------------------------------------------------------------- gate 6: the versions are what we asked for
"$DST/bin/python" - <<'PYV' || { echo "ABORT: torch identity wrong"; exit 1; }
import sys
import torch, torchvision, torchaudio
print("torch      ", torch.__version__)
print("torchvision", torchvision.__version__)
print("torchaudio ", torchaudio.__version__)
print("torch.version.cuda", torch.version.cuda)
assert torch.__version__ == "2.7.1+cu128", torch.__version__
assert torchvision.__version__ == "0.22.1+cu128", torchvision.__version__
assert torchaudio.__version__ == "2.7.1+cu128", torchaudio.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
print("torch identity OK")
PYV

# ---------------------------------------------------------------- gate 7: THE arch gate
# This is the whole point of the rebuild. It must carry sm_120 for the RTX Pro 6000 AND keep sm_90
# so the H200 node stays usable -- if sm_90 were dropped, this env could not replace of3_env.
LIB=$SITE_DST/torch/lib/libtorch_cuda.so
[ -f "$LIB" ] || { echo "ABORT: $LIB missing"; exit 1; }
ARCH_FOUND=$(strings "$LIB" | grep -oE 'sm_[0-9]+' | sort -u | tr '\n' ' ' || true)
echo "libtorch_cuda.so arches: $ARCH_FOUND"
echo "$ARCH_FOUND" | grep -qw sm_120 || { echo "ABORT: cu128 libtorch has NO sm_120 -- the premise of this build is wrong"; exit 1; }
echo "$ARCH_FOUND" | grep -qw sm_90  || { echo "ABORT: cu128 libtorch DROPPED sm_90 -- it cannot replace of3_env on the H200"; exit 1; }
echo "arch gate PASSED: sm_90 and sm_120 both present"

# ---------------------------------------------------------------- 8. what else moved
"$DST/bin/pip" freeze > "$S/envs/of3_env_sm120.freeze.$STAMP.txt"
echo "=== freeze diff (SRC -> DST); anything beyond the torch trio + nvidia-*/triton is collateral ==="
diff "$S/envs/of3_env.freeze.$STAMP.txt" "$S/envs/of3_env_sm120.freeze.$STAMP.txt" || true

# ---------------------------------------------------------------- 9. preserve the cu126 extension
# The repo-root .so is gitignored+untracked, built for the H200 path. Relocate a copy into of3_env's
# site-packages so that path is unaffected once the repo root no longer supplies one.
BK=$REPO/.so_backup_$STAMP
mkdir -p "$BK"
cp -a "$REPO/$SO" "$BK/$SO"
cp -a "$REPO/$SO" "$SITE_SRC/$SO"
echo "cu126 .so backed up to $BK and installed into of3_env site-packages"
"$SRC/bin/python" -c "import importlib,sys; sys.path=[p for p in sys.path if p not in ('','$REPO')]; m=importlib.import_module('attn_core_inplace_cuda'); print('of3_env imports from', m.__file__)" \
  || { echo "ABORT: of3_env cannot import the relocated .so -- NOT removing the repo copy"; exit 1; }

# ---------------------------------------------------------------- 10. build the fat extension
echo "=== building attn_core_inplace_cuda for $ARCHES $(date -Is) ==="
cd "$REPO"
rm -rf build/lib.linux-x86_64-cpython-311 build/temp.linux-x86_64-cpython-311
OPENFOLD_CUDA_ARCH_LIST="$ARCHES" "$DST/bin/python" setup.py build_ext --build-lib build/sm120
NEW_SO=$(ls build/sm120/$SO)
[ -f "$NEW_SO" ] || { echo "ABORT: extension did not build"; exit 1; }
echo "built $NEW_SO"

# gate: the cubins actually inside it
echo "=== cuobjdump embedded cubins ==="
CUBINS=$(cuobjdump --list-elf "$NEW_SO" | grep -oE 'sm_[0-9]+' | sort -u | tr '\n' ' ' || true)
echo "extension cubins: $CUBINS"
echo "$CUBINS" | grep -qw sm_120 || { echo "ABORT: extension has no sm_120 cubin"; exit 1; }
echo "$CUBINS" | grep -qw sm_90  || { echo "ABORT: extension has no sm_90 cubin"; exit 1; }
cp -a "$NEW_SO" "$SITE_DST/$SO"
echo "fat .so installed into of3_env_sm120 site-packages"

# ---------------------------------------------------------------- 11. retire the shadowing repo copy
mv "$REPO/$SO" "$BK/$SO.was_at_repo_root"
echo "repo root .so retired -> $BK/$SO.was_at_repo_root"
for E in "$SRC" "$DST"; do
  "$E/bin/python" -c "import importlib; m=importlib.import_module('attn_core_inplace_cuda'); print('$E ->', m.__file__)" \
    || { echo "ABORT: $E cannot import attn_core_inplace_cuda after retiring the repo copy"; exit 1; }
done

# ---------------------------------------------------------------- 12. does openfold still import
cd "$REPO"
PYTHONPATH="$REPO/openfold" "$DST/bin/python" - <<'PYIMP' || { echo "ABORT: openfold import broke under cu128"; exit 1; }
import torch
import openfold.model.structure_module as sm
import openfold.utils.kernel.attention_core as ac
print("openfold structure_module + attention_core import OK under", torch.__version__)
PYIMP

# ---------------------------------------------------------------- 13. record the recipe
{
  echo "# of3_env_sm120 -- build recipe (generated by prune_work/build_of3_env_sm120.sbatch)"
  echo "# job=$SLURM_JOB_ID  finished=$(date -Is)  host=$(hostname)"
  echo "# WHY: of3_env (torch 2.7.1+cu126) has no sm_120 kernels; 22526403 died in 7 s on node4007."
  echo "base_env          = $SRC   (cp -a, NOT re-resolved)"
  echo "new_env           = $DST"
  echo "python            = 3.11.12 (uv venv, uv 0.11.20)"
  echo "cuda_module       = cuda/12.9.1   (nvcc $NVCC_VER, gcc $(gcc -dumpversion))"
  echo "index_url         = https://download.pytorch.org/whl/cu128"
  echo "pins              = torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128"
  echo "libtorch_arches   = $ARCH_FOUND"
  echo "ext_arch_list     = OPENFOLD_CUDA_ARCH_LIST=$ARCHES"
  echo "ext_cubins        = $CUBINS"
  echo "ext_location      = <env>/lib/python3.11/site-packages/$SO  (per-env, repo root no longer supplies one)"
  echo "cu126_so_backup   = $BK"
  echo "src_freeze        = $S/envs/of3_env.freeze.$STAMP.txt"
  echo "dst_freeze        = $S/envs/of3_env_sm120.freeze.$STAMP.txt"
  echo "repo_commit       = $(git -C "$REPO" rev-parse HEAD)"
  echo "REMAINING         = a real fwd+bwd on an sm_120 GPU (prune_work/verify_sm120_gpu.sbatch)"
} > "$REC"
echo "=== recipe -> $REC ==="
cat "$REC"
echo "BUILD OK $(date -Is)"
