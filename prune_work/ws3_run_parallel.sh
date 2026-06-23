#!/bin/bash
# WS3 multi-start hallucination across ALL 4 A6000 GPUs in parallel, then pick best + render.
# rep0 = deterministic zeros init; rep1-3 = gaussian-init seeds (diversity). Best chosen by TM(slim,target).
set -e
cd /home/jupyter-chenxi/openfold
export PYTHONPATH=/home/jupyter-chenxi/openfold/openfold
PY=/home/jupyter-chenxi/miniconda3/envs/cue_openfold_gated/bin/python
PYMOL=/home/jupyter-chenxi/miniconda3/envs/pymol/bin/pymol
ROOT=${OUT_ROOT:-/home/jupyter-chenxi/runs/ws3_multistart}
TARGET=${TARGET_PDB:-/home/jupyter-chenxi/data/7ad5_example/7ad5_A_cath_3.40.50.720_0_cg2all.pdb}
STEPS=${STEPS:-300}
[ -f "$TARGET" ] || { echo "ERROR: target not found: $TARGET"; exit 1; }
mkdir -p "$ROOT"

D=prune_work/ws3_hallucinate_slim.py
CUDA_VISIBLE_DEVICES=0 $PY $D --target_pdb "$TARGET" --init_seq 0                  --steps $STEPS --out_dir "$ROOT/rep0" > "$ROOT/rep0.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY $D --target_pdb "$TARGET" --init_seq gaussian --seed 1  --steps $STEPS --out_dir "$ROOT/rep1" > "$ROOT/rep1.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 $PY $D --target_pdb "$TARGET" --init_seq gaussian --seed 2  --steps $STEPS --out_dir "$ROOT/rep2" > "$ROOT/rep2.log" 2>&1 &
CUDA_VISIBLE_DEVICES=3 $PY $D --target_pdb "$TARGET" --init_seq gaussian --seed 3  --steps $STEPS --out_dir "$ROOT/rep3" > "$ROOT/rep3.log" 2>&1 &
wait
echo "=== all 4 replicas done ==="
$PY prune_work/ws3_pick_best.py "$ROOT"
echo "=== render best (pymol + legend) ==="
$PYMOL -cq prune_work/ws3_render_pymol.py -- "$ROOT/best"
$PY prune_work/ws3_legend.py "$ROOT/best"
echo "=== DONE ==="
ls -la "$ROOT"/best/*.png
