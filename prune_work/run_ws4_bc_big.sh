#!/bin/bash
# WS4-BC big pipeline (self-advancing, agent-independent): PHASE1 cache ALL chains up to L=128 (4-way,
# resumes over the 2k already done) -> PHASE2 4-way training sweep on the full 12k (soft/both x lr).
BASE=/home/jupyter-chenxi
PY=$BASE/miniconda3/envs/cue_openfold_gated/bin/python
L=/tmp/ws4bc
mkdir -p $L
cd $BASE/prune_work || exit 1

echo "=== PHASE1 big cache (SOURCE=all MAXLEN=128, 4-way) start $(date) ==="
for s in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$s SOURCE=all N=999999 MAXLEN=128 RECYCLE=1 nohup $PY cache_grads_bc.py $s 4 > $L/cache_$s.log 2>&1 &
done
wait
NC=$(ls $BASE/data/grad_cache_bc/*.pt 2>/dev/null | wc -l)
echo "done=$NC $(date)" > $L/cache_done.marker
echo "=== PHASE1 DONE $(date) cached=$NC ==="

echo "=== PHASE2 big train sweep (4-way, 12k data, STEPS=600) start $(date) ==="
CUDA_VISIBLE_DEVICES=0 STUDENT=converged2 POINTS=soft STEPS=600 BATCH=8 VAL_N=80 MAXLEN=128 LR=2e-4 EVAL_EVERY=25 nohup $PY ws4_bc_train.py > $L/train_soft.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 STUDENT=converged2 POINTS=both STEPS=600 BATCH=8 VAL_N=80 MAXLEN=128 LR=2e-4 EVAL_EVERY=25 nohup $PY ws4_bc_train.py > $L/train_both.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 STUDENT=converged2 POINTS=soft STEPS=600 BATCH=8 VAL_N=80 MAXLEN=128 LR=4e-4 EVAL_EVERY=25 nohup $PY ws4_bc_train.py > $L/train_soft_hi.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 STUDENT=converged2 POINTS=both STEPS=600 BATCH=8 VAL_N=80 MAXLEN=128 LR=4e-4 EVAL_EVERY=25 nohup $PY ws4_bc_train.py > $L/train_both_hi.log 2>&1 &
wait
echo "done $(date)" > $L/train_done.marker
echo "=== PHASE2 DONE $(date) ==="
