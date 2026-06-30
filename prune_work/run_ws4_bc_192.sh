#!/bin/bash
# WS4-BC L<=192 escalation (self-advancing): WAIT for the L128 train sweep to finish (train_done.marker)
# -> PHASE3 cache L<=192 (4-way, resumes over the <=128 set) -> ckpt-double-backprop SMOKE GATE (validates
# the use_reentrant=False fix at L<=192) -> PHASE4 4-way 192 train sweep (USE_CKPT=1). Agent not the link.
BASE=/home/jupyter-chenxi
PY=$BASE/miniconda3/envs/cue_openfold_gated/bin/python
L=/tmp/ws4bc
mkdir -p $L
cd $BASE/prune_work || exit 1

echo "=== 192 escalation: waiting for L128 train_done.marker $(date) ==="
while [ ! -f $L/train_done.marker ]; do sleep 60; done
echo "=== L128 train done; PHASE3 cache L<=192 (SOURCE=all, 4-way) start $(date) ==="
for s in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$s SOURCE=all N=999999 MAXLEN=192 RECYCLE=1 USE_CKPT=1 nohup $PY cache_grads_bc.py $s 4 > $L/cache192_$s.log 2>&1 &
done
wait
NC=$(ls $BASE/data/grad_cache_bc/*.pt 2>/dev/null | wc -l)
echo "done=$NC $(date)" > $L/cache192_done.marker
echo "=== PHASE3 DONE cached=$NC $(date) ==="

echo "=== SMOKE GATE: checkpointed double-backprop at L<=192 (USE_CKPT=1, 1 chain) $(date) ==="
CUDA_VISIBLE_DEVICES=0 STUDENT=converged2 POINTS=soft STEPS=2 BATCH=1 VAL_N=2 MAXLEN=192 LR=1e-4 USE_CKPT=1 EVAL_EVERY=2 \
  $PY ws4_bc_train.py > $L/train192_smoke.log 2>&1
if grep -q "^DONE" $L/train192_smoke.log; then
  echo "OK $(date)" > $L/ckpt_smoke.marker
  echo "=== smoke OK; PHASE4 192 train sweep (USE_CKPT=1, 4-way) start $(date) ==="
  CUDA_VISIBLE_DEVICES=0 STUDENT=converged2 POINTS=soft STEPS=600 BATCH=4 VAL_N=80 MAXLEN=192 LR=2e-4 USE_CKPT=1 EVAL_EVERY=25 nohup $PY ws4_bc_train.py > $L/train192_soft.log 2>&1 &
  CUDA_VISIBLE_DEVICES=1 STUDENT=converged2 POINTS=both STEPS=600 BATCH=4 VAL_N=80 MAXLEN=192 LR=2e-4 USE_CKPT=1 EVAL_EVERY=25 nohup $PY ws4_bc_train.py > $L/train192_both.log 2>&1 &
  CUDA_VISIBLE_DEVICES=2 STUDENT=converged2 POINTS=soft STEPS=600 BATCH=8 VAL_N=80 MAXLEN=192 LR=2e-4 USE_CKPT=1 EVAL_EVERY=25 nohup $PY ws4_bc_train.py > $L/train192_soft_b8.log 2>&1 &
  CUDA_VISIBLE_DEVICES=3 STUDENT=converged2 POINTS=soft STEPS=600 BATCH=4 VAL_N=80 MAXLEN=192 LR=4e-4 USE_CKPT=1 EVAL_EVERY=25 nohup $PY ws4_bc_train.py > $L/train192_soft_hi.log 2>&1 &
  wait
  echo "done $(date)" > $L/train192_done.marker
  echo "=== PHASE4 DONE $(date) ==="
else
  echo "FAIL $(date)" > $L/ckpt_smoke.marker
  echo "=== SMOKE GATE FAILED — 192 train NOT launched; see train192_smoke.log ==="
fi
