#!/bin/bash
# Test 3 (memory): with structure-module (IPA) checkpointing NOW ADDED on top of the
# existing evoformer+extra-MSA checkpointing, how far does double-backprop actually reach?
# PROBE_LONGEST smoke (worst-case: longest chains in val) at increasing MAXLEN, tiny STEPS
# just to read peak torch.cuda.max_memory_allocated(); no real training here.
BASE=/home/jupyter-chenxi
PY=$BASE/miniconda3/envs/cue_openfold_gated/bin/python
L=/tmp/ws4push
cd $BASE/prune_work || exit 1
echo "=== IPA-ckpt memory ladder $(date) ===" > $L/ipa_memladder.log
for ML in 144 160 176 192 208 224; do
  echo "--- MAXLEN=$ML ---" >> $L/ipa_memladder.log
  CUDA_VISIBLE_DEVICES=2 STUDENT=converged2 POINTS=soft STEPS=3 BATCH=2 VAL_N=2 MAXLEN=$ML LR=1e-4 \
    USE_CKPT=1 USE_IPA_CKPT=1 PROBE_LONGEST=1 EVAL_EVERY=3 \
    $PY ws4_bc_train.py > $L/ipa_smoke_$ML.log 2>&1
  if grep -q "^DONE" $L/ipa_smoke_$ML.log; then
    mem=$(grep "mem=" $L/ipa_smoke_$ML.log | tail -1 | grep -oE "mem=[0-9.]+GB")
    echo "OK MAXLEN=$ML $mem $(date)" | tee -a $L/ipa_memladder.log
  else
    echo "OOM/FAIL at MAXLEN=$ML $(date)" | tee -a $L/ipa_memladder.log
    tail -15 $L/ipa_smoke_$ML.log >> $L/ipa_memladder.log
    break
  fi
done
echo "=== ladder done $(date) ===" >> $L/ipa_memladder.log
