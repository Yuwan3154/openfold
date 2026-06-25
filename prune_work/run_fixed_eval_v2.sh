#!/bin/bash
# Launcher for the faithful full-MSA fixed-eval sweep (test 2). Creates the fixed eval sets from
# PRE-CUTOFF train proteins (post-cutoff val lacks OpenProteinSet alignments -> validate stalls), then
# runs the driver: validate(init) -> train each M (4-way) -> validate each raw ckpt (4-way) -> table.
cd /home/jupyter-chenxi/openfold
. ~/miniconda3/etc/profile.d/conda.sh && conda deactivate && conda activate cue_openfold_gated
export PYTHONPATH=/home/jupyter-chenxi/openfold/openfold
ulimit -n 65536
L=/home/jupyter-chenxi/prune_work/lists_pdb
head -4 "$L/slim_struct_train.list" > "$L/fe_test4.list"    # tiny training end-val (ignored)
head -16 "$L/slim_struct_train.list" > "$L/fe_eval16.list"  # fixed eval set (before/after)
python prune_work/fixed_eval_v2.py --mults ${FE_MULTS:-0.003,0.01,0.03,0.1,0.3,1,3} --steps ${FE_STEPS:-100} --gpus 0,1,2,3
