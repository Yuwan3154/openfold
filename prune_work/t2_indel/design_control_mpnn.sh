#!/bin/bash
# Stage 1 of the design control: ProteinMPNN (SAME settings as the pool: v_48_020, T 0.1, seed 37, 32 seqs) on the NATIVE backbones.
# usage: design_control_mpnn.sh <inputs_dir> <work_dir> <chain> [<chain> ...]   (env protpardelle)
IN=$1; W=$2; shift 2
M=/home/gridsan/cou/ProteinMPNN
rm -rf $W; mkdir -p $W/pdb
for c in "$@"; do cp $IN/$c/native.pdb $W/pdb/$c.pdb; done
python $M/helper_scripts/parse_multiple_chains.py --input_path $W/pdb --output_path $W/parsed.jsonl || exit 1
python $M/protein_mpnn_run.py --jsonl_path $W/parsed.jsonl --out_folder $W --num_seq_per_target 32 --sampling_temp 0.1 --seed 37 --batch_size 32 --model_name v_48_020 || exit 1
ls $W/seqs
