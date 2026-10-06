#!/bin/bash
#SBATCH -J t2indel_cleanup
#SBATCH -p xeon-p8
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=/home/gridsan/cou/logs/t2indel_cleanup-%j.out
#SBATCH --error=/home/gridsan/cou/logs/t2indel_cleanup-%j.err
# User 10-06 13:40: "Delete if done with all analyses. Save the summary statistics before deleting."
# Summary statistics were saved locally first (_box_snapshot/t2_indel/summary_stats.csv + every per-item score/diag csv).
# Literal ALLOWLIST only; each entry must sit directly under $D, be a real directory/file (not a symlink).
# KEPT on purpose (the pool build needs them): inputs, inputs_compF, inputs_raygun_n0.5_full, inputs_rg3_full,
# sweep_gly, sweep_compF, sweep_raygun_full, sweep_rg3, raygun_seqs3/raygun seq jsons, all scores_*/diag_*/*_scores/*_diag dirs.
D=/home/gridsan/cou/t2_indel_data
TARGETS="inputs_comp inputs_comp_bb inputs_comp_cg inputs_compF_bb inputs_compF_cg inputs_raygun_n0.5 inputs_rg2_bb inputs_rg2_full inputs_rg3_bb sweep_rg2 sweep_raygun_n0.5 sweep_comp pilot pilot200 pilot_tmp pilot200_tmp cg_test raygun_seqs2_n0.5.json"
rc=0
for t in $TARGETS; do
  p=$D/$t
  if [ -L "$p" ]; then echo "SKIP symlink $p"; rc=1; continue; fi
  if [ -e "$p" ]; then echo "BEFORE $t files=$(find "$p" -type f | wc -l) size=$(du -sh "$p" | cut -f1)"; else echo "ABSENT $t"; fi
done
echo "KEEP check:"
for k in inputs inputs_compF inputs_raygun_n0.5_full inputs_rg3_full sweep_gly sweep_compF sweep_raygun_full sweep_rg3; do
  [ -d "$D/$k" ] && echo "KEEP $k files=$(find "$D/$k" -type f | wc -l)" || { echo "MISSING KEEP $k"; rc=1; }
done
[ $rc = 0 ] || { echo "aborting: preconditions failed"; echo "rc_final=1"; exit 1; }
for t in $TARGETS; do
  p=$D/$t
  case "$p" in $D/*) ;; *) echo "REFUSE $p"; exit 1;; esac
  [ -e "$p" ] && rm -rf -- "$p"
done
for t in $TARGETS; do [ -e "$D/$t" ] && { echo "STILL PRESENT $t"; rc=1; }; done
for k in inputs inputs_compF inputs_raygun_n0.5_full inputs_rg3_full sweep_gly sweep_compF sweep_raygun_full sweep_rg3; do echo "AFTER keep $k files=$(find "$D/$k" -type f | wc -l)"; done
echo "rc_final=$rc"
exit $rc
