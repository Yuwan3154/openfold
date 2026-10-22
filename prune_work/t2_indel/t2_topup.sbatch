#!/bin/bash
#SBATCH -J t2topup
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclude=node5101
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -o /orcd/compute/so3/002/chenxi/t2_synth_v2/logs/topup_%x_%a_%j.log
# One generation round of the top-up (T2_TOPUP_DIVERSITY_PLAN.md): stage A (new edit draws) + stage B (partial diffusion + gates) for the chains of <topup>/<ROUND>/ids_shard<array id>.txt
# (written by `t2_topup.py plan`; settings in <topup>/<ROUND>/round.json). Resumable (stage A json and stage B metrics.csv are skipped when present).
# Submit: sbatch --array=0-1 --export=ALL,ROUND=P1 t2_topup.sbatch   (backups: add -p mit_preemptable --gres=gpu:l40s:1 / gpu:h200:1 -c 8 --requeue, one process per shard)
set -uo pipefail
S=${SLURM_ARRAY_TASK_ID:-0}
ROUND=${ROUND:?set ROUND=P1|P2|P3|P4|R1|R2|R3|R4|R5|R6}
CH=/orcd/scratch/orcd/011/chenxiou
R=$CH/t2pipe
O=${T2_OUT:-/orcd/compute/so3/002/chenxi/t2_synth_v2}   # T8b regeneration: /orcd/compute/so3/002/chenxi/t2_synth_v2_t8b
W=$O/work
T=$O/topup/$ROUND
PTH=$R/hf/hub/models--biohub--esmc-300m-2024-12/snapshots/7f10b20ae75017b2dbc884070e03434515709a8d/data/weights/esmc_300m_2024_12_v0.pth
GEOM=$HOME/orcd/scratch/t2pipe_data/geom_pools.csv
REPO=${REPO_DIR:-$R/repo}
CODE=$REPO/prune_work/t2_indel
git -C $REPO pull -q --ff-only || echo "git pull failed (continuing with the checked-out code)"
[ -s $T/ids_shard$S.txt ] || { echo "no chain list $T/ids_shard$S.txt"; exit 1; }
PYJ=$CH/pp1c/env/bin/python
N=$($PYJ -c "import json;print(json.load(open('$T/round.json'))['n_draws'])")
D=$($PYJ -c "import json;print(json.load(open('$T/round.json'))['draw_start'])")
ARGS=$($PYJ -c "import json;print(json.load(open('$T/round.json'))['stage_b_args'])")
[ -n "$N" ] && [ -n "$D" ] && [ -n "$ARGS" ] || { echo "could not read $T/round.json"; exit 1; }
TAB=$($PYJ -c "import json;print(json.load(open('$T/round.json')).get('table', 't2'))")
if [ -n "${T2_OUT:-}" ] && [ "$TAB" != t8b ]; then echo "T2_OUT is set but $T/round.json was planned with table $TAB (use t2_topup.py plan --table t8b)"; exit 1; fi
cd $W   # stage B's sampler config refers to a relative test_dir path (as in t2_prod.sbatch)
echo "round $ROUND shard $S: $(wc -l < $T/ids_shard$S.txt) chains, $N draws from id $D, stage B args: $ARGS"
HF_HUB_OFFLINE=1 PYTHONPATH=$CODE $HOME/.conda/envs/esmfold2_env/bin/python $CODE/t2_stage_a.py --chains-file $T/ids_shard$S.txt --natives-dir $W/natives --out-dir $T/stageA --n-draws $N --draw-start $D --esmc-pth $PTH --batch-size 64 --esmc-dtype fp16
rc=$?
if [ $rc -ne 0 ]; then echo "stage A failed rc=$rc"; exit $rc; fi
export PYTHONPATH=$R/pp/src:$R/extra:$CODE PYTHONUNBUFFERED=1 PROTPARDELLE_KEEP_TRAJECTORY=0 PROTPARDELLE_EMPTY_CACHE_PER_STEP=0
$CH/pp1c/env/bin/python $CODE/t2_stage_b.py --stage-a-dir $T/stageA --chains-file $T/ids_shard$S.txt --natives-dir $W/natives --out-dir $T/templates --geom-ref $GEOM --fast $ARGS
rc=$?
echo "stage B rc=$rc"
exit $rc
