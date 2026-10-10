#!/bin/bash
#SBATCH -J t2synth
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclude=node5101
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -o /orcd/scratch/orcd/011/chenxiou/t2_synth_v2/logs/%x_%a_%j.log
# Production run of the T2 synthetic-template pipeline for the T8 chain list (RAW 137). One array task = one shard (chain i of the list goes to shard i % NSHARD), 1 L40S each.
# Resumable at every stage (natives npz, stage A json, stage B metrics.csv are skipped when present); resubmit the same array to continue after the 6 h limit.
# Submit: sbatch --array=0-1 --export=ALL,NSHARD=2 t2_prod.sbatch   (NSHARD defaults to 2)
set -uo pipefail
NSHARD=${NSHARD:-2}
S=${SLURM_ARRAY_TASK_ID:-0}
CH=/orcd/scratch/orcd/011/chenxiou
LIST=$CH/t8_template_list
R=$CH/t2pipe
OUT=$CH/t2_synth_v2
W=$OUT/work
PR=/orcd/pool/006/chenxiou/proteina/data/pdb_train/processed
PTH=$R/hf/hub/models--biohub--esmc-300m-2024-12/snapshots/7f10b20ae75017b2dbc884070e03434515709a8d/data/weights/esmc_300m_2024_12_v0.pth
GEOM=$HOME/orcd/scratch/t2pipe_data/geom_pools.csv
CODE=$R/repo/prune_work/t2_indel
ESMC_DTYPE=${ESMC_DTYPE:-fp32}
mkdir -p $OUT/logs $OUT/templates $W/natives $W/stageA
cd $W

# 1. the chain list is used only when READY.txt is present and its count and sha256 match the file (T8 handoff)
want_n=$(awk '$1=="count"{print $2}' $LIST/READY.txt)
want_sha=$(awk '$1=="sha256"{print $2}' $LIST/READY.txt)
got_n=$(wc -l < $LIST/template_chains.txt)
got_sha=$(sha256sum $LIST/template_chains.txt | awk '{print $1}')
echo "list: count $got_n (READY $want_n) sha $got_sha"
if [ -z "$want_n" ] || [ "$want_n" != "$got_n" ] || [ "$want_sha" != "$got_sha" ]; then echo "LIST VERIFICATION FAILED"; exit 1; fi

# 2. this shard's chains, natives from the proteina processed .pt files (label ids), lengths for the length-sorted order
git -C $R/repo pull -q --ff-only || exit 1
awk -v s=$S -v n=$NSHARD '(NR-1)%n==s' $LIST/template_chains.txt > ids_shard$S.txt
[ -s $W/pt_index.tsv ] || { echo "missing $W/pt_index.tsv (run the index step once)"; exit 1; }
PYTHONPATH=$CODE $HOME/.conda/envs/proteina/bin/python $CODE/t2_extract_natives.py --chains-file ids_shard$S.txt --processed-dir $PR --index $W/pt_index.tsv --out-dir $W/natives --proteina-src $HOME/orcd/scratch/proteina_tri --lengths-out lengths_shard$S.tsv || exit 1
sort -t$'\t' -k2,2n -k1,1 lengths_shard$S.tsv | cut -f1 > order_shard$S.txt
echo "shard $S: $(wc -l < ids_shard$S.txt) listed, $(wc -l < order_shard$S.txt) usable"

# 3. stage A (edit plans + ESMC-300M fill), then stage B (partial diffusion + gates), both in length order
HF_HUB_OFFLINE=1 PYTHONPATH=$CODE $HOME/.conda/envs/esmfold2_env/bin/python $CODE/t2_stage_a.py --chains-file order_shard$S.txt --natives-dir $W/natives --out-dir $W/stageA --n-draws 64 --esmc-pth $PTH --batch-size 64 --esmc-dtype $ESMC_DTYPE
rc=$?
if [ $rc -ne 0 ]; then echo "stage A failed rc=$rc"; exit $rc; fi
export PYTHONPATH=$R/pp/src:$R/extra:$CODE PYTHONUNBUFFERED=1 PROTPARDELLE_KEEP_TRAJECTORY=0 PROTPARDELLE_EMPTY_CACHE_PER_STEP=0
$CH/pp1c/env/bin/python $CODE/t2_stage_b.py --stage-a-dir $W/stageA --chains-file order_shard$S.txt --natives-dir $W/natives --out-dir $OUT/templates --geom-ref $GEOM --fast --store passing
rc=$?
echo "stage B rc=$rc"
exit $rc
