#!/bin/bash
#SBATCH -J eclean_pack
#SBATCH -p mit_data_transfer
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 3-00:00:00
#SBATCH -o /home/chenxiou/e_clean/pack-%j.log
# READ-ONLY on of_run: stream the UNIQUE data to the SuperCloud hub as three tars (no staging on Engaging disks).
# Unique (not duplicated elsewhere, decided from manifests + sha256 samples): runs/runD_perpos (ckpts+logs, run-local t4_pool),
# data/ckpt/best-053-040500.ckpt (Run C v2 ep53 = Run D init; absent on the A6000), data/mmcif_cache_orcd_af3.json,
# the 4 rank*_index.jsonl of the T4 snapshot, small pp1c_work index files, prune_work (eval_out, lists_pdb, val_expanded), pylibs.
# Per-file sha256 of every ckpt is written for the destination check.
R=/orcd/compute/so3/002/chenxi/of_run
E=/home/chenxiou/e_clean
ssh -o BatchMode=yes -o ConnectTimeout=30 SuperCloud 'mkdir -p ~/e_clean/xfer' || { echo "no route to SuperCloud"; echo rc_final=1; exit 1; }
cd $R || exit 1
find runs/runD_perpos/lightning_logs data/ckpt -name '*.ckpt' -print0 | xargs -0 sha256sum > $E/ckpt_sha256.txt
echo "ckpt sha256 lines: $(wc -l < $E/ckpt_sha256.txt)"
stream() {
  name=$1; shift
  tar cf - -C $R "$@" | tee >(sha256sum | cut -d' ' -f1 > $E/sha_$name.txt) | ssh -o BatchMode=yes SuperCloud "cat > ~/e_clean/xfer/$name.tar"
  ps=("${PIPESTATUS[@]}")
  echo "STREAM $name tar=${ps[0]} tee=${ps[1]} ssh=${ps[2]} sha=$(cat $E/sha_$name.txt)"
  [ "${ps[0]}" = 0 ] && [ "${ps[2]}" = 0 ]
}
rc=0
stream runD_logs runs/runD_perpos/lightning_logs runs/runD_perpos/val_population_index.json || rc=1
stream runD_t4pool runs/runD_perpos/t4_pool || rc=1
stream misc data/ckpt data/mmcif_cache_orcd_af3.json data/obsolete.dat data/t4_pool_snapshot/rank0_index.jsonl data/t4_pool_snapshot/rank1_index.jsonl data/t4_pool_snapshot/rank2_index.jsonl data/t4_pool_snapshot/rank3_index.jsonl pp1c_work/alias pp1c_work/index_band.npz pp1c_work/index_merged.npz pp1c_work/prefiltered_counts.npz pp1c_work/qmap_all_v2.npz pp1c_work/t2_chain_alias.csv pp1c_work/t2_chain_alias.npz pp1c_work/train_chains_ok.list prune_work pylibs || rc=1
ssh -o BatchMode=yes SuperCloud 'cd ~/e_clean/xfer && ls -l *.tar && sha256sum *.tar' | tee $E/dest_sha.txt
echo "rc_final=$rc"
exit $rc
