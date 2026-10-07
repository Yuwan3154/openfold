#!/bin/bash
#SBATCH -J eclean_diff
#SBATCH -p mit_data_transfer
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH -t 02:00:00
#SBATCH -o /home/chenxiou/e_clean/diff-%j.log
R=/orcd/compute/so3/002/chenxi/of_run
echo "== depth 1-2 entries (type path)"
find $R -mindepth 1 -maxdepth 2 -printf '%y %P\n' | sort
echo "== per-component regular-file counts"
for c in data/ckpt data/mmcif_cache_a6000.json data/mmcif_cache_orcd_af3.json data/mmcif_files data/obsolete.dat data/openproteinset_aln data/t4_pool_snapshot pp1c_work/alias pp1c_work/index_band.npz pp1c_work/index_merged.npz pp1c_work/prefiltered_counts.npz pp1c_work/qmap_all_v2.npz pp1c_work/t2_chain_alias.csv pp1c_work/t2_chain_alias.npz pp1c_work/templates_band pp1c_work/templates_merged pp1c_work/train_chains_ok.list prune_work pylibs runs/runD_perpos; do
  echo "$c $(find $R/$c -type f | wc -l)"
done
echo "== total regular files: $(find $R -type f | wc -l)"
