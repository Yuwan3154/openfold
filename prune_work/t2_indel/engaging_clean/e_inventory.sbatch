#!/bin/bash
#SBATCH -J eclean_inv
#SBATCH -p mit_normal
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 12:00:00
#SBATCH -o /home/chenxiou/e_clean/inv-%j.log
# READ-ONLY inventory of /orcd/compute/so3/002/chenxi/of_run (user 10-06: find the unique data, then free the SSD).
# One manifest per component: relative path <TAB> size <TAB> mtime, gzipped. Nothing is modified.
R=/orcd/compute/so3/002/chenxi/of_run
O=/home/chenxiou/e_clean/manifests
mkdir -p $O
rc=0
for c in data/ckpt data/mmcif_cache_a6000.json data/mmcif_cache_orcd_af3.json data/mmcif_files data/obsolete.dat \
         data/openproteinset_aln data/t4_pool_snapshot pp1c_work/alias pp1c_work/index_band.npz pp1c_work/index_merged.npz \
         pp1c_work/prefiltered_counts.npz pp1c_work/qmap_all_v2.npz pp1c_work/t2_chain_alias.csv pp1c_work/t2_chain_alias.npz \
         pp1c_work/templates_band pp1c_work/templates_merged pp1c_work/train_chains_ok.list prune_work pylibs runs/runD_perpos; do
  f=$O/$(echo $c | tr '/' '_').tsv.gz
  [ -e "$R/$c" ] || { echo "ABSENT $c"; rc=1; continue; }
  find "$R/$c" -type f -printf '%P\t%s\t%T@\n' | gzip > $f || rc=$?
  n=$(zcat $f | wc -l); b=$(zcat $f | awk -F'\t' '{s+=$2} END {printf "%.0f", s}')
  echo "COMP $c files=$n bytes=$b ($(awk -v b=$b 'BEGIN{printf "%.1f GB", b/1e9}'))"
done
echo "rc_final=$rc"
exit $rc
