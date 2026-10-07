#!/bin/bash
#SBATCH -J eclean_hash
#SBATCH -p mit_normal
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH -t 04:00:00
#SBATCH -o /home/chenxiou/e_clean/hash-%j.log
R=/orcd/compute/so3/002/chenxi/of_run
H=/home/chenxiou/e_clean/hash_sample.sh
for pair in "mmcif_files data/mmcif_files" "openproteinset_aln data/openproteinset_aln" "templates_band pp1c_work/templates_band" "templates_merged pp1c_work/templates_merged" "t4_pool_snapshot data/t4_pool_snapshot"; do
  set -- $pair
  bash $H $R/$2 /home/chenxiou/e_clean/sample_$1.txt /home/chenxiou/e_clean/hash_engaging_$1.txt
done
sha256sum $R/data/ckpt/* | tee /home/chenxiou/e_clean/hash_engaging_ckpt.txt
echo done
