#!/bin/bash
#SBATCH -J eclean_delete
#SBATCH -p mit_data_transfer
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH -t 3-00:00:00
#SBATCH -o /home/chenxiou/e_clean/delete-%j.log
# IRREVERSIBLE (user 10-06: "delete all of them on Engaging" once unique data is safe). PRECONDITIONS, all checked here:
#  1. the path is the exact real directory (not a symlink);
#  2. the A6000 verification passed (sha256 of 3 tars, member counts, 33/33 ckpt sha256): recorded by the caller as the
#     existence of /home/chenxiou/e_clean/A6000_VERIFIED (written by me only after rc 0);
#  3. file count == the inventory total (1,934,578 regular files = 1,934,519 data + 59 empty markers); any NEW file => abort, nothing deleted.
R=/orcd/compute/so3/002/chenxi/of_run
EXPECT=1934578   # 1,934,519 inventoried data files + 59 EMPTY top-level .done_* migration markers (Sep 2-4; checked: 59 names, size 0)
[ -d "$R" ] && [ ! -L "$R" ] && [ "$(readlink -f "$R")" = "$R" ] || { echo "ABORT path check"; echo rc_final=1; exit 1; }
[ -f /home/chenxiou/e_clean/A6000_VERIFIED ] || { echo "ABORT: no A6000_VERIFIED marker"; echo rc_final=1; exit 1; }
n=$(find "$R" -type f | wc -l)
echo "files now: $n expected: $EXPECT"
[ "$n" = "$EXPECT" ] || { echo "ABORT: file count differs from the inventory"; echo rc_final=1; exit 1; }
echo "BEFORE: $(du -sh "$R" 2>/dev/null | cut -f1)  $(date)"
rm -rf --one-file-system -- "$R"
rc=$?
[ -e "$R" ] && { echo "STILL PRESENT"; rc=1; }
echo "AFTER: exists=$([ -e "$R" ] && echo yes || echo no) rc_rm=$rc $(date)"
ls -la /orcd/compute/so3/002/chenxi | head -5
echo "rc_final=$rc"
exit $rc
