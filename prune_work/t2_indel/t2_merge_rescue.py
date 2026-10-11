"""Add the rescued chains (rewind-100 run, user 10-10 'Yes') to the template tree. A rescued chain had ZERO survivors in the main run, so templates/<id[1:3]>/<id>.npz does not exist: the rescue npz (rewind 100 of 250 steps, 256 variants run, only passing ones stored, draw ids 0-255) is copied there, and the rescue metrics.csv (256 rows) is copied next to it as <id>.rescue.metrics.csv; the main <id>.metrics.csv (64 original variants) is NOT touched. A chain that already has an npz is never overwritten.
Run: python t2_merge_rescue.py --rescue-dir R/templates --templates T
"""
import argparse
import glob
import os
import shutil

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rescue-dir", required=True)
    ap.add_argument("--templates", required=True)
    a = ap.parse_args()
    n_new = n_skip_existing = n_zero = 0
    for f in sorted(glob.glob(os.path.join(a.rescue_dir, "*", "*.metrics.csv"))):
        cid = os.path.basename(f)[:-len(".metrics.csv")]
        src_npz = f[:-len(".metrics.csv")] + ".npz"
        shard = os.path.join(a.templates, cid[1:3])
        if not os.path.isfile(src_npz):
            n_zero += 1
            shutil.copy(f, os.path.join(shard, cid + ".rescue.metrics.csv"))
            continue
        dst_npz = os.path.join(shard, cid + ".npz")
        if os.path.isfile(dst_npz):
            n_skip_existing += 1
            continue
        assert pd.read_csv(os.path.join(shard, cid + ".metrics.csv")).pass_all.sum() == 0, (cid, "main run had survivors: not a rescue chain")
        shutil.copy(src_npz, dst_npz + ".tmp")
        os.replace(dst_npz + ".tmp", dst_npz)
        shutil.copy(f, os.path.join(shard, cid + ".rescue.metrics.csv"))
        n_new += 1
    print(f"rescue chains merged: {n_new} new npz, {n_skip_existing} already present (left alone), {n_zero} rescue chains still without a survivor (rescue metrics copied only)")


if __name__ == "__main__":
    main()
