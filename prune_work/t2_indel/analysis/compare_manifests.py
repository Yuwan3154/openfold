"""Engaging of_run vs counterpart manifests (path,size identity) and sha256 samples: the comparison behind RAW §34. Run in e_clean/."""
import re

import numpy as np
import pandas as pd

R = lambda f: pd.read_csv(f, sep="\t", header=None, names=["p", "s", "m"], dtype={"p": str, "s": np.int64, "m": float}, keep_default_na=False)


def load_hash(f):
    d = {}
    for ln in open(f):
        m = re.match(r"^([0-9a-f]{64})\s+(?:\./)?(.+)$", ln.strip())
        if m:
            d[m.group(2)] = m.group(1)
    return d


def main():
    pairs = [("mmcif_files", "data_mmcif_files", "a6000_mmcif_files", "a6000"), ("openproteinset_aln", "data_openproteinset_aln", "a6000_openproteinset_aln", "a6000"),
             ("templates_band", "pp1c_work_templates_band", "a6000_templates_band", "a6000"), ("templates_merged", "pp1c_work_templates_merged", "sc_templates_merged", "sc"),
             ("t4_pool_snapshot", "data_t4_pool_snapshot", "a6000_t4_pool", "a6000")]
    for name, eng, oth, host in pairs:
        A, B = R(eng + ".tsv.gz"), R(oth + ".tsv.gz")
        ka, kb = set(zip(A.p, A.s)), set(zip(B.p, B.s))
        e, o = load_hash(f"hash_engaging_{name}.txt"), load_hash(f"hash_{host}_{name}.txt")
        same = sum(1 for k in e if o.get(k) == e[k])
        print(f"{name}: engaging {len(A)} files, identical (path,size) {len(ka & kb)}, engaging-only {len(ka - kb)}; sha256 sample {same}/{len(e)} identical")


if __name__ == "__main__":
    main()
