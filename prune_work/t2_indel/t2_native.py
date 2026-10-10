"""Native-structure helpers shared by the T2 pipeline stages (numpy only).
load_npz reads a native written by t2_extract_natives.py (label-ID keyed, AF atom37 order). Missing backbone atoms of a kept residue are filled with its CA in the backbone array that
indel_edit.edit() uses (edit geometry only; the sampler is told those atoms are unknown, see t2_stage_b.build_inputs). native_ref() resolves a chains-file line: either 'id' (npz under
<natives_dir>/<id[1:3]>/<id>.npz) or the legacy 'id<TAB>native.pdb'.
"""
import os

import numpy as np

BB_AF = [0, 1, 2, 4]    # AF atom37: N, CA, C, O (CB is 3)
STANDARD = set("ACDEFGHIKLMNPQRSTVWY")


def load_npz(path):
    z = np.load(path)
    pos, mask = z["coords"].astype(np.float32), z["mask"].astype(bool)
    bb, bm = pos[:, BB_AF].copy(), mask[:, BB_AF]
    for j in range(4):
        bb[~bm[:, j], j] = pos[~bm[:, j], 1]
    return dict(pos=pos, mask=mask, bb=bb.astype(np.float64), names=str(z["sequence"]), complete=bm.all(1))


def shard_path(root, cid, ext):
    """<root>/<id[1:3]>/<id><ext>: the processed-file layout of proteina (keeps every directory far below 1024 files)."""
    return os.path.join(root, cid[1:3], cid + ext)


def native_ref(line, natives_dir):
    parts = line.rstrip("\n").split("\t")
    if len(parts) == 2:
        return parts[0], parts[1]
    cid = parts[0]
    return cid, shard_path(natives_dir, cid, ".npz")
