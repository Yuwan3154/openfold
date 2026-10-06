"""Variable-length indel template pool: build (ragged per-chain npz + index) and read.

One npz per chain, templates of DIFFERENT lengths stored in CSR layout (template i = rows res_offsets[i]:res_offsets[i+1]).
Keys (N templates, R = sum of lengths, A = present atoms):
  n_templates ()            res_offsets (N+1,) int64       atom_offsets (N+1,) int64 (offsets into `coords`)
  atom_mask (R,37) bool     coords (A,3) float32           aatype (R,) int8   [generation sequence, protpardelle restype order]
  residue_index (R,) int32  contiguous 1..L_i per template orig_idx (R,) int32 native-frame index, -1 = inserted
  arm (N,) <U16   model (N,) <U8   rewind (N,) int16   draw (N,) int16   L_native () int32   chain () <U16
  tm_native (N,) float32   tm_template (N,) float32      ops_json () str (edit plan per template, list)
  aa_order () str  atom_types () str (atom37 names, space separated)
Design stage (added by merge_design): design_aatype (32, R) int8, design_score (N,32), design_global_score (N,32),
  design_recovery (N,32), design_meta_json () str (model, temperature, seed, ...).
"""
import json
import os
import zlib

import numpy as np

ATOM37 = ("N CA C CB O CG CG1 CG2 OG OG1 SG CD CD1 CD2 ND1 ND2 OD1 OD2 SD CE CE1 CE2 CE3 NE NE1 NE2 OE1 OE2 CH2 NH1 "
          "NH2 OH CZ CZ2 CZ3 NZ OXT")
AA_ORDER = "ARNDCQEGHILKMFPSTWYV"


def shard_path(root, chain):
    return os.path.join(root, f"shard{zlib.crc32(chain.encode()) % 1000:04d}", f"{chain}.npz")


def pack_chain(chain, items):
    """items: list of dicts with keys arm, model, rewind, draw, coords (A_i,3), atom_mask (L_i,37), aatype (L_i,),
    residue_index_orig (L_i,), orig_idx (L_i,), ops, tm_native, tm_template, L_native."""
    L = np.array([len(it["aatype"]) for it in items], np.int64)
    A = np.array([len(it["coords"]) for it in items], np.int64)
    for it in items:
        assert int(it["atom_mask"].sum()) == len(it["coords"]), "atom_mask does not match the packed coords"
        assert len(it["orig_idx"]) == len(it["aatype"])
    return dict(
        n_templates=np.int32(len(items)),
        res_offsets=np.concatenate([[0], np.cumsum(L)]),
        atom_offsets=np.concatenate([[0], np.cumsum(A)]),
        atom_mask=np.concatenate([it["atom_mask"] for it in items]).astype(bool),
        coords=np.concatenate([it["coords"] for it in items]).astype(np.float32),
        aatype=np.concatenate([it["aatype"] for it in items]).astype(np.int8),
        residue_index=np.concatenate([np.arange(1, len(it["aatype"]) + 1) for it in items]).astype(np.int32),
        orig_idx=np.concatenate([it["orig_idx"] for it in items]).astype(np.int32),
        arm=np.array([it["arm"] for it in items]), model=np.array([it["model"] for it in items]),
        rewind=np.array([it["rewind"] for it in items], np.int16), draw=np.array([it["draw"] for it in items], np.int16),
        L_native=np.int32(items[0]["L_native"]), chain=np.array(chain),
        tm_native=np.array([it["tm_native"] for it in items], np.float32),
        tm_template=np.array([it["tm_template"] for it in items], np.float32),
        ops_json=np.array(json.dumps([it["ops"] for it in items])),
        aa_order=np.array(AA_ORDER), atom_types=np.array(ATOM37),
    )


def read_template(path, i):
    z = np.load(path)
    r0, r1 = int(z["res_offsets"][i]), int(z["res_offsets"][i + 1])
    a0, a1 = int(z["atom_offsets"][i]), int(z["atom_offsets"][i + 1])
    out = dict(aatype=z["aatype"][r0:r1], atom_mask=z["atom_mask"][r0:r1], coords=z["coords"][a0:a1],
               residue_index=z["residue_index"][r0:r1], orig_idx=z["orig_idx"][r0:r1], arm=str(z["arm"][i]),
               model=str(z["model"][i]), rewind=int(z["rewind"][i]), draw=int(z["draw"][i]),
               tm_native=float(z["tm_native"][i]), tm_template=float(z["tm_template"][i]),
               ops=json.loads(str(z["ops_json"]))[i], L_native=int(z["L_native"]))
    if "design_aatype" in z.files:
        out["design_aatype"] = z["design_aatype"][:, r0:r1]
        out["design_score"], out["design_global_score"] = z["design_score"][i], z["design_global_score"][i]
        out["design_recovery"] = z["design_recovery"][i]
    return out


def atom37_coords(t):
    """(L,37,3) with zeros for absent atoms, from a read_template dict."""
    full = np.zeros((len(t["aatype"]), 37, 3), np.float32)
    full[t["atom_mask"]] = t["coords"]
    return full
