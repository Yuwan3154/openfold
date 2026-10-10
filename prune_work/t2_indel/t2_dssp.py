"""DSSP for the T2 pipeline filters with the T8 rules (handoff 10-10): pydssp with the PROLINE DONOR MASK (proline cannot donate an H-bond) and -1 for residues without a full N/CA/C/O
backbone. pydssp.assign returns 0 = loop ('-'), 1 = helix (H), 2 = strand (E). loop_fraction() = share of loop residues among the residues with a full backbone.
Needs pydssp (pip install pydssp; numpy only).
"""
import numpy as np
import pydssp

CODE = {"-": 0, "H": 1, "E": 2}


def assign(bb, is_pro, complete=None):
    """bb (L,4,3) N,CA,C,O; is_pro (L,) bool; complete (L,) bool (default all) -> int array, -1 where the backbone is incomplete."""
    ss = np.asarray(pydssp.assign(np.asarray(bb, dtype=np.float32), donor_mask=(~np.asarray(is_pro)).astype(np.float32)))
    if ss.dtype.kind in "US":   # this pydssp returns the characters '-', 'H', 'E'
        ss = np.array([CODE[c] for c in ss])
    ss = ss.astype(int)
    if complete is not None:
        ss = np.where(complete, ss, -1)
    return ss


def loop_fraction(bb, is_pro, complete=None):
    ss = assign(bb, is_pro, complete)
    ok = ss >= 0
    return float(np.mean(ss[ok] == 0))
