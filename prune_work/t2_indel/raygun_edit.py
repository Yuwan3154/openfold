"""Option 1: turn (native sequence, Raygun sequence) into a backbone edit via a global alignment.

Aligned pairs (match or mismatch) keep the native backbone; unmatched native residues are DELETED; unmatched
generated residues are INSERTED (interpolated / extrapolated by indel_edit). The edited chain's residue j is the
generated sequence's residue j, so the template sequence is Raygun's output. Alignment parameters match
raygun_probe.py (BLOSUM62, open -10, extend -0.5, global).
"""
import numpy as np
from Bio import Align
from Bio.Align import substitution_matrices


def aligner():
    al = Align.PairwiseAligner()
    al.substitution_matrix = substitution_matrices.load("BLOSUM62")
    al.open_gap_score, al.extend_gap_score, al.mode = -10.0, -0.5, "global"
    return al


def alignment_pairs(native, gen, al=None):
    aln = (al or aligner()).align(native, gen)[0]
    pairs = []
    for (n0, n1), (g0, g1) in zip(*aln.aligned):
        pairs.extend(zip(range(n0, n1), range(g0, g1)))
    return pairs


def ops_from_pairs(L, n_gen, pairs):
    """ops for indel_edit.edit(): maximal runs of unmatched native residues -> del; maximal runs of unmatched
    generated residues -> ins, placed after the native index of the last survivor before the run (-1 = N end)."""
    matched_nat = np.zeros(L, bool)
    for i, _ in pairs:
        matched_nat[i] = True
    ops = []
    i = 0
    while i < L:
        if not matched_nat[i]:
            j = i
            while j + 1 < L and not matched_nat[j + 1]:
                j += 1
            ops.append(("del", i, j))
            i = j + 1
        else:
            i += 1
    gen_to_nat = {g: n for n, g in pairs}
    g = 0
    last_nat = -1
    while g < n_gen:
        if g in gen_to_nat:
            last_nat = gen_to_nat[g]
            g += 1
        else:
            h = g
            while h + 1 < n_gen and (h + 1) not in gen_to_nat:
                h += 1
            ops.append(("ins", last_nat, h - g + 1))
            g = h + 1
    return ops
