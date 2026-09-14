"""Shared helper for the T2 verification scripts.

`SyntheticTemplatePool.sample_features` gained a REQUIRED `query_sequence` parameter, and both
`verify_prune_equivalence.py` and `verify_t2_readpath.py` were still calling the 3-argument form.
Neither noticed until 5568964 died with
    TypeError: sample_features() missing 1 required positional argument: 'query_sequence'
after an hour of real work, taking the whole round-2 merge chain down with it
(`t2_merge_r2` -> DependencyNeverSatisfied). One definition, imported by both, so the next
signature change breaks one place instead of drifting silently in two.
"""
import numpy as np

from openfold.np import residue_constants as rc


def native_frame_query_sequence(pool, chain: str) -> str:
    """The query sequence implied by a chain's own npz, for pools built WITHOUT a qmap.

    ⛔⛔ THIS MAKES sample_features' SEQUENCE-AGREEMENT ASSERTION VACUOUS, BY CONSTRUCTION.
    That assertion compares the npz `aatype` letters against `query_sequence[q]`; this function
    builds the string FROM those same letters, so it can never disagree
    ([[feedback_vacuous_check_and_symlink_parent]]). That is acceptable ONLY because neither caller
    is trying to validate the residue mapping:
      - verify_prune_equivalence asks "does the PRUNED tree return what the ORIGINAL returns",
        and passes the identical string to both pools, so the sequence is a fixed input;
      - verify_t2_readpath asks "does the reader run end-to-end on production files".
    ⭐ The NON-vacuous check on the mapping is the qmap path in real training, where the query
    sequence comes from the alignment rather than from the npz. Run C v2 exercised it for 103
    epochs with zero assertion hits.

    With no qmap, sample_features derives query positions as `residue_index - 1`, so the implied
    query length is `max(residue_index)`. Unmapped positions are filled with "X" and are never
    indexed by `q`, hence never read.
    """
    d = np.load(pool.npz_path(chain), allow_pickle=False)
    q = d["residue_index"].astype(int) - 1
    aat = d["aatype"].astype(int)
    assert q.min() >= 0, f"{chain}: residue_index below 1, cannot form a query frame"
    seq = ["X"] * (int(q.max()) + 1)
    for j, p in enumerate(q):
        seq[p] = rc.restypes[aat[j]] if aat[j] < len(rc.restypes) else "X"
    return "".join(seq)
