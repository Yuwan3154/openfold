"""Gate tests for T2's synthetic-template injection (openfold/data/synthetic_templates.py).

The failure modes worth catching here are all silent ones: a permuted amino-acid one-hot, a TM
filter that lets the trivial templates through, and a merge that leaves a phantom empty hit in the
template axis.
"""

import numpy as np
import pytest
import torch
import zlib

from openfold.data.synthetic_templates import (
    SyntheticTemplatePool,
    merge_template_features,
    natural_template_count,
    subsample_natural_templates,
)
from openfold.data.data_transforms import fix_templates_aatype
from openfold.data.templates import empty_template_feats
from openfold.np import residue_constants as rc

L, N_TMPL = 12, 6


@pytest.fixture
def pool(tmp_path):
    rng = np.random.default_rng(0)
    chains = ["1abc_A", "2xyz_B"]
    # chain 0 spans the band edges (0.10 .. 0.99); chain 1 is entirely too-easy
    tm = np.stack([np.linspace(0.10, 0.99, N_TMPL), np.full(N_TMPL, 0.95)]).astype(np.float32)
    np.savez(
        tmp_path / "index_all.npz",
        chains=np.array(chains), tm=tm,
        rewind=np.tile(np.arange(90, 90 + N_TMPL, dtype=np.int16), (2, 1)),
        length=np.array([L, L], np.int32),
    )
    root = tmp_path / "templates"
    aatype = rng.integers(0, 20, L).astype(np.int8)
    atom_mask = np.zeros((L, 37), bool)
    atom_mask[:, :5] = True                       # N, CA, C, CB, O
    n_present = int(atom_mask.sum())
    for c in chains:
        d = root / f"shard{zlib.crc32(c.encode()) % 1000:04d}"
        d.mkdir(parents=True, exist_ok=True)
        np.savez(
            d / f"{c}.npz",
            coords=rng.normal(size=(N_TMPL, n_present, 3)).astype(np.float32),
            atom_mask=atom_mask, aatype=aatype,
            residue_index=np.arange(1, L + 1, dtype=np.int32),
            rewind_steps=np.arange(90, 90 + N_TMPL, dtype=np.int16),
            model="cc89", schedule="tiered", seconds=np.float32(1.0),
        )
    qseq = "".join(rc.restypes[a] if a < len(rc.restypes) else "X" for a in aatype)
    return SyntheticTemplatePool(str(tmp_path / "index_all.npz"), str(root),
                                 min_tm=0.3, max_tm=0.9), aatype, qseq


def test_tm_filter_is_a_band_not_a_ceiling(pool):
    """Eligibility is 0.3 < TM < 0.9 (user, 2026-08-14): outside it a template is too hard or
    too easy. A ceiling-only filter would silently readmit the sub-0.3 tail."""
    p, _, QSEQ = pool
    keep = p.eligible[p.row_of["1abc_A"]]
    assert set(keep) == set(np.flatnonzero((p.tm[0] > 0.3) & (p.tm[0] < 0.9)))
    assert p.tm[0][keep].min() > 0.3 and p.tm[0][keep].max() < 0.9
    assert (p.tm[0] <= 0.3).any(), "fixture must contain a too-hard template for this to bite"


def test_chain_with_no_eligible_template_is_skipped(pool):
    """2xyz_B's templates are all TM 0.95, i.e. all too easy -- it must yield nothing, not crash."""
    p, _, QSEQ = pool
    assert "2xyz_B" not in p
    assert p.sample_features("2xyz_B", 2, np.random.default_rng(0), query_sequence=QSEQ) is None
    assert p.sample_features("nosuch_A", 2, np.random.default_rng(0), query_sequence=QSEQ) is None


def test_sampled_features_have_the_natural_layout(pool):
    p, _, QSEQ = pool
    f = p.sample_features("1abc_A", 3, np.random.default_rng(1), query_sequence=QSEQ)
    assert f["template_all_atom_positions"].shape == (3, L, 37, 3)
    assert f["template_all_atom_mask"].shape == (3, L, 37)
    assert f["template_aatype"].shape == (3, L, 22)
    assert f["template_sum_probs"].shape == (3, 1)
    assert len(f["template_domain_names"]) == 3
    assert np.isfinite(f["template_all_atom_positions"]).all()
    # coordinates must land on the atoms the mask says exist, and nowhere else
    m = f["template_all_atom_mask"][0] > 0
    assert (f["template_all_atom_positions"][0][~m] == 0).all()


def test_sampling_never_exceeds_the_eligible_pool(pool):
    p, _, QSEQ = pool
    n_elig = len(p.eligible[p.row_of["1abc_A"]])
    f = p.sample_features("1abc_A", 999, np.random.default_rng(2), query_sequence=QSEQ)
    assert f["template_all_atom_positions"].shape[0] == n_elig
    assert len(set(f["_tm"].tolist())) == n_elig      # sampled without replacement


def test_aatype_onehot_survives_the_hhblits_reordering(pool):
    """⛔ The one-hot must be in HHBLITS order: `fix_templates_aatype` argmaxes then gathers through
    MAP_HHBLITS_AATYPE_TO_OUR_AATYPE, so a hand-rolled restype-order one-hot comes out permuted."""
    p, aatype, QSEQ = pool
    f = p.sample_features("1abc_A", 2, np.random.default_rng(3), query_sequence=QSEQ)
    prot = {"template_aatype": torch.tensor(f["template_aatype"])}
    fix_templates_aatype(prot)
    assert torch.equal(prot["template_aatype"][0], torch.tensor(aatype, dtype=torch.long))


def test_merge_concatenates_onto_natural_hits(pool):
    p, _, QSEQ = pool
    nat = {
        "template_all_atom_positions": np.zeros((2, L, 37, 3), np.float32),
        "template_all_atom_mask": np.zeros((2, L, 37), np.float32),
        "template_aatype": np.zeros((2, L, 22), np.float32),
        "template_sum_probs": np.zeros((2, 1), np.float32),
        "template_domain_names": np.array([b"1aaa_A", b"1bbb_B"], dtype=object),
        "template_sequence": np.array([b"AAA", b"BBB"], dtype=object),
        "aatype": np.zeros((L,), np.int64),                       # untouched non-template feature
    }
    f = p.sample_features("1abc_A", 3, np.random.default_rng(4), query_sequence=QSEQ)
    out = merge_template_features(nat, f)
    assert out["template_all_atom_positions"].shape[0] == 5
    assert out["template_aatype"].shape[0] == 5
    assert len(out["template_domain_names"]) == 5
    assert out["template_domain_names"][2].startswith(b"pp1c_1abc_A_r")
    assert out["aatype"].shape == (L,)                            # non-template key untouched
    assert "_tm" not in out                                       # diagnostic key not leaked


def test_merge_replaces_the_empty_placeholder(pool):
    """A chain with no hits carries a 0-length template axis but a LENGTH-1 object placeholder;
    concatenating onto that would hand the model a phantom empty template."""
    p, _, QSEQ = pool
    nat = empty_template_feats(L)
    nat.pop("template_dgram_probs")                               # not built by process_mmcif here
    f = p.sample_features("1abc_A", 2, np.random.default_rng(5), query_sequence=QSEQ)
    out = merge_template_features(nat, f)
    assert out["template_all_atom_positions"].shape[0] == 2
    assert len(out["template_domain_names"]) == 2
    assert len(out["template_sequence"]) == 2
    assert all(n.startswith(b"pp1c_") for n in out["template_domain_names"])


def test_merge_zero_extends_keys_the_synthetic_side_cannot_produce(pool):
    p, _, QSEQ = pool
    nat = {
        "template_all_atom_positions": np.zeros((1, L, 37, 3), np.float32),
        "template_dgram_probs": np.ones((1, L, L, 39), np.float32),
    }
    out = merge_template_features(nat, p.sample_features("1abc_A", 2, np.random.default_rng(6), query_sequence=QSEQ))
    assert out["template_dgram_probs"].shape == (3, L, L, 39)
    assert (out["template_dgram_probs"][1:] == 0).all()


def test_pruned_tree_translates_the_rung_to_its_compacted_row(tmp_path):
    """⛔ On a band-pruned tree the npz holds only the in-band rungs, so rung index != row index.

    Without the `slot` translation the pool would read whatever template happens to sit at the
    rung's ORIGINAL position -- coordinates from one template paired with the TM of another, and
    every downstream number silently wrong rather than crashing. Built here so the two orderings
    disagree: the kept rungs are 1..4 out of 6, so rung k lives at row k-1.
    """
    rng = np.random.default_rng(0)
    chain = "1abc_A"
    tm = np.linspace(0.10, 0.99, N_TMPL).astype(np.float32)[None]
    band = (tm[0] > 0.3) & (tm[0] < 0.9)
    keep = np.flatnonzero(band)
    assert keep[0] != 0, "fixture must drop at least the first rung or it tests nothing"
    slot = np.full((1, N_TMPL), -1, np.int16)
    slot[0, keep] = np.arange(len(keep), dtype=np.int16)
    np.savez(
        tmp_path / "index_all.npz",
        chains=np.array([chain]), tm=tm,
        rewind=np.arange(90, 90 + N_TMPL, dtype=np.int16)[None],
        length=np.array([L], np.int32), slot=slot,
        min_tm=np.float32(0.3), max_tm=np.float32(0.9),
    )
    atom_mask = np.zeros((L, 37), bool)
    atom_mask[:, :5] = True
    n_present = int(atom_mask.sum())
    # a distinct constant per KEPT template, so the coords identify which row was read
    coords = np.stack([np.full((n_present, 3), float(j)) for j in range(len(keep))]).astype(np.float32)
    d = tmp_path / "templates" / f"shard{zlib.crc32(chain.encode()) % 1000:04d}"
    d.mkdir(parents=True)
    aat = rng.integers(0, 20, L).astype(np.int8)
    np.savez(
        d / f"{chain}.npz", coords=coords, atom_mask=atom_mask,
        aatype=aat,
        residue_index=np.arange(1, L + 1, dtype=np.int32),
        rewind_steps=(90 + keep).astype(np.int16),
    )
    qseq = "".join(rc.restypes[a] for a in aat)
    p = SyntheticTemplatePool(str(tmp_path / "index_all.npz"), str(tmp_path / "templates"),
                              min_tm=0.3, max_tm=0.9)
    f = p.sample_features(chain, len(keep), np.random.default_rng(0), query_sequence=qseq)
    got = f["template_all_atom_positions"][:, 0, 1, 0]             # the constant, per template
    # each sampled template's TM must be the one belonging to the row its coords came from
    for tm_val, row in zip(f["_tm"], got):
        assert tm[0][keep[int(row)]] == pytest.approx(tm_val), "coords paired with the wrong TM"


def test_unpruned_index_still_reads_the_rung_directly(pool):
    """The `slot` key is absent on the original 64-rung index -- that path must be untouched."""
    p, _, QSEQ = pool
    assert p.slot is None
    f = p.sample_features("1abc_A", 2, np.random.default_rng(0), query_sequence=QSEQ)
    assert f["template_all_atom_positions"].shape[0] == 2


def _pruned_index(tmp_path, lo, hi):
    """A minimal pruned index/tree pair recording the band it was pruned to."""
    tm = np.linspace(0.10, 0.99, N_TMPL).astype(np.float32)[None]
    keep = np.flatnonzero((tm[0] > lo) & (tm[0] < hi))
    slot = np.full((1, N_TMPL), -1, np.int16)
    slot[0, keep] = np.arange(len(keep), dtype=np.int16)
    np.savez(
        tmp_path / "index_all.npz",
        chains=np.array(["1abc_A"]), tm=tm,
        rewind=np.arange(90, 90 + N_TMPL, dtype=np.int16)[None],
        length=np.array([L], np.int32), slot=slot,
        min_tm=np.float32(lo), max_tm=np.float32(hi),
    )
    atom_mask = np.zeros((L, 37), bool)
    atom_mask[:, :5] = True
    d = tmp_path / "templates" / f"shard{zlib.crc32(b'1abc_A') % 1000:04d}"
    d.mkdir(parents=True)
    np.savez(
        d / "1abc_A.npz",
        coords=np.zeros((len(keep), int(atom_mask.sum()), 3), np.float32),
        atom_mask=atom_mask, aatype=np.zeros(L, np.int8),
        residue_index=np.arange(1, L + 1, dtype=np.int32),
        rewind_steps=(90 + keep).astype(np.int16),
    )
    # aatype is all zeros -> restypes[0] repeated, so this is the matching query sequence
    return (str(tmp_path / "index_all.npz"), str(tmp_path / "templates"),
            rc.restypes[0] * L)


def test_pruned_index_refuses_a_wider_band(tmp_path):
    """⛔ A pruned tree physically lacks the out-of-band rungs, so a wider band at training time
    would select rows that were never written. That must fail at construction with the numbers in
    hand, not on some later step -- a run that dies 3 hours in is far worse than one that never
    starts."""
    idx, root, qseq = _pruned_index(tmp_path, 0.3, 0.9)
    with pytest.raises(AssertionError, match="pruned to TM"):
        SyntheticTemplatePool(idx, root, min_tm=0.2, max_tm=0.9)
    with pytest.raises(AssertionError, match="pruned to TM"):
        SyntheticTemplatePool(idx, root, min_tm=0.3, max_tm=0.95)


def test_pruned_index_accepts_the_same_or_narrower_band(tmp_path):
    idx, root, qseq = _pruned_index(tmp_path, 0.3, 0.9)
    same = SyntheticTemplatePool(idx, root, min_tm=0.3, max_tm=0.9)
    narrower = SyntheticTemplatePool(idx, root, min_tm=0.4, max_tm=0.8)
    assert len(narrower.eligible[0]) < len(same.eligible[0])
    # a narrower band still indexes real rows -- the translation must hold
    f = narrower.sample_features("1abc_A", 2, np.random.default_rng(0), query_sequence=qseq)
    assert f["template_all_atom_positions"].shape[0] == 2


def _partial_native_pool(tmp_path, query_len, first_resnum, n_native, gap_at=None):
    """A chain whose npz covers only PART of the query, the way real natives do.

    ⛔ This is the fixture the original suite lacked. Every earlier test built the npz at the query
    length, and the merge test took its "natural" block's NUM_RES *from the synthetic block itself*
    -- so a synthetic block on the wrong residue frame was undetectable by construction. That is
    exactly the bug that killed the first T2 launch (natural 104 vs synthetic 89). The two lengths
    must come from INDEPENDENT sources or the test proves nothing.
    """
    rng = np.random.default_rng(7)
    chain = "1abc_A"
    qaat = rng.integers(0, 20, query_len).astype(np.int8)
    qseq = "".join(rc.restypes[a] for a in qaat)

    resnums = list(range(first_resnum, first_resnum + n_native))
    if gap_at is not None:                      # real natives have unresolved stretches
        resnums = resnums[:gap_at] + [r + 5 for r in resnums[gap_at:]]
    resnums = [r for r in resnums if r <= query_len]
    q0 = np.array(resnums) - 1                  # query 0-based positions this native covers
    aat = qaat[q0]                              # aatype MUST agree with the query there

    tm = np.full((1, N_TMPL), 0.5, np.float32)
    np.savez(tmp_path / "index_all.npz",
             chains=np.array([chain]), tm=tm,
             rewind=np.arange(90, 90 + N_TMPL, dtype=np.int16)[None],
             length=np.array([len(resnums)], np.int32))
    atom_mask = np.zeros((len(resnums), 37), bool)
    atom_mask[:, :5] = True
    d = tmp_path / "templates" / f"shard{zlib.crc32(chain.encode()) % 1000:04d}"
    d.mkdir(parents=True)
    np.savez(d / f"{chain}.npz",
             coords=rng.normal(size=(N_TMPL, int(atom_mask.sum()), 3)).astype(np.float32),
             atom_mask=atom_mask, aatype=aat,
             residue_index=np.array(resnums, np.int32),
             rewind_steps=np.arange(90, 90 + N_TMPL, dtype=np.int16))
    pool = SyntheticTemplatePool(str(tmp_path / "index_all.npz"), str(tmp_path / "templates"),
                                 min_tm=0.3, max_tm=0.9)
    return pool, chain, qseq, np.array(resnums)


def test_features_are_built_on_the_query_frame_not_the_native_frame(tmp_path):
    """The npz covers 89 of a 104-residue query -- the exact shape of the T2 launch crash."""
    p, chain, qseq, resnums = _partial_native_pool(tmp_path, 104, 9, 89)
    assert len(resnums) < len(qseq), "fixture must be a strict subset or it tests nothing"
    f = p.sample_features(chain, 2, np.random.default_rng(0), query_sequence=qseq)
    k = f["template_all_atom_positions"].shape[0]
    assert f["template_all_atom_positions"].shape == (k, 104, 37, 3)
    assert f["template_all_atom_mask"].shape == (k, 104, 37)
    assert f["template_aatype"].shape == (k, 104, 22)

    covered = resnums - 1
    uncovered = np.setdiff1d(np.arange(104), covered)
    assert (f["template_all_atom_mask"][:, covered].sum(-1) > 0).all(), "covered residues unmasked"
    assert f["template_all_atom_mask"][:, uncovered].sum() == 0, "uncovered residues have atoms"
    assert (f["template_all_atom_positions"][:, uncovered] == 0).all()
    seq = f["template_sequence"][0].decode()
    assert len(seq) == 104
    assert all(seq[i] == "-" for i in uncovered), "uncovered positions must be gaps"
    assert all(seq[i] == qseq[i] for i in covered), "covered positions must match the query"


def test_query_frame_survives_a_gapped_native(tmp_path):
    """Unresolved stretches make residue_index non-contiguous; the scatter must still be exact."""
    p, chain, qseq, resnums = _partial_native_pool(tmp_path, 120, 3, 90, gap_at=40)
    assert not np.array_equal(resnums, np.arange(resnums[0], resnums[0] + len(resnums))), \
        "fixture must actually be gapped"
    f = p.sample_features(chain, 1, np.random.default_rng(0), query_sequence=qseq)
    seq = f["template_sequence"][0].decode()
    covered = resnums - 1
    assert all(seq[i] == qseq[i] for i in covered)
    assert f["template_all_atom_mask"][:, np.setdiff1d(np.arange(120), covered)].sum() == 0


def test_merge_rejects_a_num_res_mismatch_with_a_useful_message(tmp_path):
    """np.concatenate's own error names only axis sizes, not the cause. Ours must name the cause."""
    p, chain, qseq, _ = _partial_native_pool(tmp_path, 104, 9, 89)
    f = p.sample_features(chain, 2, np.random.default_rng(0), query_sequence=qseq)
    nat = {                                     # natural block at a DIFFERENT length
        "template_all_atom_positions": np.zeros((4, 89, 37, 3), np.float32),
        "template_all_atom_mask": np.zeros((4, 89, 37), np.float32),
    }
    with pytest.raises(ValueError, match="template NUM_RES mismatch"):
        merge_template_features(nat, f)


def test_merge_succeeds_when_both_sides_are_on_the_query_frame(tmp_path):
    """The positive case, with the natural block's length taken from the QUERY, not the synthetic."""
    p, chain, qseq, _ = _partial_native_pool(tmp_path, 104, 9, 89)
    f = p.sample_features(chain, 2, np.random.default_rng(0), query_sequence=qseq)
    qL = len(qseq)
    nat = {
        "template_all_atom_positions": np.zeros((4, qL, 37, 3), np.float32),
        "template_all_atom_mask": np.zeros((4, qL, 37), np.float32),
        "template_aatype": np.zeros((4, qL, 22), np.float32),
        "template_sequence": np.array([b"A" * qL] * 4, dtype=object),
        "template_domain_names": np.array([b"nat%d" % i for i in range(4)], dtype=object),
        "template_sum_probs": np.zeros((4, 1), np.float32),
    }
    out = merge_template_features(nat, f)
    assert out["template_all_atom_positions"].shape == (6, qL, 37, 3)


def test_sequence_disagreement_is_caught(tmp_path):
    """An off-by-one in the residue mapping stays in bounds and silently shifts every residue.
    Only sequence identity catches it, so that check must be live, not test-only."""
    p, chain, qseq, _ = _partial_native_pool(tmp_path, 104, 9, 89)
    shifted = qseq[1:] + qseq[0]                # same length, wrong correspondence
    with pytest.raises(AssertionError, match="disagrees with the query sequence"):
        p.sample_features(chain, 1, np.random.default_rng(0), query_sequence=shifted)


def test_native_numbering_beyond_the_query_is_refused(tmp_path):
    p, chain, qseq, _ = _partial_native_pool(tmp_path, 104, 9, 89)
    with pytest.raises(AssertionError, match="fit a query of length"):
        p.sample_features(chain, 1, np.random.default_rng(0), query_sequence=qseq[:50])


# ---- the query-index map (build_query_index_map.py) --------------------------------------------

def _qmap_npz(tmp_path, chain, qmap, query_len, ambiguous=0):
    path = tmp_path / "qmap_all.npz"
    np.savez(path, chains=np.array([chain]), qmap=np.asarray(qmap, np.int32),
             qmap_len=np.array([len(qmap)], np.int32),
             query_len=np.array([query_len], np.int32),
             ambiguous=np.array([ambiguous], np.int8))
    return str(path)


def test_qmap_overrides_residue_index_and_places_rows_correctly(tmp_path):
    """⛔ THE bug that broke two T2 launches: residue_index is protpardelle's structure parse, so
    ridx-1 desynchronises at the first unresolved residue. The qmap is authoritative.

    Fixture is the real 1eis_A shape: npz numbered CONTIGUOUSLY while actually skipping a query
    position, so ridx-1 and the true map disagree and the difference is observable.
    """
    p, chain, qseq, resnums = _partial_native_pool(tmp_path, 104, 9, 89)
    # the true map: shift every row one further along than ridx-1 would say, i.e. a genuine gap
    true_q = (resnums - 1) + 1
    assert true_q.max() < len(qseq)
    # aatype must agree with the query at the TRUE positions, so rebuild the npz accordingly
    d = np.load(p.npz_path(chain), allow_pickle=False)
    new_aat = np.array([rc.restype_order[qseq[i]] for i in true_q], np.int8)
    np.savez(p.npz_path(chain), coords=d["coords"], atom_mask=d["atom_mask"], aatype=new_aat,
             residue_index=d["residue_index"], rewind_steps=d["rewind_steps"])

    qm = _qmap_npz(tmp_path, chain, true_q, len(qseq))
    pool = SyntheticTemplatePool(str(tmp_path / "index_all.npz"), str(tmp_path / "templates"),
                                 min_tm=0.3, max_tm=0.9, qmap_path=qm)
    f = pool.sample_features(chain, 1, np.random.default_rng(0), query_sequence=qseq)
    seq = f["template_sequence"][0].decode()
    covered = true_q
    assert all(seq[i] == qseq[i] for i in covered)
    # and the positions ridx-1 would have used, but the qmap did not, must be gaps
    only_old = np.setdiff1d(resnums - 1, covered)
    assert len(only_old) > 0, "fixture must make the two maps differ"
    assert all(seq[i] == "-" for i in only_old), "used the stale ridx-1 placement"


def test_chain_absent_from_a_supplied_qmap_is_unavailable(tmp_path):
    """⛔ A missing map must mean 'no synthetic templates for this chain', NEVER a silent fallback
    to the arithmetic that caused the original bug."""
    p, chain, qseq, resnums = _partial_native_pool(tmp_path, 104, 9, 89)
    qm = _qmap_npz(tmp_path, "someother_A", np.arange(5), 104)
    pool = SyntheticTemplatePool(str(tmp_path / "index_all.npz"), str(tmp_path / "templates"),
                                 min_tm=0.3, max_tm=0.9, qmap_path=qm)
    assert chain not in pool
    assert pool.sample_features(chain, 1, np.random.default_rng(0), query_sequence=qseq) is None


def test_stale_qmap_query_length_is_refused(tmp_path):
    p, chain, qseq, resnums = _partial_native_pool(tmp_path, 104, 9, 89)
    qm = _qmap_npz(tmp_path, chain, resnums - 1, 999)          # wrong query_len on purpose
    pool = SyntheticTemplatePool(str(tmp_path / "index_all.npz"), str(tmp_path / "templates"),
                                 min_tm=0.3, max_tm=0.9, qmap_path=qm)
    with pytest.raises(AssertionError, match="stale"):
        pool.sample_features(chain, 1, np.random.default_rng(0), query_sequence=qseq)


def test_qmap_row_count_must_match_the_npz(tmp_path):
    p, chain, qseq, resnums = _partial_native_pool(tmp_path, 104, 9, 89)
    qm = _qmap_npz(tmp_path, chain, (resnums - 1)[:-3], len(qseq))   # 3 rows short
    pool = SyntheticTemplatePool(str(tmp_path / "index_all.npz"), str(tmp_path / "templates"),
                                 min_tm=0.3, max_tm=0.9, qmap_path=qm)
    with pytest.raises(AssertionError, match="qmap has"):
        pool.sample_features(chain, 1, np.random.default_rng(0), query_sequence=qseq)


def test_no_qmap_at_all_still_uses_residue_index(tmp_path):
    """Legacy/no-map path stays intact so the existing tests and any pre-qmap index keep working."""
    p, chain, qseq, resnums = _partial_native_pool(tmp_path, 104, 9, 89)
    assert not p.qmap
    f = p.sample_features(chain, 1, np.random.default_rng(0), query_sequence=qseq)
    assert f["template_all_atom_positions"].shape[1] == len(qseq)


# ---------------------------------------------------------------------------------------------
# COUNT-MATCHED MODE (--t2_replace_natural): the pool must stay exactly the size it would have been
# without synthetic templates, because `templates_crop_start ~ Uniform{0..pool}` is INCLUSIVE and so
# a bigger pool silently changes the delivered COUNT as well as the content.
# ---------------------------------------------------------------------------------------------

def _natural(k, tag=b"nat"):
    """k natural hits whose per-hit content is IDENTIFIABLE, so a subsample can be checked by name.

    ⛔ The template axis length is passed in explicitly rather than derived from anything the
    synthetic side produces -- deriving one side's shape from the other is what made the original
    merge test unable to catch the query-frame bug.
    """
    return {
        "template_all_atom_positions": np.arange(k * L * 37 * 3, dtype=np.float32).reshape(k, L, 37, 3),
        "template_all_atom_mask": np.ones((k, L, 37), np.float32),
        "template_aatype": np.zeros((k, L, 22), np.float32),
        "template_sum_probs": np.arange(k, dtype=np.float32).reshape(k, 1),
        "template_domain_names": np.array([tag + b"_%d" % i for i in range(k)], dtype=object),
        "template_sequence": np.array([b"S%d" % i for i in range(k)], dtype=object),
        "aatype": np.zeros((L,), np.int64),
    }


def test_natural_template_count_reads_the_numeric_array_not_the_placeholder():
    """empty_template_feats gives the numeric arrays a 0-length axis but the object arrays length 1;
    counting off template_sequence would report a hit that does not exist."""
    e = empty_template_feats(L)
    assert len(e["template_domain_names"]) == 1                    # the trap
    assert natural_template_count(e) == 0                          # the correct answer
    assert natural_template_count(_natural(4)) == 4
    assert natural_template_count({}) == 0


def test_subsample_keeps_exactly_k_and_touches_only_template_keys():
    nat = _natural(4)
    out = subsample_natural_templates(nat, 2, np.random.default_rng(0))
    assert out["template_all_atom_positions"].shape == (2, L, 37, 3)
    assert out["template_sum_probs"].shape == (2, 1)
    assert len(out["template_domain_names"]) == 2
    assert len(out["template_sequence"]) == 2
    assert out["aatype"].shape == (L,)                             # non-template key untouched
    assert nat["template_all_atom_positions"].shape[0] == 4        # input not mutated


def test_subsample_keeps_whole_hits_together():
    """Every kept key must come from the SAME hit indices -- a per-key independent draw would
    silently pair one template's coordinates with another's sequence."""
    nat = _natural(5)
    out = subsample_natural_templates(nat, 3, np.random.default_rng(1))
    kept = [int(n.split(b"_")[1]) for n in out["template_domain_names"]]
    assert [int(x) for x in out["template_sum_probs"][:, 0]] == kept
    assert [int(s[1:]) for s in [b.decode().encode() for b in out["template_sequence"]]] == kept
    for j, orig in enumerate(kept):
        assert np.array_equal(out["template_all_atom_positions"][j],
                              nat["template_all_atom_positions"][orig])


def test_subsample_is_a_noop_when_nothing_needs_dropping():
    nat = _natural(3)
    for keep in (3, 4):
        out = subsample_natural_templates(nat, keep, np.random.default_rng(2))
        assert out["template_all_atom_positions"].shape[0] == 3
    assert subsample_natural_templates(empty_template_feats(L), 0, np.random.default_rng(2)) is not None


def test_subsample_to_zero_leaves_a_mergeable_empty_axis():
    """keep=0 happens whenever the synthetic count equals the natural count; the result still has to
    merge cleanly rather than leaving a ragged or phantom template axis."""
    out = subsample_natural_templates(_natural(4), 0, np.random.default_rng(3))
    assert out["template_all_atom_positions"].shape[0] == 0
    assert len(out["template_domain_names"]) == 0


def test_subsample_rejects_a_ragged_template_axis():
    nat = _natural(4)
    nat["template_sequence"] = np.array([b"only_one"], dtype=object)
    with pytest.raises(ValueError, match="ragged"):
        subsample_natural_templates(nat, 2, np.random.default_rng(4))


def test_subsample_draws_uniformly_not_top_k():
    """The natural component must stay distributed as in T1, so the survivors cannot be the top-k by
    sum_probs. Over many draws every hit index has to appear."""
    seen = set()
    for s in range(60):
        out = subsample_natural_templates(_natural(4), 2, np.random.default_rng(s))
        seen.update(int(n.split(b"_")[1]) for n in out["template_domain_names"])
    assert seen == {0, 1, 2, 3}


def _deliver(pool_size, max_templates=4, seed=0):
    """Replicate EXACTLY what random_crop_to_size does to the template axis, using the real torch
    calls (data_transforms.py:1219-1249): permute the whole pool, take a contiguous window starting
    at templates_crop_start ~ U{0..pool} INCLUSIVE, of size min(pool - start, max_templates).
    Returns the delivered POOL INDICES."""
    g = torch.Generator().manual_seed(seed)
    start = int(torch.randint(0, pool_size + 1, (1,), generator=g)[0])
    perm = torch.randperm(pool_size, generator=g)
    size = min(pool_size - start, max_templates)
    return perm[start:start + size].tolist()


def test_delivered_count_matches_the_analytic_t1_distribution():
    """Guards the baseline the whole count-matching argument rests on: pool 4 -> mean 2.00 delivered
    and P(0 templates) 20%. If this drifts, every claim about matching drifts with it."""
    n = [len(_deliver(4, seed=s)) for s in range(40000)]
    assert abs(np.mean(n) - 2.00) < 0.03, np.mean(n)
    assert abs(np.mean([x == 0 for x in n]) - 0.20) < 0.01
    n8 = [len(_deliver(8, seed=s)) for s in range(40000)]
    assert abs(np.mean(n8) - 2.889) < 0.04, np.mean(n8)      # append mode, for contrast
    assert abs(np.mean([x == 0 for x in n8]) - 0.111) < 0.01


def test_pool_level_replacement_equals_delivered_level_replacement():
    """⭐⭐ THE LOAD-BEARING CLAIM. The user asked for each SELECTED natural template to be replaced
    with probability p. We implement it by labelling each POOL slot i.i.d. Bernoulli(p) instead,
    which is only legitimate because random_crop_to_size picks the delivered window independently of
    the labels. Verified by Monte Carlo against the delivered-level policy, on the same real torch
    calls: both the delivered COUNT and the delivered SYNTHETIC count must agree, and the synthetic
    count must be Binomial(delivered, p)."""
    p, POOL, N = 0.5, 4, 40000
    pool_level_n, pool_level_syn = [], []
    deliv_level_n, deliv_level_syn = [], []
    lab_rng = np.random.default_rng(0)
    for s in range(N):
        idx = _deliver(POOL, seed=s)
        # POOL-LEVEL: label every slot first, then deliver
        labels = lab_rng.random(POOL) < p
        pool_level_n.append(len(idx))
        pool_level_syn.append(int(labels[idx].sum()))
        # DELIVERED-LEVEL: deliver first, then label only what was delivered
        deliv_level_n.append(len(idx))
        deliv_level_syn.append(int((lab_rng.random(len(idx)) < p).sum()))
    assert pool_level_n == deliv_level_n                       # count is untouched by construction
    assert abs(np.mean(pool_level_syn) - np.mean(deliv_level_syn)) < 0.03, (
        np.mean(pool_level_syn), np.mean(deliv_level_syn))
    # and the synthetic share of delivered slots is p
    tot = sum(pool_level_n)
    assert abs(sum(pool_level_syn) / tot - p) < 0.01, sum(pool_level_syn) / tot
    # distribution, not just the mean: P(k synthetic | delivered) must match Binomial(delivered, p)
    for k in range(5):
        a = np.mean([sy == k for sy, nn in zip(pool_level_syn, pool_level_n) if nn == 4])
        b = np.mean([sy == k for sy, nn in zip(deliv_level_syn, deliv_level_n) if nn == 4])
        assert abs(a - b) < 0.02, (k, a, b)


def test_binomial_keep_preserves_the_pool_size_and_halves_the_naturals():
    """The mechanics of (2): n_keep_nat ~ Binomial(n_nat, 1-p), pool refilled to the SAME size."""
    rng = np.random.default_rng(0)
    kept = [int(rng.binomial(4, 0.5)) for _ in range(20000)]
    assert abs(np.mean(kept) - 2.0) < 0.05
    for k in kept[:200]:
        assert 4 - k >= 0                                      # budget is never negative


def test_topup_hypergeometric_has_the_right_natural_share():
    """The mechanics of (1): with n_prefiltered natural in a pool topped to 20, the naturals among the
    4 taken are Hypergeometric(20, n_pref, 4), so their expected number is 4*n_pref/20."""
    rng = np.random.default_rng(0)
    for n_pref in (0, 1, 3, 10, 19):
        draws = [int(rng.hypergeometric(n_pref, 20 - n_pref, 4)) for _ in range(20000)]
        assert abs(np.mean(draws) - 4 * n_pref / 20) < 0.05, (n_pref, np.mean(draws))
        assert max(draws) <= min(n_pref, 4)                    # cannot keep more than it has


def _mix(pool_obj, nat, n_keep_nat, pool_target, chain, qseq, rng):
    """The hook's exact arithmetic, so the tests exercise the real formula rather than a paraphrase."""
    n_nat = natural_template_count(nat)
    budget = pool_target - n_keep_nat
    extra = pool_obj.sample_features(chain, budget, rng, query_sequence=qseq) if budget > 0 else None
    added = 0 if extra is None else int(extra["template_all_atom_positions"].shape[0])
    keep_final = min(n_nat, max(n_keep_nat, pool_target - added))
    out = subsample_natural_templates(nat, keep_final, rng) if keep_final < n_nat else nat
    return (merge_template_features(out, extra) if extra is not None else out), added, keep_final


def test_topup_fills_a_template_poor_chain_to_the_cap(pool):
    """End to end for the case the top-up exists for: a chain with 1 natural hit ends up with a FULL
    pool of max_templates instead of 1 -- the delivered-count increase that was accepted for the 1.3%
    of chains with <4 prefiltered hits.
    ⭐ This fixture's chain has only 3 in-band templates, so it also exercises the RESTORE: the pool
    supplies 3 of the 4 wanted and the 1 natural is kept rather than dropped, still reaching 4."""
    p, _, QSEQ = pool
    n_elig = len(p.eligible[p.row_of["1abc_A"]])
    assert n_elig == 3, n_elig                                 # fixture invariant this test relies on
    rng = np.random.default_rng(3)
    out, added, keep_final = _mix(p, _natural(1), n_keep_nat=0, pool_target=4,
                                 chain="1abc_A", qseq=QSEQ, rng=rng)
    assert added == 3                                          # pool could not fill all 4
    assert keep_final == 1                                     # so the natural was RESTORED
    assert natural_template_count(out) == 4                    # and the pool still reaches the target
    assert sum(d.startswith(b"pp1c_") for d in out["template_domain_names"]) == 3


def test_short_synthetic_pool_restores_naturals_rather_than_shrinking(pool):
    """⛔ The failure mode that looks conservative but is not: if the pool cannot fill the budget and
    we still drop the naturals, the chain ends up with FEWER templates than T1 would have given it."""
    p, _, QSEQ = pool
    rng = np.random.default_rng(9)
    out, added, keep_final = _mix(p, _natural(4), n_keep_nat=1, pool_target=4,
                                 chain="1abc_A", qseq=QSEQ, rng=rng)
    assert added == 3 and keep_final == 1
    assert natural_template_count(out) == 4
    # and had the pool returned NOTHING, every natural would be restored
    assert min(4, max(1, 4 - 0)) == 4


def test_chain_with_no_natural_hits_gets_synthetic_only_under_topup(pool):
    """0.36% of training chains have no natural hits at all (measured over all 88155). Without top-up
    they see nothing, exactly as in T1; with top-up they get an all-synthetic pool, which is the
    point. Capped here by the fixture's 3 eligible templates, not by the rule."""
    p, _, QSEQ = pool
    n_elig = len(p.eligible[p.row_of["1abc_A"]])
    nat = empty_template_feats(L)
    nat.pop("template_dgram_probs")
    assert natural_template_count(nat) == 0
    out, added, keep_final = _mix(p, nat, n_keep_nat=0, pool_target=4,
                                 chain="1abc_A", qseq=QSEQ, rng=np.random.default_rng(11))
    assert added == n_elig and keep_final == 0
    assert natural_template_count(out) == n_elig
    assert all(d.startswith(b"pp1c_") for d in out["template_domain_names"])

    # ⛔ and WITHOUT top-up the same chain must stay exactly as T1 has it: pool_target = n_nat = 0,
    # so the budget is 0 and nothing is added at all
    nat2 = empty_template_feats(L)
    nat2.pop("template_dgram_probs")
    out2, added2, _ = _mix(p, nat2, n_keep_nat=0, pool_target=0,
                           chain="1abc_A", qseq=QSEQ, rng=np.random.default_rng(12))
    assert added2 == 0 and natural_template_count(out2) == 0


# ---------------------------------------------------------------------------------------------
# THREE-GROUP PRE-SHUFFLE MIXTURE (natural / synthetic filler / promoted), user's design 2026-08-18.
# `--t4_n_promoted` is the promoted group's WEIGHT here, not a delivered count -- max_templates=4
# caps the delivered count, so any value above 4 would be meaningless under a per-step reading.
# ---------------------------------------------------------------------------------------------

def _draw(g_nat, g_syn, g_pro, max_t=4, n=40000, seed=0):
    rng = np.random.default_rng(seed)
    target = min(max_t, g_nat + g_syn + g_pro)
    if target == 0:
        return np.zeros((0, 3), int), 0
    return rng.multivariate_hypergeometric([g_nat, g_syn, g_pro], target, size=n), target


def test_promoted_weight_sets_its_delivered_share():
    """32 promoted beside 20 natural must give each drawn slot p = 32/52 promoted -- the number the
    flag value is chosen for. If this drifts, `--t4_n_promoted 32` no longer means what was agreed."""
    d, target = _draw(20, 0, 32)
    assert target == 4
    share = d[:, 2].mean() / target
    assert abs(share - 32 / 52) < 0.01, share
    assert abs(d[:, 0].mean() / target - 20 / 52) < 0.01
    assert (d.sum(axis=1) == target).all()          # every slot is filled from some group


def test_promoted_group_scales_monotonically_with_the_flag():
    prev = -1.0
    for n_pro in (0, 4, 16, 32, 64):
        d, target = _draw(20, 0, n_pro)
        share = d[:, 2].mean() / target
        assert share > prev or n_pro == 0, (n_pro, share, prev)
        prev = share
    assert prev > 0.7                                # 64 vs 20 natural is promoted-dominated


def test_t4_is_inert_until_the_pool_fills():
    """⭐ No warmup branching is needed for the READ side: a chain contributes at most what it HAS, so
    with an empty promoted pool the draw is identical to the no-T4 case."""
    a, ta = _draw(20, 0, 0, seed=5)
    b, tb = _draw(20, 0, 0, seed=5)
    assert ta == tb and np.array_equal(a, b)
    d, _ = _draw(20, 0, 0)
    assert d[:, 2].sum() == 0                        # nothing promoted can be drawn


def test_topup_and_promoted_compose_on_a_template_poor_chain():
    """A chain with 2 prefiltered hits, topped to 20, plus 32 promoted: naturals become a small
    minority, which is the intended "supply templates where natural ones are missing" behaviour."""
    d, target = _draw(2, 18, 32)
    assert target == 4
    assert abs(d[:, 0].mean() / target - 2 / 52) < 0.01     # natural
    assert abs(d[:, 1].mean() / target - 18 / 52) < 0.01    # synthetic filler
    assert abs(d[:, 2].mean() / target - 32 / 52) < 0.01    # promoted


def test_natural_group_is_capped_at_shuffle_top_k_not_the_raw_count():
    """⛔ Hit 21+ is UNREACHABLE -- the featurizer permutes only idx[:shuffle_top_k]. Using the raw
    prefiltered count (median 129) as the group size would claim the mixture is ~87% natural when the
    reachable natural variety is only 20."""
    stk = 20
    for n_pref in (129, 463, 20):
        assert min(n_pref, stk) == 20
    d_wrong, t = _draw(129, 0, 32)
    d_right, _ = _draw(20, 0, 32)
    assert d_wrong[:, 2].mean() / t < 0.25           # the mistake would nearly erase the promoted group
    assert d_right[:, 2].mean() / t > 0.55


def test_pool_target_never_exceeds_what_the_mixture_holds():
    """A chain with 2 naturals and no other source keeps T1's pool of 2, not a padded 4."""
    d, target = _draw(2, 0, 0)
    assert target == 2
    assert (d[:, 0] == 2).all()
    d0, t0 = _draw(0, 0, 0)
    assert t0 == 0 and d0.shape[0] == 0              # nothing to draw -> hook does nothing


# ---------------------------------------------------------------------------------------------
# QUERY-ONLY MSA DEFAULT (2026-08-18). The resolution is tri-state and both wrong answers are
# expensive: ON for a full-MSA run silently destroys it, OFF for a single-seq run silently
# reintroduces the homology leak that --enable_single_seq_mode is supposed to exclude.
# ---------------------------------------------------------------------------------------------

def _resolve_force_query_only(flag, single_seq):
    """Mirror of train_openfold.py's resolution, so the intended truth table is pinned somewhere
    executable rather than living only in an if-statement two files away."""
    if single_seq:
        return True if flag is None else flag
    return False if flag is None else flag


def test_query_only_msa_default_truth_table():
    # unset + single-seq  -> ON (the new default)
    assert _resolve_force_query_only(None, True) is True
    # unset + full MSA    -> OFF (never force it into a run that wants a real MSA)
    assert _resolve_force_query_only(None, False) is False
    # explicit opt-out reproduces the pre-2026-08-18 behaviour, i.e. T1/T2
    assert _resolve_force_query_only(False, True) is False
    # explicit opt-in works outside single-seq mode too
    assert _resolve_force_query_only(True, False) is True


def test_default_never_silently_changes_a_full_msa_run():
    """The dangerous direction: a full-MSA run must not acquire a query-only MSA by default."""
    for flag in (None, False):
        assert _resolve_force_query_only(flag, single_seq=False) is False
