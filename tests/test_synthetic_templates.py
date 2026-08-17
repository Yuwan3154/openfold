"""Gate tests for T2's synthetic-template injection (openfold/data/synthetic_templates.py).

The failure modes worth catching here are all silent ones: a permuted amino-acid one-hot, a TM
filter that lets the trivial templates through, and a merge that leaves a phantom empty hit in the
template axis.
"""

import numpy as np
import pytest
import torch
import zlib

from openfold.data.synthetic_templates import SyntheticTemplatePool, merge_template_features
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
    return SyntheticTemplatePool(str(tmp_path / "index_all.npz"), str(root),
                                 min_tm=0.3, max_tm=0.9), aatype


def test_tm_filter_is_a_band_not_a_ceiling(pool):
    """Eligibility is 0.3 < TM < 0.9 (user, 2026-08-14): outside it a template is too hard or
    too easy. A ceiling-only filter would silently readmit the sub-0.3 tail."""
    p, _ = pool
    keep = p.eligible[p.row_of["1abc_A"]]
    assert set(keep) == set(np.flatnonzero((p.tm[0] > 0.3) & (p.tm[0] < 0.9)))
    assert p.tm[0][keep].min() > 0.3 and p.tm[0][keep].max() < 0.9
    assert (p.tm[0] <= 0.3).any(), "fixture must contain a too-hard template for this to bite"


def test_chain_with_no_eligible_template_is_skipped(pool):
    """2xyz_B's templates are all TM 0.95, i.e. all too easy -- it must yield nothing, not crash."""
    p, _ = pool
    assert "2xyz_B" not in p
    assert p.sample_features("2xyz_B", 2, np.random.default_rng(0)) is None
    assert p.sample_features("nosuch_A", 2, np.random.default_rng(0)) is None


def test_sampled_features_have_the_natural_layout(pool):
    p, _ = pool
    f = p.sample_features("1abc_A", 3, np.random.default_rng(1))
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
    p, _ = pool
    n_elig = len(p.eligible[p.row_of["1abc_A"]])
    f = p.sample_features("1abc_A", 999, np.random.default_rng(2))
    assert f["template_all_atom_positions"].shape[0] == n_elig
    assert len(set(f["_tm"].tolist())) == n_elig      # sampled without replacement


def test_aatype_onehot_survives_the_hhblits_reordering(pool):
    """⛔ The one-hot must be in HHBLITS order: `fix_templates_aatype` argmaxes then gathers through
    MAP_HHBLITS_AATYPE_TO_OUR_AATYPE, so a hand-rolled restype-order one-hot comes out permuted."""
    p, aatype = pool
    f = p.sample_features("1abc_A", 2, np.random.default_rng(3))
    prot = {"template_aatype": torch.tensor(f["template_aatype"])}
    fix_templates_aatype(prot)
    assert torch.equal(prot["template_aatype"][0], torch.tensor(aatype, dtype=torch.long))


def test_merge_concatenates_onto_natural_hits(pool):
    p, _ = pool
    nat = {
        "template_all_atom_positions": np.zeros((2, L, 37, 3), np.float32),
        "template_all_atom_mask": np.zeros((2, L, 37), np.float32),
        "template_aatype": np.zeros((2, L, 22), np.float32),
        "template_sum_probs": np.zeros((2, 1), np.float32),
        "template_domain_names": np.array([b"1aaa_A", b"1bbb_B"], dtype=object),
        "template_sequence": np.array([b"AAA", b"BBB"], dtype=object),
        "aatype": np.zeros((L,), np.int64),                       # untouched non-template feature
    }
    f = p.sample_features("1abc_A", 3, np.random.default_rng(4))
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
    p, _ = pool
    nat = empty_template_feats(L)
    nat.pop("template_dgram_probs")                               # not built by process_mmcif here
    f = p.sample_features("1abc_A", 2, np.random.default_rng(5))
    out = merge_template_features(nat, f)
    assert out["template_all_atom_positions"].shape[0] == 2
    assert len(out["template_domain_names"]) == 2
    assert len(out["template_sequence"]) == 2
    assert all(n.startswith(b"pp1c_") for n in out["template_domain_names"])


def test_merge_zero_extends_keys_the_synthetic_side_cannot_produce(pool):
    p, _ = pool
    nat = {
        "template_all_atom_positions": np.zeros((1, L, 37, 3), np.float32),
        "template_dgram_probs": np.ones((1, L, L, 39), np.float32),
    }
    out = merge_template_features(nat, p.sample_features("1abc_A", 2, np.random.default_rng(6)))
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
    np.savez(
        d / f"{chain}.npz", coords=coords, atom_mask=atom_mask,
        aatype=rng.integers(0, 20, L).astype(np.int8),
        residue_index=np.arange(1, L + 1, dtype=np.int32),
        rewind_steps=(90 + keep).astype(np.int16),
    )
    p = SyntheticTemplatePool(str(tmp_path / "index_all.npz"), str(tmp_path / "templates"),
                              min_tm=0.3, max_tm=0.9)
    f = p.sample_features(chain, len(keep), np.random.default_rng(0))
    got = f["template_all_atom_positions"][:, 0, 1, 0]             # the constant, per template
    # each sampled template's TM must be the one belonging to the row its coords came from
    for tm_val, row in zip(f["_tm"], got):
        assert tm[0][keep[int(row)]] == pytest.approx(tm_val), "coords paired with the wrong TM"


def test_unpruned_index_still_reads_the_rung_directly(pool):
    """The `slot` key is absent on the original 64-rung index -- that path must be untouched."""
    p, _ = pool
    assert p.slot is None
    f = p.sample_features("1abc_A", 2, np.random.default_rng(0))
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
    return str(tmp_path / "index_all.npz"), str(tmp_path / "templates")


def test_pruned_index_refuses_a_wider_band(tmp_path):
    """⛔ A pruned tree physically lacks the out-of-band rungs, so a wider band at training time
    would select rows that were never written. That must fail at construction with the numbers in
    hand, not on some later step -- a run that dies 3 hours in is far worse than one that never
    starts."""
    idx, root = _pruned_index(tmp_path, 0.3, 0.9)
    with pytest.raises(AssertionError, match="pruned to TM"):
        SyntheticTemplatePool(idx, root, min_tm=0.2, max_tm=0.9)
    with pytest.raises(AssertionError, match="pruned to TM"):
        SyntheticTemplatePool(idx, root, min_tm=0.3, max_tm=0.95)


def test_pruned_index_accepts_the_same_or_narrower_band(tmp_path):
    idx, root = _pruned_index(tmp_path, 0.3, 0.9)
    same = SyntheticTemplatePool(idx, root, min_tm=0.3, max_tm=0.9)
    narrower = SyntheticTemplatePool(idx, root, min_tm=0.4, max_tm=0.8)
    assert len(narrower.eligible[0]) < len(same.eligible[0])
    # a narrower band still indexes real rows -- the translation must hold
    f = narrower.sample_features("1abc_A", 2, np.random.default_rng(0))
    assert f["template_all_atom_positions"].shape[0] == 2
