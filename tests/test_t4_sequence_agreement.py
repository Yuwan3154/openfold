"""T4's read path: the ported sequence-agreement assert, and the padding clobber it exposed.

⛔⛔ TWO DEFECTS THIS FILE PINS (both found 2026-08-25 while testing the pipeline end to end):

1. NO ALIGNMENT CHECK. `SyntheticTemplatePool.sample_features` asserts that its npz aatype equals the
   query sequence at every mapped position -- an off-by-one stays in bounds and would silently place
   every residue's coordinates one slot over, and only that compare catches it. The promoted pool had
   no equivalent: just `(ridx >= 0) & (ridx < n_res)`. A wrong residue frame would have trained
   happily. The assert is now ported.

2. PADDING CLOBBERED REAL RESIDUE 0. `make_fixed_size` pads a chain shorter than crop_size up to
   crop_size with a strictly TRAILING block of residue_index=0 / aatype=0 / no atoms. Those rows pass
   a bounds test, so every one of them scatters onto query position 0; numpy fancy-index assignment
   lets the LAST write win, so they overwrote a genuine residue 0 with zeros and a zero mask.
   Measured on the live pool: 841 of 1500 records (56%) lost their first residue. It is also what
   makes the new assert possible at all -- comparing padded rows against query position 0 produces
   nothing but false failures (that mistake cost a bogus "14.6% misaligned" reading first time round).
"""

import numpy as np
import pytest

from openfold.np import residue_constants as rc
from openfold.utils.t4_pool import PromotedTemplatePool, PromotedTemplateWriter

QUERY = "ACDEFGHIKLMNPQRSTVWY"          # 20 residues, every standard type exactly once
L = len(QUERY)
CROP = 32                                # > L, so a full-length record must be padded


def _write(pool_dir, chain, ridx, aat, *, n_atoms=3, epoch=0, step=0, rank=0, sample=0):
    """Persist one promoted record with EXACTLY the given residue_index / aatype rows."""
    n = len(ridx)
    mask = np.zeros((n, 37), bool)
    for r in range(n):
        # a padded row carries no atoms at all, exactly as make_fixed_size leaves it
        if not (ridx[r] == 0 and aat[r] == 0 and r > 0):
            mask[r, :n_atoms] = True
    coords = np.zeros((n, 37, 3), np.float32)
    coords[:, :n_atoms] = np.arange(n * n_atoms * 3, dtype=np.float32).reshape(n, n_atoms, 3) + 1.0
    w = PromotedTemplateWriter(str(pool_dir), rank)
    w.submit(chain=chain, epoch=epoch, step=step, tm_pred=0.8, tm_template=0.5,
             coords37=coords, atom_mask37=mask, aatype=np.asarray(aat, np.int8),
             residue_index=np.asarray(ridx, np.int32), sample=sample)
    w.close()


def _aat_for(positions):
    return np.array([rc.restype_order[QUERY[p]] for p in positions], np.int8)


def _padded(first, n):
    """A crop covering [first, first+n) then padded out to CROP rows, as make_fixed_size does."""
    real = np.arange(first, first + n)
    ridx = np.concatenate([real, np.zeros(CROP - n, int)])
    aat = np.concatenate([_aat_for(real), np.zeros(CROP - n, np.int8)])
    return ridx, aat


def _sample(pool_dir, chain="1abc_A"):
    p = PromotedTemplatePool(str(pool_dir))
    assert p.refresh() >= 1
    return p.sample_features(chain, 1, np.random.default_rng(0), query_sequence=QUERY)


# ----------------------------------------------------------------- the assert (defect 1)

def test_assert_passes_on_a_correctly_aligned_record(tmp_path):
    ridx = np.arange(L)
    _write(tmp_path, "1abc_A", ridx, _aat_for(ridx))
    f = _sample(tmp_path)
    assert f["template_all_atom_mask"][0, :, 1].sum() == L


def test_assert_fires_when_residue_index_is_shifted_by_one(tmp_path):
    """⭐ THE LOAD-BEARING NEGATIVE CONTROL. Coordinates one slot over stays in bounds, so the old
    bounds-only guard accepted it silently."""
    ridx = np.arange(L)
    shifted = np.clip(ridx + 1, 0, L - 1)
    _write(tmp_path, "1abc_A", shifted, _aat_for(ridx))
    with pytest.raises(AssertionError, match="disagreeing with the query sequence"):
        _sample(tmp_path)


def test_assert_message_names_the_offending_record_and_position(tmp_path):
    ridx = np.arange(L)
    aat = _aat_for(ridx).copy()
    aat[7] = rc.restype_order["W"] if QUERY[7] != "W" else rc.restype_order["A"]
    _write(tmp_path, "1abc_A", ridx, aat, epoch=4, step=99)
    with pytest.raises(AssertionError) as e:
        _sample(tmp_path)
    msg = str(e.value)
    assert "epoch 4" in msg and "step 99" in msg and "query index 7" in msg


def test_unknown_aatype_is_exempt_like_the_T2_assert(tmp_path):
    """aatype 20 decodes to X; T2 exempts X on either side, so T4 must too, or every all-X crop
    (which the live pool does contain) would crash training."""
    ridx = np.arange(L)
    aat = _aat_for(ridx).copy()
    aat[3] = 20
    _write(tmp_path, "1abc_A", ridx, aat)
    f = _sample(tmp_path)
    assert f is not None


# ----------------------------------------------------------------- padding (defect 2)

def test_padding_does_not_clobber_real_residue_zero(tmp_path):
    """⭐ The regression: a padded record covering [0, L) must still deliver residue 0."""
    ridx, aat = _padded(0, L)
    _write(tmp_path, "1abc_A", ridx, aat)
    f = _sample(tmp_path)
    assert f["template_all_atom_mask"][0, 0, 1] == 1.0, "residue 0 was masked out by padding"
    assert np.any(f["template_all_atom_positions"][0, 0] != 0.0), "residue 0 coords zeroed by padding"


def test_padded_record_delivers_exactly_the_real_residues(tmp_path):
    ridx, aat = _padded(0, L)
    _write(tmp_path, "1abc_A", ridx, aat)
    f = _sample(tmp_path)
    assert f["template_all_atom_mask"][0, :, 1].sum() == L


def test_padding_is_not_written_into_the_template_sequence(tmp_path):
    """A pad row decodes to aatype 0 = ALA; if it reached position 0 the sequence would read A."""
    ridx, aat = _padded(0, L)
    _write(tmp_path, "1abc_A", ridx, aat)
    f = _sample(tmp_path)
    seq = f["template_sequence"][0].decode()
    assert seq == QUERY, f"template_sequence is {seq!r}, expected the query"


def test_padded_record_does_not_trip_the_new_assert(tmp_path):
    """Comparing pad rows against query position 0 is exactly the false failure that produced a
    bogus 14.6% misalignment reading; the pad must be cut BEFORE the compare."""
    ridx, aat = _padded(0, L)
    _write(tmp_path, "1abc_A", ridx, aat)
    assert _sample(tmp_path) is not None


def test_offset_crop_with_padding(tmp_path):
    """A crop that does not start at 0 still has its pad rows at residue_index 0, which would land on
    a query position the crop never covered."""
    ridx, aat = _padded(5, L - 5)
    _write(tmp_path, "1abc_A", ridx, aat)
    f = _sample(tmp_path)
    m = f["template_all_atom_mask"][0, :, 1]
    assert m[:5].sum() == 0, "padding leaked onto positions the crop does not cover"
    assert m[5:].sum() == L - 5


def test_unpadded_record_is_unchanged_by_the_fix(tmp_path):
    """Records with no padding must behave exactly as before -- the live pool is 42% unpadded."""
    ridx = np.arange(3, 3 + 10)
    _write(tmp_path, "1abc_A", ridx, _aat_for(ridx))
    f = _sample(tmp_path)
    m = f["template_all_atom_mask"][0, :, 1]
    assert m.sum() == 10 and m[3:13].all() and m[:3].sum() == 0 and m[13:].sum() == 0


def test_out_of_bounds_residue_index_still_guarded(tmp_path):
    """The pre-existing stale-pool guard must survive the rewrite."""
    ridx = np.arange(L - 2, L - 2 + 6)                  # runs past the end of the query
    aat = np.array([rc.restype_order[QUERY[min(p, L - 1)]] for p in ridx], np.int8)
    _write(tmp_path, "1abc_A", ridx, aat)
    f = _sample(tmp_path)
    assert f["template_all_atom_mask"].shape[1] == L
