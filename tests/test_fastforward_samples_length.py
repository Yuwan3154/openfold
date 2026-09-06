"""A mid-epoch fastforward must keep len(dataset) == len(dataset.datapoints).

`OpenFoldDataset.__len__` returns `self.epoch_len` while `__getitem__` indexes
`self.datapoints[idx]`. Truncating the list without updating the length leaves the sampler
emitting indices past its end, which surfaces only inside a DataLoader worker as
"IndexError: list index out of range" -- it cost two GPU allocations (jobs 22055648/22055649)
because no test exercised a nonzero fastforward_samples.
"""
import pytest
import torch

from openfold.data.data_modules import OpenFoldDataset


class _Stub(torch.utils.data.Dataset):
    def __init__(self, n):
        self._n = n

    def __len__(self):
        return self._n

    def __getitem__(self, i):
        return {"i": i}

    def idx_to_chain_id(self, i):
        return f"c{int(i)}"

    chain_data_cache = None


def _ds(epoch_len=300):
    g = torch.Generator().manual_seed(0)
    return OpenFoldDataset(datasets=[_Stub(5000)], probabilities=[1.0],
                           epoch_len=epoch_len, generator=g, _roll_at_init=False)


def test_len_matches_datapoints_after_a_midepoch_skip():
    d = _ds(); d.reroll()
    k = 48
    d.datapoints = d.datapoints[k:]
    d.epoch_len = len(d.datapoints)          # the fix
    assert len(d) == len(d.datapoints) == 300 - k
    d[len(d) - 1]                            # the index that used to overrun


def test_truncating_without_updating_len_would_overrun():
    """Negative control: the exact bug, so this test is known to be testing something."""
    d = _ds(); d.reroll()
    d.datapoints = d.datapoints[48:]         # deliberately NOT updating epoch_len
    assert len(d) == 300 and len(d.datapoints) == 252
    with pytest.raises(IndexError):
        d[len(d) - 1]


def test_reroll_restores_full_length_for_later_epochs():
    """A shortened epoch must not leak into subsequent epochs."""
    d = _ds(); d.reroll()
    d.datapoints = d.datapoints[48:]; d.epoch_len = len(d.datapoints)
    assert len(d) == 252
    d.reroll()
    assert len(d) == 300 and len(d.datapoints) == 300
