"""`_flush_per_entry_records` must issue the same collectives on every DDP rank.

⛔⛔ THE BUG THIS PINS (root-caused 2026-09-22, T-1): the flush read `trainer.log_dir` only inside
`if records and self.trainer.is_global_zero`. In Lightning 2.5.1 that property ends in
`strategy.broadcast(dirpath)` = broadcast_object_list = two NCCL BROADCASTs, so rank 0 ran two
collectives ahead of ranks 1-3 for the rest of the process. Every NCCL watchdog timeout on record
carries that +2 offset on rank 0 (T1 2026-08-11 x2, Run C v2 ep89 and 2026-09-21, Run D 2026-09-08),
and prune_work/nccl_asym_repro.py reproduces the 2026-09-21 hang exactly (BROADCAST NumelIn=4472).

The test counts `trainer.log_dir` reads per simulated rank: equal counts on rank 0 and a non-zero
rank is the invariant. The old code read it once on rank 0 and never on rank 1, so this fails on it.
"""

import types

from train_openfold import OpenFoldWrapper


class _Trainer:
    def __init__(self, is_global_zero, log_dir):
        self.is_global_zero = is_global_zero
        self._log_dir = log_dir
        self.log_dir_reads = 0

    @property
    def log_dir(self):
        self.log_dir_reads += 1
        return self._log_dir


def _rank(is_global_zero, log_dir, records):
    return types.SimpleNamespace(
        trainer=_Trainer(is_global_zero, log_dir),
        _is_distributed=False,  # records stand in for the post-gather list, identical on every rank
        _val_per_entry_records=list(records),
        _per_entry_csv_path=None,
        _val_per_entry_epoch=102,
        _val_per_entry_step=78001,
    )


def _flush(r, records):
    r._val_per_entry_records = list(records)
    OpenFoldWrapper._flush_per_entry_records(r)


REC = [(0, 0.5, 1.0, 0.25, 0.4), (1, 0.6, 2.0, 0.5, 0.45)]


def test_log_dir_is_read_on_every_rank_the_same_number_of_times(tmp_path):
    r0, r1 = _rank(True, str(tmp_path), REC), _rank(False, str(tmp_path), REC)
    for records in (REC, [], REC):  # first flush, an empty epoch, a later flush
        _flush(r0, records)
        _flush(r1, records)
        assert r0.trainer.log_dir_reads == r1.trainer.log_dir_reads, \
            "rank 0 and rank 1 issued different numbers of log_dir broadcasts -> DDP desync"
    assert r0.trainer.log_dir_reads == 1, "the path must be resolved once per process, then cached"


def test_only_rank_zero_writes_and_the_csv_is_unchanged(tmp_path):
    r0, r1 = _rank(True, str(tmp_path), REC), _rank(False, str(tmp_path), REC)
    _flush(r0, REC)
    _flush(r1, REC)
    rows = (tmp_path / "per_entry_val_history.csv").read_text().splitlines()
    assert rows[0] == "epoch,global_step,batch_idx,lddt_ca,alignment_rmsd,recall_2A,gdt_ts"
    assert rows[1:] == ["102,78001,0,0.500000,1.000000,0.250000,0.400000",
                        "102,78001,1,0.600000,2.000000,0.500000,0.450000"], \
        "rank 1 must not append, and the row format must not change"


def test_no_records_means_no_log_dir_read_on_any_rank(tmp_path):
    r0, r1 = _rank(True, str(tmp_path), []), _rank(False, str(tmp_path), [])
    _flush(r0, [])
    _flush(r1, [])
    assert r0.trainer.log_dir_reads == r1.trainer.log_dir_reads == 0
    assert not (tmp_path / "per_entry_val_history.csv").exists()
