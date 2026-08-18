"""T4 phase 2: persist promoted predictions and read them back as templates.

When `template_gate_metrics` says a prediction beat the template it was handed, that prediction is
a better template than the one we have -- so it is written here and mixed back into later epochs,
letting the template distribution improve with the model instead of staying frozen at
Protpardelle's output quality.

Two things make this safe to run inside a training loop:

⭐ **DDP without coordination.** Every rank writes only to its own `rank{R}/` subtree and appends to
its own `index.jsonl`; readers merge all ranks at epoch start. No locking, no barrier, no rank-0
bottleneck.

⭐ **The training step never blocks on I/O.** Writes go to a single background thread through a
bounded queue; when the queue is full the promotion is DROPPED rather than stalling the step (and
the drop is counted, so a silently-throttled pool is visible rather than mysterious).

⛔ A promoted prediction is a CROP, not a whole chain. `random_crop_to_size` does NOT record its
offset anywhere -- `train_openfold.py` pops a `num_res_crop_start` key that nothing ever sets -- so
the crop is located by its `residue_index`, which IS a NUM_RES feature and therefore gets cropped
along with the coordinates. On read the template is placed at those residues and masked everywhere
else, which is just an ordinary partial-coverage template as far as the model is concerned.
"""

from __future__ import annotations

import json
import queue
import threading
import zlib
from pathlib import Path

import numpy as np

from openfold.np import residue_constants as rc

CA = 1


class PromotedTemplateWriter:
    """Background writer for promoted predictions. One instance per rank."""

    def __init__(self, pool_dir: str, rank: int, max_queue: int = 64):
        self.root = Path(pool_dir) / f"rank{rank}"
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.jsonl"
        self.n_written = 0
        self.n_dropped = 0
        self._q: queue.Queue = queue.Queue(maxsize=max_queue)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while True:
            item = self._q.get()
            if item is None:
                return
            rec, coords, atom_mask, aatype, residue_index = item
            # ⛔ crc32, NOT builtin hash(): hash() is randomized per process, so the same chain would
            # land in a different directory in every worker and every restart (this exact bug hit
            # the production template run). Sharded because a pool can exceed 1024 files/dir.
            d = self.root / f"shard{zlib.crc32(rec['chain'].encode()) % 1000:04d}"
            d.mkdir(exist_ok=True)
            path = d / f"{rec['chain']}_e{rec['epoch']}_s{rec['step']}.npz"
            np.savez(
                path, coords=coords, atom_mask=atom_mask,
                aatype=aatype, residue_index=residue_index,
            )
            rec["npz"] = str(path.relative_to(self.root))
            with open(self.index_path, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            self.n_written += 1

    def submit(self, chain, epoch, step, tm_pred, tm_template,
               coords37, atom_mask37, aatype, residue_index):
        """Queue one promoted prediction. Non-blocking: drops rather than stalling the step."""
        mask = np.asarray(atom_mask37, dtype=bool)
        rec = {
            "chain": chain, "epoch": int(epoch), "step": int(step),
            "tm_pred": float(tm_pred), "tm_template": float(tm_template),
            "n_res": int(mask.shape[0]),
        }
        item = (
            rec,
            np.asarray(coords37, np.float32)[mask],       # present atoms only, as the T2 npz does
            mask,
            np.asarray(aatype, np.int8),
            np.asarray(residue_index, np.int32),
        )
        try:
            self._q.put_nowait(item)
        except queue.Full:
            # deliberate: a full queue means I/O cannot keep up, and stalling the GPU to persist a
            # template is a worse trade than skipping one. Counted so it is never silent.
            self.n_dropped += 1

    def close(self):
        self._q.put(None)
        self._thread.join(timeout=30)


class PromotedTemplatePool:
    """Read side: merges every rank's index and serves promoted templates per chain.

    Rebuilt at epoch start (`refresh()`), so an epoch trains on a fixed snapshot rather than a pool
    mutating underneath it -- otherwise two dataloader workers could disagree about what exists.
    """

    def __init__(self, pool_dir: str, max_per_chain: int = 0):
        self.root = Path(pool_dir)
        self.max_per_chain = max_per_chain
        self.by_chain: dict[str, list] = {}

    def refresh(self) -> int:
        by_chain: dict[str, list] = {}
        for idx in sorted(self.root.glob("rank*/index.jsonl")):
            rank_root = idx.parent
            for line in idx.read_text().splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                rec["_path"] = rank_root / rec["npz"]
                by_chain.setdefault(rec["chain"], []).append(rec)
        if self.max_per_chain > 0:
            # keep the BEST by the prediction's own TM -- a cap that kept the newest instead would
            # let a late bad epoch evict good templates
            for c, v in by_chain.items():
                v.sort(key=lambda r: -r["tm_pred"])
                by_chain[c] = v[: self.max_per_chain]
        self.by_chain = by_chain
        return sum(len(v) for v in by_chain.values())

    def __contains__(self, chain: str) -> bool:
        return bool(self.by_chain.get(chain))

    def n_for_chain(self, chain: str) -> int:
        """How many promoted templates this chain has in the current snapshot.

        The pre-shuffle draw needs this: `--t4_n_promoted` is the promoted group's WEIGHT in that
        mixture, but a chain cannot contribute more than it actually has, and early in training most
        chains have none at all.
        """
        return len(self.by_chain.get(chain, ()))

    def sample_features(self, chain: str, n: int, rng: np.random.Generator,
                        query_sequence: str) -> dict | None:
        """`n` promoted templates for `chain`, in the NATURAL-HIT layout on the query frame.

        ⛔⛔ THE LAYOUT IS THE WHOLE POINT OF THIS SIGNATURE. Phase 2 returned
        `positions/mask/aatype/tm`, which `merge_template_features` does not consume: it copies only
        keys already present in the natural features and ZERO-FILLS anything it cannot find, so a
        `positions` key would have been silently discarded and the model handed four blank templates.
        This emits exactly what `SyntheticTemplatePool.sample_features` emits, with the same
        `(chain, n, rng, query_sequence)` signature, so the two pools are interchangeable at the call
        site and neither can drift from the layout the feature pipeline expects.

        ⛔ `template_aatype` must be HHBLITS-ordered: `fix_templates_aatype` argmaxes it and then
        gathers through `MAP_HHBLITS_AATYPE_TO_OUR_AATYPE`, so a restype-ordered one-hot silently
        becomes the WRONG amino acids.

        ⚠️ Unlike the synthetic pool -- where every template of a chain covers the same residues and
        can share one sequence -- each promoted template is a different CROP, so coverage differs per
        template and the sequence/one-hot are built per template rather than broadcast.
        """
        recs = self.by_chain.get(chain)
        if not recs:
            return None
        n_res = len(query_sequence)
        pick = rng.choice(len(recs), size=min(n, len(recs)), replace=False)
        k = len(pick)
        pos = np.zeros((k, n_res, 37, 3), np.float32)
        msk = np.zeros((k, n_res, 37), np.float32)
        onehot = np.zeros((k, n_res, 22), np.float32)
        seqs, names = [], []
        for j, i in enumerate(pick):
            rec = recs[i]
            d = np.load(rec["_path"], allow_pickle=False)
            m = d["atom_mask"]                                  # (n_crop, 37) bool
            ridx = d["residue_index"].astype(int)               # (n_crop,) query positions
            # guard a stale pool entry: a chain re-cropped shorter, or a pool carried across runs
            keep = (ridx >= 0) & (ridx < n_res)
            full = np.zeros((m.shape[0], 37, 3), np.float32)
            full[m] = d["coords"]
            q = ridx[keep]
            pos[j, q] = full[keep]
            msk[j, q] = m[keep].astype(np.float32)
            aat = d["aatype"].astype(int)[keep]
            # "-" wherever this crop does not reach, matching templates._extract_template_features
            chars = ["-"] * n_res
            for t, p in enumerate(q):
                chars[p] = rc.restypes[aat[t]] if aat[t] < len(rc.restypes) else "X"
            seq = "".join(chars)
            onehot[j] = rc.sequence_to_onehot(seq, rc.HHBLITS_AA_TO_ID).astype(np.float32)
            seqs.append(seq.encode())
            names.append(f"t4_{chain}_e{rec['epoch']}_s{rec['step']}".encode())
        return {
            "template_all_atom_positions": pos,
            "template_all_atom_mask": msk,
            "template_aatype": onehot,
            "template_sequence": np.array(seqs, dtype=object),
            "template_domain_names": np.array(names, dtype=object),
            # no hhsearch analogue; zeros match empty_template_feats' own convention
            "template_sum_probs": np.zeros((k, 1), np.float32),
            "_tm": np.array([recs[i]["tm_pred"] for i in pick], np.float32),
        }
