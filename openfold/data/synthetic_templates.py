"""T2: feed Protpardelle-1c partial-diffusion templates into training as extra template hits.

Each training chain has one npz holding 64 synthetic templates of itself, generated at rewind levels
375->90 so their TM to the native sweeps roughly 0.2 -> 0.99 (see ESMFOLD2_RECYCLE_SCALING.md,
T2-ALT). `build_template_index.py` scores every one of them against its native and writes the index
this module reads.

Sampling policy (user, 2026-08-14): keep only templates inside the **TM band `min_tm` .. `max_tm`**
(0.3-0.9) -- outside it a template is "too difficult or too easy" -- then **mix with the natural
hhsearch templates and sample uniformly over the mixture**. ⭐ The band matters because the two
sources cover different ranges: natural hhsearch hits pile up at TM 0.5-0.9 (88.8%, only 8.4% below
0.5) while the synthetic ladder is near-uniform over 0.2-0.9, so the synthetic contribution is
mostly the 0.3-0.5 region natural templates barely reach.
The mixing is achieved by concatenating `n_sample` synthetic hits onto the natural
ones and letting the existing train-mode subsampler
(`random_crop_to_size(..., subsample_templates=True)`) draw uniformly from the combined list, which
is exactly the requested behaviour and requires no change to the sampler itself.

⛔ The synthetic template covers the SAME residues in the SAME order as the query (Protpardelle was
seeded from this chain's own structure), so no alignment or residue remapping is needed -- unlike a
real hit, where `_build_query_to_hit_index_mapping` does that work. The one-hot is still built with
`sequence_to_onehot(..., HHBLITS_AA_TO_ID)`, the same call the natural path uses, because
`data_transforms.fix_templates_aatype` reorders from HHBLITS to restype indices downstream and a
hand-rolled one-hot would be silently permuted.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from openfold.np import residue_constants as rc

CA = 1


class SyntheticTemplatePool:
    """Index of generated templates, plus per-chain sampling of template features."""

    def __init__(self, index_path: str, templates_root: str,
                 min_tm: float = 0.3, max_tm: float = 0.9):
        z = np.load(index_path, allow_pickle=False)
        self.tm = z["tm"]                                        # (n_chain, 64)
        self.rewind = z["rewind"]
        # `slot` is present only for an index built by prune_templates_to_band.py, where each npz
        # holds just the in-band templates: it maps an original rung to its row in the pruned file
        # (-1 = dropped). Absent => the npz still has all 64 and the rung index IS the row.
        self.slot = z["slot"] if "slot" in z.files else None
        chains = [str(c) for c in z["chains"]]
        self.row_of = {c: i for i, c in enumerate(chains)}
        self.root = Path(templates_root)
        self.min_tm, self.max_tm = min_tm, max_tm
        # eligible[i] = template indices for chain i inside the TM BAND. It is a band, not a
        # ceiling (user, 2026-08-14): below min_tm the template is too hard to be a useful hint,
        # above max_tm the task is trivial.
        self.eligible = [
            np.flatnonzero((self.tm[i] > min_tm) & (self.tm[i] < max_tm)) for i in range(len(chains))
        ]

    def __contains__(self, chain: str) -> bool:
        return chain in self.row_of and len(self.eligible[self.row_of[chain]]) > 0

    def npz_path(self, chain: str) -> Path:
        import zlib
        # must match generate_templates.py's sharding EXACTLY -- and it is crc32, not hash(),
        # because builtin hash() is randomized per process (see the plan's production-bug notes)
        return self.root / f"shard{zlib.crc32(chain.encode()) % 1000:04d}" / f"{chain}.npz"

    def sample_features(self, chain: str, n: int, rng: np.random.Generator) -> dict | None:
        """`n` synthetic template hits for `chain`, in the raw pre-transform feature layout."""
        if chain not in self:
            return None
        row = self.row_of[chain]
        pool = self.eligible[row]
        pick = rng.choice(pool, size=min(n, len(pool)), replace=False)

        d = np.load(self.npz_path(chain), allow_pickle=False)
        atom_mask = d["atom_mask"]                                # (L, 37) bool
        L = atom_mask.shape[0]
        # ⛔ On a pruned tree the npz rows are a compacted subset, so the rung index picked out of
        # `eligible` is NOT the row index -- translate it, or the coords silently belong to a
        # different template than the TM that justified picking it.
        rows = self.slot[row, pick] if self.slot is not None else pick
        assert (rows >= 0).all(), f"{chain}: picked a rung missing from the pruned npz"
        coords = d["coords"][rows]                                # (k, n_present, 3)
        k = coords.shape[0]
        pos = np.zeros((k, L, 37, 3), np.float32)
        pos[:, atom_mask] = coords

        seq = "".join(rc.restypes[a] if a < len(rc.restypes) else "X"
                      for a in d["aatype"].astype(int))
        onehot = rc.sequence_to_onehot(seq, rc.HHBLITS_AA_TO_ID).astype(np.float32)
        return {
            "template_all_atom_positions": pos,
            "template_all_atom_mask": np.broadcast_to(
                atom_mask.astype(np.float32), (k, L, 37)).copy(),
            "template_aatype": np.broadcast_to(onehot, (k,) + onehot.shape).copy(),
            "template_sequence": np.array([seq.encode()] * k, dtype=object),
            "template_domain_names": np.array(
                [f"pp1c_{chain}_r{int(self.rewind[row, p])}".encode() for p in pick], dtype=object),
            # hhsearch's sum_probs has no synthetic analogue and no model code reads it (it is only
            # carried through and reshaped); zeros match empty_template_feats' own convention.
            "template_sum_probs": np.zeros((k, 1), np.float32),
            "_tm": self.tm[row, pick].astype(np.float32),         # diagnostic, dropped on merge
        }


def merge_template_features(feats: dict, synth: dict) -> dict:
    """Concatenate synthetic hits onto whatever template features `feats` already has.

    Only keys already present in `feats` are touched, so the merged dict keeps exactly the schema
    the feature pipeline expects. A key the synthetic side cannot produce (e.g.
    `template_dgram_probs`) is zero-extended to keep the template axis consistent -- a ragged
    template axis would break `np.stack`/cropping downstream in a way that is hard to trace.
    """
    keys = [k for k in feats if k.startswith("template_")]
    if not keys:
        return feats
    k_new = synth["template_all_atom_positions"].shape[0]
    out = dict(feats)
    for key in keys:
        old = np.asarray(feats[key])
        if key in synth:
            add = np.asarray(synth[key]).astype(old.dtype) if old.dtype != object \
                else np.asarray(synth[key])
        else:
            add = np.zeros((k_new,) + old.shape[1:], old.dtype)
        # an all-zeros "no templates found" placeholder has a 0-length template axis for the real
        # arrays but a length-1 axis for the two object arrays; drop that placeholder rather than
        # concatenating onto it, or the model gets a phantom empty hit
        if old.dtype == object and old.shape[0] == 1 and old[0] in (b"", ""):
            out[key] = add
        else:
            out[key] = np.concatenate([old, add], axis=0)
    return out
