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
                 min_tm: float = 0.3, max_tm: float = 0.9,
                 qmap_path: str | None = None):
        z = np.load(index_path, allow_pickle=False)
        self.tm = z["tm"]                                        # (n_chain, 64)
        self.rewind = z["rewind"]
        # `slot` is present only for an index built by prune_templates_to_band.py, where each npz
        # holds just the in-band templates: it maps an original rung to its row in the pruned file
        # (-1 = dropped). Absent => the npz still has all 64 and the rung index IS the row.
        self.slot = z["slot"] if "slot" in z.files else None
        if self.slot is not None:
            # ⛔ A pruned tree physically contains only its own band, so asking for a WIDER band
            # would select rungs whose npz rows were never written. Fail at construction with the
            # actual numbers rather than on some later training step's assert.
            assert "min_tm" in z.files and "max_tm" in z.files, (
                "index carries `slot` (pruned) but does not record the band it was pruned to, so "
                "the widening guard cannot run -- rebuild it with prune_templates_to_band.py"
            )
            # ⛔ compare in float32, not float64: np.float32(0.3) is 0.30000001192, which is
            # GREATER than the python float 0.3, so a float64 comparison rejects an exactly
            # matching band. Casting both sides to the stored precision makes equality exact.
            pruned_lo, pruned_hi = np.float32(z["min_tm"]), np.float32(z["max_tm"])
            assert np.float32(min_tm) >= pruned_lo and np.float32(max_tm) <= pruned_hi, (
                f"index was pruned to TM {float(pruned_lo)}-{float(pruned_hi)} but the pool asks "
                f"for {min_tm}-{max_tm}; re-prune from the full template tree for a wider band"
            )
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

        # ⛔⛔ The npz-row -> query-position map. This is NOT optional bookkeeping: the npz's own
        # `residue_index` is protpardelle's structure parse (see build_query_index_map.py), and
        # `residue_index - 1` desynchronises at the first unresolved residue. A chain with no entry
        # here is treated as UNAVAILABLE rather than falling back to that arithmetic -- a silent
        # fallback to a subtly wrong mapping is exactly what cost this project a debugging detour.
        self.qmap: dict[str, np.ndarray] = {}
        self.qmap_query_len: dict[str, int] = {}
        self.qmap_ambiguous: dict[str, bool] = {}
        if qmap_path is not None:
            zq = np.load(qmap_path, allow_pickle=False)
            flat, lens = zq["qmap"], zq["qmap_len"]
            offs = np.concatenate([[0], np.cumsum(lens)])
            amb = zq["ambiguous"] if "ambiguous" in zq.files else np.zeros(len(lens), np.int8)
            for j, c in enumerate(zq["chains"]):
                c = str(c)
                self.qmap[c] = flat[offs[j]:offs[j + 1]]
                self.qmap_query_len[c] = int(zq["query_len"][j])
                self.qmap_ambiguous[c] = bool(amb[j])

    def __contains__(self, chain: str) -> bool:
        if chain not in self.row_of or len(self.eligible[self.row_of[chain]]) == 0:
            return False
        # no query map => cannot place this chain's templates correctly => it has none
        return not self.qmap or chain in self.qmap

    def npz_path(self, chain: str) -> Path:
        import zlib
        # must match generate_templates.py's sharding EXACTLY -- and it is crc32, not hash(),
        # because builtin hash() is randomized per process (see the plan's production-bug notes)
        return self.root / f"shard{zlib.crc32(chain.encode()) % 1000:04d}" / f"{chain}.npz"

    def sample_features(self, chain: str, n: int, rng: np.random.Generator,
                        query_sequence: str) -> dict | None:
        """`n` synthetic template hits for `chain`, on the QUERY's residue frame.

        ⛔⛔ The npz is on the NATIVE STRUCTURE's frame, not the query's: it holds only the residues
        that were RESOLVED in the deposited structure, numbered by the native PDB (1-based, and not
        necessarily contiguous). The query frame is the full sequence, 0..L-1. Those differ for the
        large majority of chains -- measured 10 of 12 in a random sample, e.g. 3dls_A is a
        335-residue query whose npz has 285 residues numbered 9-293. Building the arrays at the npz
        length is what crashed the first T2 launch inside `merge_template_features`
        ("size 104 vs 89").
        So this scatters onto the query frame and masks everywhere else -- exactly what the natural
        path does in `templates._extract_template_features`, which zero-initializes over the full
        query, writes only mapped positions, and leaves "-" in the sequence elsewhere.
        """
        if chain not in self:
            return None
        row = self.row_of[chain]
        pool = self.eligible[row]
        pick = rng.choice(pool, size=min(n, len(pool)), replace=False)

        d = np.load(self.npz_path(chain), allow_pickle=False)
        atom_mask = d["atom_mask"]                                # (n_native, 37) bool
        # ⛔ On a pruned tree the npz rows are a compacted subset, so the rung index picked out of
        # `eligible` is NOT the row index -- translate it, or the coords silently belong to a
        # different template than the TM that justified picking it.
        rows = self.slot[row, pick] if self.slot is not None else pick
        assert (rows >= 0).all(), f"{chain}: picked a rung missing from the pruned npz"
        coords = d["coords"][rows]                                # (k, n_present, 3)
        k = coords.shape[0]
        native = np.zeros((k,) + atom_mask.shape + (3,), np.float32)
        native[:, atom_mask] = coords

        qL = len(query_sequence)
        aat = d["aatype"].astype(int)
        if chain in self.qmap:
            q = self.qmap[chain].astype(int)
            assert self.qmap_query_len[chain] == qL, (
                f"{chain}: qmap was built against a query of length "
                f"{self.qmap_query_len[chain]} but this query is {qL} -- the map is stale"
            )
        else:
            # only reachable when NO qmap was supplied at all (legacy/tests). ⛔ Not a fallback for
            # a chain merely absent from a supplied map -- __contains__ excludes those.
            assert not self.qmap, f"{chain}: absent from the qmap; should have been filtered"
            q = d["residue_index"].astype(int) - 1
        assert len(q) == len(aat), f"{chain}: qmap has {len(q)} rows, npz has {len(aat)}"
        assert q.min() >= 0 and q.max() < qL, (
            f"{chain}: mapped query positions {q.min()}-{q.max()} do not fit a query of length {qL}"
        )
        # ⭐ Sequence agreement is the real check on that mapping: an off-by-one would still be
        # in-bounds and would silently place every residue's coordinates one slot over. Cheap
        # vector compare, so it runs on every sample rather than only in tests.
        letters = np.array([rc.restypes[a] if a < len(rc.restypes) else "X" for a in aat])
        qchars = np.array(list(query_sequence))[q]
        bad = (letters != qchars) & (letters != "X") & (qchars != "X")
        assert not bad.any(), (
            f"{chain}: npz aatype disagrees with the query sequence at {int(bad.sum())}/{len(q)} "
            f"scattered positions (first at query index {int(q[bad][0])}: npz "
            f"{letters[bad][0]} vs query {qchars[bad][0]}) -- residue mapping is wrong"
        )

        pos = np.zeros((k, qL, 37, 3), np.float32)
        msk = np.zeros((k, qL, 37), np.float32)
        pos[:, q] = native
        msk[:, q] = atom_mask.astype(np.float32)

        # "-" everywhere the template does not cover, matching the natural path's convention
        seq_chars = ["-"] * qL
        for j, p in enumerate(q):
            seq_chars[p] = letters[j]
        seq = "".join(seq_chars)
        onehot = rc.sequence_to_onehot(seq, rc.HHBLITS_AA_TO_ID).astype(np.float32)
        return {
            "template_all_atom_positions": pos,
            "template_all_atom_mask": msk,
            "template_aatype": np.broadcast_to(onehot, (k,) + onehot.shape).copy(),
            "template_sequence": np.array([seq.encode()] * k, dtype=object),
            "template_domain_names": np.array(
                [f"pp1c_{chain}_r{int(self.rewind[row, p])}".encode() for p in pick], dtype=object),
            # hhsearch's sum_probs has no synthetic analogue and no model code reads it (it is only
            # carried through and reshaped); zeros match empty_template_feats' own convention.
            "template_sum_probs": np.zeros((k, 1), np.float32),
            "_tm": self.tm[row, pick].astype(np.float32),         # diagnostic, dropped on merge
        }


def natural_template_count(feats: dict) -> int:
    """How many real template hits `feats` carries.

    Read off the numeric array, never the object ones: the "no templates found" placeholder
    (`empty_template_feats`) gives the numeric arrays a 0-length template axis but the two object
    arrays a length-1 one, so `template_sequence` would report 1 hit where there are none.
    """
    pos = feats.get("template_all_atom_positions")
    return 0 if pos is None else int(np.asarray(pos).shape[0])


def subsample_natural_templates(feats: dict, keep: int, rng: np.random.Generator) -> dict:
    """Keep a uniformly random `keep` of the natural template hits, dropping the rest.

    ⭐⭐ WHY RANDOM AND NOT WORST-FIRST. This exists for the count-matched run, whose whole point is
    that the natural component stays distributionally IDENTICAL to T1's so the only difference left
    is template content. `HhsearchHitFeaturizer` hands the hits over sorted by `sum_probs`, so
    keeping the top-k would give this run systematically BETTER natural templates than the random
    subset T1 actually trains on -- a second difference on top of the one being measured, and one
    that biases in the flattering direction. A uniform draw makes each surviving slot the same draw
    T1 would have made. (Worst-first is the right choice for a *production* model, not for this
    comparison; it is a one-line change if that is ever wanted.)

    Position within the pool is irrelevant either way -- `random_crop_to_size(subsample_templates=
    True)` permutes the whole pool before taking its window -- so only WHICH hits survive matters.
    """
    n = natural_template_count(feats)
    if n == 0 or keep >= n:
        return feats
    idx = np.sort(rng.choice(n, size=keep, replace=False))
    out = dict(feats)
    for key in [k for k in feats if k.startswith("template_")]:
        arr = np.asarray(feats[key])
        if arr.shape[0] != n:
            raise ValueError(
                f"{key} has template axis {arr.shape[0]} but there are {n} natural hits; the "
                f"template features are ragged, which cropping downstream cannot handle"
            )
        out[key] = arr[idx]
    return out


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
    # ⛔ Guard the NUM_RES axis explicitly. The first T2 launch died here on "size 104 vs 89"
    # because the synthetic side was built on the native structure's resolved-residue frame while
    # the natural side is on the query's. np.concatenate's own message names only the axis sizes,
    # which is a long way from naming the cause -- so say it here.
    old_pos = np.asarray(feats["template_all_atom_positions"])
    if old_pos.shape[0] and old_pos.shape[1] != synth["template_all_atom_positions"].shape[1]:
        raise ValueError(
            f"template NUM_RES mismatch: natural hits have {old_pos.shape[1]} residues, synthetic "
            f"have {synth['template_all_atom_positions'].shape[1]}. The synthetic features must be "
            f"built on the QUERY frame (pass the query sequence to sample_features), not the "
            f"native structure's resolved-residue frame."
        )
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
