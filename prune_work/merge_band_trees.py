"""Concatenate two band-pruned T2 template trees into one the trainer can read.

`--t2_templates_root` and `--t2_template_index` are SINGULAR, and a chain's templates live in ONE
npz, so "round 2 ADDED to round 1" is physically a per-chain concatenation, not a second pool.

Both inputs must already be band-pruned by prune_templates_to_band.py, which is what makes this a
concatenation rather than a re-derivation: each source npz holds exactly its in-band rows and its
index carries the `slot` map. The merged npz is round-1 rows followed by round-2 rows, and the
merged index re-derives `slot` over that layout.

⛔⛔ THE ONE THING THAT MUST BE CHECKED, AND IS. The npz stores coords for present atoms only,
against a shared `atom_mask`, and the trainer reconstructs with `full[:, atom_mask] = coords`. If
the two rounds disagree about a chain's atom_mask / aatype / residue_index -- a re-extracted native,
a different mmCIF snapshot, anything -- concatenating their coords silently scatters round-2 atoms
onto round-1 positions and every downstream number is quietly wrong with no error. So every chain is
gated on all three being bit-identical, and a mismatch is recorded and the chain SKIPPED, never
merged (skip-and-record, not assert: one bad chain must not cost a whole shard -- round 1 lost 1,723
chains to exactly that mistake).

⛔ Writes a NEW tree and never touches either source. Two training runs read the live round-1 tree,
and `SyntheticTemplatePool.sample_features` opens a chain's npz LAZILY on every draw, so rewriting
one in place is a live-run crash, not a stale read.
"""

from __future__ import annotations

import argparse
import zlib
from pathlib import Path

import numpy as np

SHARED = ("atom_mask", "aatype", "residue_index")


def shard_dir(root: Path, chain: str) -> Path:
    # ⛔ crc32, not builtin hash(): the same stability requirement as generation and pruning
    return root / f"shard{zlib.crc32(chain.encode()) % 1000:04d}"


def load_index(p: str) -> dict:
    z = np.load(p, allow_pickle=False)
    assert "slot" in z.files, f"{p} is not a band index (no `slot`); prune it first"
    return {k: z[k] for k in z.files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-a", required=True, help="round-1 index_band.npz")
    ap.add_argument("--root-a", required=True, help="round-1 templates_band")
    ap.add_argument("--index-b", required=True, help="round-2 index_band.npz")
    ap.add_argument("--root-b", required=True, help="round-2 templates_band")
    ap.add_argument("--dst-root", required=True)
    ap.add_argument("--out-index", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    # ⛔ Without this every copy shard would np.savez the SAME index path concurrently -- the exact
    # bug round 1 hit. The index is written once, by a job chained after ALL copy shards.
    ap.add_argument("--index-only", action="store_true")
    a = ap.parse_args()

    A, B = load_index(a.index_a), load_index(a.index_b)
    for k in ("min_tm", "max_tm"):
        assert np.float32(A[k]) == np.float32(B[k]), (
            f"the two trees were pruned to different bands ({k}: {A[k]} vs {B[k]}); "
            f"a merged pool must share one band or `eligible` means two different things"
        )
    ca = [str(c) for c in A["chains"]]
    cb = [str(c) for c in B["chains"]]
    row_b = {c: i for i, c in enumerate(cb)}
    # round 1 is the reference population: round 2 regenerates for a subset of the same chains
    only_b = sorted(set(cb) - set(ca))
    chains = ca + only_b
    print(f"round 1: {len(ca)} chains   round 2: {len(cb)}   merged: {len(chains)} "
          f"({len(only_b)} present only in round 2)", flush=True)

    na, nb = A["tm"].shape[1], B["tm"].shape[1]
    print(f"rungs per chain: {na} + {nb} = {na + nb}", flush=True)

    if a.index_only:
        n = len(chains)
        tm = np.zeros((n, na + nb), np.float32)
        rw = np.zeros((n, na + nb), np.int16)
        L = np.zeros(n, np.int32)
        present_b = np.zeros(n, bool)
        for i, c in enumerate(chains):
            if i < len(ca):
                tm[i, :na], rw[i, :na], L[i] = A["tm"][i], A["rewind"][i], A["length"][i]
            else:
                # a round-2-only chain has no round-1 half; mark those rungs unusable via a TM
                # outside the band rather than 0, which would sit inside a band whose min is 0
                tm[i, :na] = -1.0
                rw[i, :na] = -1
            j = row_b.get(c)
            if j is not None:
                tm[i, na:], rw[i, na:] = B["tm"][j], B["rewind"][j]
                L[i] = B["length"][j] if i >= len(ca) else L[i]
                present_b[i] = True
            else:
                tm[i, na:] = -1.0
                rw[i, na:] = -1

        band = (tm > np.float32(A["min_tm"])) & (tm < np.float32(A["max_tm"]))
        slot = np.full(tm.shape, -1, np.int16)
        for i in range(len(chains)):
            slot[i, np.flatnonzero(band[i])] = np.arange(int(band[i].sum()), dtype=np.int16)
        np.savez(a.out_index, chains=np.array(chains, dtype="<U8"), tm=tm, rewind=rw, length=L,
                 slot=slot, min_tm=A["min_tm"], max_tm=A["max_tm"])
        kept = int((slot >= 0).sum())
        print(f"wrote {a.out_index}: {len(chains)} chains, {kept:,}/{slot.size:,} rungs in band "
              f"({100 * kept / slot.size:.1f}%), band {float(A['min_tm'])}-{float(A['max_tm'])}")
        print(f"  in-band per chain: min {band.sum(1).min()}  med {int(np.median(band.sum(1)))}  "
              f"mean {band.sum(1).mean():.1f}  max {band.sum(1).max()}")
        print(f"  chains with ZERO in-band: {int((band.sum(1) == 0).sum())}")
        return

    dst = Path(a.dst_root)
    mine = chains[a.shard::a.num_shards]
    done = copied = skipped = 0
    mismatches = []
    for c in mine:
        pa = shard_dir(Path(a.root_a), c) / f"{c}.npz"
        pb = shard_dir(Path(a.root_b), c) / f"{c}.npz"
        out = shard_dir(dst, c) / f"{c}.npz"
        if out.is_file():
            skipped += 1
            continue
        out.parent.mkdir(parents=True, exist_ok=True)

        has_a, has_b = pa.is_file(), pb.is_file()
        if not (has_a or has_b):
            mismatches.append(f"{c}:absent_from_both")
            continue
        if has_a and not has_b:
            da = np.load(pa, allow_pickle=False)
            np.savez(out, **{k: da[k] for k in da.files})
            copied += 1
            continue
        if has_b and not has_a:
            db = np.load(pb, allow_pickle=False)
            np.savez(out, **{k: db[k] for k in db.files})
            copied += 1
            continue

        da, db = np.load(pa, allow_pickle=False), np.load(pb, allow_pickle=False)
        bad = [k for k in SHARED if not np.array_equal(da[k], db[k])]
        if bad:
            # ⛔ the frames disagree; concatenating would scatter round-2 atoms onto round-1
            # positions with no error anywhere downstream
            mismatches.append(f"{c}:frame_differs_{'+'.join(bad)}")
            continue

        merged = {k: da[k] for k in SHARED}
        merged["coords"] = np.concatenate([da["coords"], db["coords"]], axis=0)
        merged["rewind_steps"] = np.concatenate(
            [da["rewind_steps"], db["rewind_steps"]]).astype(np.int16)
        merged["n_round1"] = np.int16(da["coords"].shape[0])
        merged["n_round2"] = np.int16(db["coords"].shape[0])
        for k in ("model", "schedule"):
            if k in da.files and k in db.files:
                merged[f"{k}_r1"], merged[f"{k}_r2"] = da[k], db[k]
        for k in ("ladder_start", "ladder_top", "n_rungs", "seeds_per_rung"):
            if k in db.files:
                merged[f"r2_{k}"] = db[k]
        np.savez(out, **merged)
        done += 1

    print(f"shard {a.shard}: {done} merged, {copied} copied through (one side only), "
          f"{skipped} already present, {len(mismatches)} SKIPPED")
    if mismatches:
        # ⛔ a count is not evidence -- every skip is written out so each can be re-checked
        rec = Path(a.dst_root) / f"_skipped_{a.shard:04d}.txt"
        rec.write_text("\n".join(mismatches) + "\n")
        print(f"  every skip recorded in {rec}")


if __name__ == "__main__":
    main()
