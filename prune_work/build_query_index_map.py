"""Recover the npz-row -> query-position map that the generated templates are missing.

⛔ WHY THIS EXISTS. The npz's `residue_index` came from protpardelle's `read_pdb` of the mmCIF
(`extract_train_natives.py` stores `protein_obj.residue_index`), i.e. protpardelle's own structure
parse with its own notion of a usable residue. It is NOT the openfold query index, and no arithmetic
converts it:
  * `ridx - 1` is right until the first unresolved residue, then desynchronises -- 1eis_A is a
    90-residue query whose 85-row npz is numbered contiguously 2-86 despite lacking query position 9,
    so the gap is invisible in the numbering (15/85 positions matched).
  * "the resolved seqres positions, in order" also fails -- the mmCIF reports 86 resolved for 1eis_A
    but the npz has 85 rows (5gk0_A: 380 vs 379), because the extraction additionally drops residues
    without usable atoms (the same case as 3ra5_A's 97 CA lines for L=98).
So the correspondence has to be recovered by ALIGNING the npz sequence to the query sequence, which
is what AF2 does for real templates (`_build_query_to_hit_index_mapping`).

⭐ The coordinates are fine; only this bookkeeping was missing. No regeneration is needed.

TIE-BREAKS, stated explicitly because they are correctness choices and not formatting ones:
  * The npz sequence must be a SUBSEQUENCE of the query -- same protein, residues only ever dropped,
    never inserted or substituted. Anything else is reported as a failure rather than forced.
  * `residue_index - 1` WINS whenever it is sequence-valid. It was measured sequence-exact on
    1500/1500 randomly sampled chains, whereas a deletion inside a run of identical residues makes
    the alignment ambiguous for 17-48% of chains, where a leftmost pick is arbitrary. Preferring an
    arbitrary embedding over consistent structural numbering would INTRODUCE errors in chains that
    were already right. The alignment handles only what ridx-1 provably cannot explain.
  * Where the alignment IS used and is ambiguous, the leftmost embedding is taken and the chain is
    flagged, so the residual risk is a counted number rather than a hope.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from openfold.data import mmcif_parsing
from openfold.np import residue_constants as rc

ap = argparse.ArgumentParser()
ap.add_argument("--index", required=True, help="index_band.npz (or index_all.npz)")
ap.add_argument("--templates-root", required=True)
ap.add_argument("--mmcif-dir", required=True)
ap.add_argument("--out", required=True, help="output .npz for this shard")
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--num-shards", type=int, default=1)
ap.add_argument("--progress-every", type=int, default=200)
a = ap.parse_args()


def _match(sc: str, qc: str) -> bool:
    # npz aatype 20 renders as "X" (unknown residue); it matches whatever the query has there, since
    # identity simply cannot be checked at that position. Wildcards stay correct for greedy matching
    # because a wildcard matches everything, so consuming it as early as possible never blocks a
    # later match.
    return sc == qc or sc == "X"


def subsequence_map(short: str, long: str):
    """Leftmost embedding of `short` into `long` as a subsequence, or None if there is none."""
    out = np.empty(len(short), np.int32)
    j = 0
    for i, ch in enumerate(long):
        if j < len(short) and _match(short[j], ch):
            out[j] = i
            j += 1
    return out if j == len(short) else None


def rightmost_map(short: str, long: str):
    out = np.empty(len(short), np.int32)
    j = len(short) - 1
    for i in range(len(long) - 1, -1, -1):
        if j >= 0 and _match(short[j], long[i]):
            out[j] = i
            j -= 1
    return out if j < 0 else None


z = np.load(a.index, allow_pickle=False)
chains = [str(c) for c in z["chains"]]
# ⭐ Sort by mmCIF file id and shard in CONTIGUOUS blocks, not with a stride: the cost here is
# dominated by `mmcif_parsing.parse`, and multiple chains of the same entry (1abc_A, 1abc_B, ...)
# then land in the same shard back to back so one parse serves all of them. A strided slice would
# scatter them across shards and re-parse the same file in every one.
chains = sorted(chains, key=lambda c: (c.rsplit("_", 1)[0], c))
chunk = (len(chains) + a.num_shards - 1) // a.num_shards
mine = list(range(a.shard * chunk, min((a.shard + 1) * chunk, len(chains))))
root = Path(a.templates_root)
mmcif_dir = Path(a.mmcif_dir)
_cache = {"file_id": None, "obj": None}


def parsed_for(file_id: str):
    if _cache["file_id"] != file_id:
        cif = mmcif_dir / f"{file_id}.cif"
        if not cif.is_file():
            _cache.update(file_id=file_id, obj=None)
        else:
            _cache.update(
                file_id=file_id,
                obj=mmcif_parsing.parse(file_id=file_id, mmcif_string=cif.read_text()).mmcif_object,
            )
    return _cache["obj"]

names, maps, qlens, flags = [], [], [], []
stat = dict(ok=0, ambiguous=0, not_subseq=0, no_mmcif=0, no_chain=0, no_npz=0,
            from_ridx=0, from_align=0)
failures = []

for n, i in enumerate(mine):
    chain = chains[i]
    import zlib
    npz = root / f"shard{zlib.crc32(chain.encode()) % 1000:04d}" / f"{chain}.npz"
    if not npz.is_file():
        stat["no_npz"] += 1
        continue
    file_id, chain_id = chain.rsplit("_", 1)
    mo = parsed_for(file_id)
    if mo is None and not (mmcif_dir / f"{file_id}.cif").is_file():
        stat["no_mmcif"] += 1
        failures.append((chain, "no mmcif"))
        continue
    if mo is None or chain_id not in mo.chain_to_seqres:
        stat["no_chain"] += 1
        failures.append((chain, "chain absent from mmcif"))
        continue
    query = mo.chain_to_seqres[chain_id]

    d = np.load(npz, allow_pickle=False)
    aat = d["aatype"].astype(int)
    npz_seq = "".join(rc.restypes[x] if x < len(rc.restypes) else "X" for x in aat)

    # ⭐⭐ PREFER `residue_index - 1` WHENEVER IT IS SEQUENCE-VALID, and fall back to the alignment
    # only when it is not. Reason: `ridx - 1` was measured sequence-exact on 1500/1500 randomly
    # sampled chains, while a deletion inside a run of identical residues makes the ALIGNMENT
    # ambiguous for a large minority (17-48% across shards) -- and there my leftmost pick is
    # arbitrary. `ridx - 1` carries independent structural information, so where it is consistent it
    # is the better estimate; switching to an arbitrary embedding there would INTRODUCE errors in
    # chains that were already placed correctly. The alignment exists for the cases `ridx - 1`
    # provably cannot explain (1eis_A: contiguous numbering across a real gap, 15/85 agreement).
    r = d["residue_index"].astype(int) - 1
    ridx_ok = (
        len(r) == len(npz_seq)
        and r.min() >= 0 and r.max() < len(query)
        and (np.diff(r) > 0).all()
        and all(_match(sc, query[qi]) for sc, qi in zip(npz_seq, r))
    )
    if ridx_ok:
        m, amb, src = r.astype(np.int32), False, "ridx"
    else:
        m = subsequence_map(npz_seq, query)
        if m is None:
            stat["not_subseq"] += 1
            failures.append((chain, "ridx-1 invalid AND npz seq is not a subsequence of the query",
                             f"npz {len(npz_seq)} vs query {len(query)}"))
            continue
        rm = rightmost_map(npz_seq, query)
        amb = rm is None or not np.array_equal(rm, m)
        src = "align"
    assert (np.diff(m) > 0).all() and 0 <= m.min() and m.max() < len(query)
    assert all(_match(sc, query[qi]) for sc, qi in zip(npz_seq, m)), chain

    stat["ambiguous"] += amb
    stat["ok"] += 1
    stat["from_ridx"] += (src == "ridx")
    stat["from_align"] += (src == "align")

    names.append(chain)
    maps.append(m)
    qlens.append(len(query))
    flags.append((2 if src == "align" else 0) | (1 if amb else 0))
    if (n + 1) % a.progress_every == 0:
        print(f"  {n+1}/{len(mine)}  ok={stat['ok']} amb={stat['ambiguous']} "
              f"bad={stat['not_subseq']}", flush=True)

# ragged maps -> flat storage plus offsets, so one npz holds every chain
lens = np.array([len(m) for m in maps], np.int32)
np.savez(
    a.out,
    chains=np.array(names),
    qmap=np.concatenate(maps) if maps else np.zeros(0, np.int32),
    qmap_len=lens,
    query_len=np.array(qlens, np.int32),
    # bit0 = ambiguous alignment, bit1 = placement came from the alignment not ridx-1
    ambiguous=np.array(flags, np.int8),
)
print(f"\nshard {a.shard}: wrote {len(names)} chains to {a.out}")
for k, v in stat.items():
    print(f"  {k:24s} {v}")
if failures:
    print(f"  first {min(8, len(failures))} failures:")
    for f in failures[:8]:
        print("   ", f)
sys.exit(0)
