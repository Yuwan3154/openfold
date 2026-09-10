"""Build the label_asym_id <-> auth_asym_id alias index for the T2 template trees.

⛔⛔ WHY. Every T2 template is keyed `<pdbid>_<auth_asym_id>` (openfold's slim_struct_train.list).
Consumers keyed on `label_asym_id` -- proteina's processed .pt is, because graphein_utils.py
overwrites the auth id with the label id -- silently resolve the same string to a DIFFERENT POLYMER
of the same entry. Measured: 6.17% of such joins return the wrong chain and 11.6% of those have
IDENTICAL residue counts, so no length check can find them. This file is the join key that makes
the two conventions interoperable.

Emits BOTH directions plus the residue counts, so a consumer can assert the length it expected
instead of trusting the mapping blindly.

⛔ Writes incrementally: a kill loses the tail, not the run.
⛔ Every chain that cannot be mapped is recorded WITH A REASON. No silent passthrough -- returning
the input id unchanged on failure is exactly the bug this exists to prevent.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ATOM_KEYS = ("group_PDB", "label_asym_id", "auth_asym_id", "auth_seq_id")
CIF_DIR = None


def parse_atom_site(path: Path):
    cols, rows, header, in_loop = [], [], False, False
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if s.startswith("_atom_site."):
                in_loop = header = True
                cols.append(s.split(".", 1)[1])
                continue
            if header and not s.startswith("_atom_site."):
                header = False
            if in_loop and not header:
                if s.startswith(("#", "_", "loop_")):
                    if rows:
                        break
                    continue
                if s:
                    rows.append(s.split())
    idx = {c: i for i, c in enumerate(cols)}
    if any(k not in idx for k in ATOM_KEYS):
        return None
    mx = max(idx.values())
    a2l, an, ln = defaultdict(set), defaultdict(set), defaultdict(set)
    for r in rows:
        if len(r) <= mx or r[idx["group_PDB"]] != "ATOM":
            continue
        a, l, sq = r[idx["auth_asym_id"]], r[idx["label_asym_id"]], r[idx["auth_seq_id"]]
        a2l[a].add(l)
        an[a].add(sq)
        ln[l].add(sq)
    return a2l, {k: len(v) for k, v in an.items()}, {k: len(v) for k, v in ln.items()}


def one_entry(item):
    pdb, auths = item
    f = Path(CIF_DIR) / f"{pdb}.cif"
    if not f.is_file():
        return [(f"{pdb}_{a}", pdb, a, "", "", "", "cif_missing") for a in auths]
    parsed = parse_atom_site(f)
    if parsed is None:
        return [(f"{pdb}_{a}", pdb, a, "", "", "", "cif_unparseable") for a in auths]
    a2l, an, ln = parsed
    out = []
    for a in auths:
        if a not in a2l:
            out.append((f"{pdb}_{a}", pdb, a, "", "", "", "auth_absent_from_cif"))
        elif len(a2l[a]) != 1:
            out.append((f"{pdb}_{a}", pdb, a, "|".join(sorted(a2l[a])), an[a], "",
                        "auth_maps_to_multiple_label"))
        else:
            lab = next(iter(a2l[a]))
            out.append((f"{pdb}_{a}", pdb, a, lab, an[a], ln.get(lab, ""),
                        "ok_same_letter" if lab == a else "ok_differs"))
    return out


def main():
    global CIF_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--t2-index", required=True, help="index_band.npz / index_merged.npz")
    ap.add_argument("--cif-dir", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-npz", required=True)
    ap.add_argument("--nproc", type=int, default=16)
    a = ap.parse_args()
    CIF_DIR = a.cif_dir

    chains = [str(c) for c in np.load(a.t2_index, allow_pickle=False)["chains"]]
    by_pdb = defaultdict(list)
    for cid in chains:
        p, c = cid.rsplit("_", 1)
        by_pdb[p].append(c)
    items = sorted(by_pdb.items())
    print(f"{len(chains)} T2 chains over {len(items)} entries; nproc={a.nproc}", flush=True)

    t0 = time.time()
    tally = defaultdict(int)
    rows = []
    fields = ["t2_auth_id", "pdb", "auth", "label", "n_res_auth", "n_res_label", "status"]
    with open(a.out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(fields)
        with Pool(a.nproc) as pool:
            for k, res in enumerate(pool.imap_unordered(one_entry, items, chunksize=25), 1):
                for r in res:
                    w.writerow(r)
                    rows.append(r)
                    tally[r[-1]] += 1
                if k % 5000 == 0:
                    fh.flush()
                    el = time.time() - t0
                    print(f"  {k}/{len(items)} entries  {el:.0f}s  eta {el/k*(len(items)-k):.0f}s",
                          flush=True)

    ok = [r for r in rows if r[-1].startswith("ok")]
    # ⛔ the inverse must be a FUNCTION; a label id colliding across auth chains would silently
    #    reintroduce the ambiguity this file exists to remove.
    inv = defaultdict(list)
    for r in ok:
        inv[f"{r[1]}_{r[3]}"].append(r[0])
    collisions = {k: v for k, v in inv.items() if len(v) > 1}

    np.savez(
        a.out_npz,
        t2_auth_id=np.array([r[0] for r in ok]),
        label_id=np.array([f"{r[1]}_{r[3]}" for r in ok]),
        n_res_auth=np.array([int(r[4]) for r in ok], np.int32),
        n_res_label=np.array([int(r[5]) for r in ok], np.int32),
        differs=np.array([r[-1] == "ok_differs" for r in ok], bool),
    )

    tot = len(rows)
    print(f"\n{tot} chains -> {a.out_csv} / {a.out_npz}\n")
    print(f"{'status':>30} {'n':>8} {'%':>8}")
    for k in sorted(tally, key=lambda k: -tally[k]):
        print(f"{k:>30} {tally[k]:>8} {100*tally[k]/tot:>7.2f}%")
    print(f"\nmapped: {len(ok)} = {100*len(ok)/tot:.2f}%   "
          f"of which the label id DIFFERS from the auth id: {tally['ok_differs']} "
          f"({100*tally['ok_differs']/tot:.2f}%) <- these are the silent-wrong-chain cases")
    print(f"label_id collisions (must be 0): {len(collisions)}")
    for k, v in list(collisions.items())[:5]:
        print(f"   {k} <- {v}")
    # ⛔ a count is not evidence: name what failed
    for k in sorted(tally):
        if not k.startswith("ok"):
            names = [r[0] for r in rows if r[-1] == k][:10]
            print(f"  {k} ({tally[k]}): " + " ".join(names) + (" ..." if tally[k] > 10 else ""))


if __name__ == "__main__":
    sys.exit(main())
