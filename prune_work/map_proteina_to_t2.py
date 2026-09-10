"""Join proteina chain ids (label_asym_id) to T2 template ids (auth_asym_id).

⛔⛔ WHY. T2 keys every template on openfold's `<pdbid>_<auth_asym_id>`; proteina's processed .pt
keys on `<pdbid>_<label_asym_id>` -- graphein_utils.py deliberately overwrites the auth id with the
label id ("standardized chain IDs"). Joining the two by string therefore selects a DIFFERENT
POLYMER of the same entry whenever the two conventions disagree, silently and with a plausible
structure attached. Confirmed on 2hvy_B (74 vs 320), 2qr1_E (324 vs 91), 4plo_B (192 vs 412).

⛔ Unlike proteinfoundation/prediction_pipeline/cif_chain_mapping.py, this NEVER falls back to
returning the input id. A chain that cannot be mapped is reported with a reason and left out --
a silent passthrough is what produced the defect in the first place.
"""

from __future__ import annotations

import argparse
import csv
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

ATOM_KEYS = ("group_PDB", "label_asym_id", "auth_asym_id", "auth_seq_id")


def parse_atom_site(cif_path: Path):
    """-> (label -> {auth}), (label -> n_res), (auth -> n_res); None if unparseable."""
    cols, rows, header, in_loop = [], [], False, False
    with open(cif_path) as fh:
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
    l2a, ln, an = defaultdict(set), defaultdict(set), defaultdict(set)
    for r in rows:
        if len(r) <= mx or r[idx["group_PDB"]] != "ATOM":
            continue
        lab, aut, seq = r[idx["label_asym_id"]], r[idx["auth_asym_id"]], r[idx["auth_seq_id"]]
        l2a[lab].add(aut)
        ln[lab].add(seq)
        an[aut].add(seq)
    return l2a, {k: len(v) for k, v in ln.items()}, {k: len(v) for k, v in an.items()}


def npz_rel(chain: str) -> str:
    # crc32, matching generate_templates.py -- NOT builtin hash(), which is per-process randomised
    return f"shard{zlib.crc32(chain.encode()) % 1000:04d}/{chain}.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proteina-chains", required=True,
                    help="text file, one <pdbid>_<label_asym_id> per line")
    ap.add_argument("--t2-index", required=True, help="index_band.npz (its `chains` array)")
    ap.add_argument("--cif-dir", required=True)
    ap.add_argument("--out-csv", required=True)
    a = ap.parse_args()

    t2 = {str(c) for c in np.load(a.t2_index, allow_pickle=False)["chains"]}
    want = [l.strip() for l in open(a.proteina_chains) if l.strip()]
    by_pdb = defaultdict(list)
    for cid in want:
        if "_" not in cid:
            continue
        p, c = cid.rsplit("_", 1)
        by_pdb[p].append((cid, c))

    cif_dir = Path(a.cif_dir)
    rows, tally = [], defaultdict(int)
    cache: dict[str, object] = {}
    for pdb in sorted(by_pdb):
        if pdb not in cache:
            f = cif_dir / f"{pdb}.cif"
            cache[pdb] = parse_atom_site(f) if f.is_file() else None
        parsed = cache[pdb]
        for cid, lab in by_pdb[pdb]:
            if parsed is None:
                st, auth, nl, na = "cif_missing_or_unparseable", "", "", ""
            else:
                l2a, ln, an = parsed
                if lab not in l2a:
                    st, auth, nl, na = "label_absent_from_cif", "", "", ""
                elif len(l2a[lab]) != 1:
                    st, auth = "label_maps_to_multiple_auth", "|".join(sorted(l2a[lab]))
                    nl, na = ln[lab], ""
                else:
                    auth = next(iter(l2a[lab]))
                    nl, na = ln[lab], an.get(auth, "")
                    t2id = f"{pdb}_{auth}"
                    if t2id not in t2:
                        st = "no_t2_template_for_mapped_auth"
                    elif auth == lab:
                        st = "ok_ids_already_agree"
                    else:
                        st = "ok_REMAPPED"
            t2id = f"{pdb}_{auth}" if auth and "|" not in auth else ""
            tally[st] += 1
            rows.append({"proteina_id": cid, "pdb": pdb, "label": lab, "auth": auth,
                         "t2_id": t2id, "npz_rel": npz_rel(t2id) if t2id and st.startswith("ok") else "",
                         "n_res_label": nl, "n_res_auth": na, "status": st})

    with open(a.out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    tot = len(rows)
    print(f"{tot} proteina chains -> {a.out_csv}\n")
    print(f"{'status':>34} {'n':>8} {'%':>8}")
    for k in sorted(tally, key=lambda k: -tally[k]):
        print(f"{k:>34} {tally[k]:>8} {100 * tally[k] / tot:>7.2f}%")
    ok = tally["ok_ids_already_agree"] + tally["ok_REMAPPED"]
    print(f"\nusable after remap: {ok} = {100 * ok / tot:.2f}%"
          f"   (of which {tally['ok_REMAPPED']} would have been the WRONG chain under a string join)")
    # ⛔ a count is not evidence: name what is left out
    for k in sorted(tally):
        if not k.startswith("ok"):
            names = [r["proteina_id"] for r in rows if r["status"] == k][:8]
            print(f"  {k} ({tally[k]}): " + " ".join(names) + (" ..." if tally[k] > 8 else ""))


if __name__ == "__main__":
    main()
