"""Gate: the label<->auth alias must COVER the template index it ships beside.

⛔ The alias is built from one index (round 1's index_band.npz). Shipping it beside a DIFFERENT
index (the round-1+2 merge) without checking set equality is how a stale mapping silently becomes
partial coverage -- a consumer would then find no template for the uncovered chains and record it
as "missing", not as "the alias is out of date".
⛔ Assert the SAMPLE SIZE, never just "0 bad": a vacuous check prints PASS.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alias", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--max-report", type=int, default=15)
    a = ap.parse_args()

    al = np.load(a.alias, allow_pickle=False)
    auth = [str(x) for x in al["t2_auth_id"]]
    label = [str(x) for x in al["label_id"]]
    idx = [str(c) for c in np.load(a.index, allow_pickle=False)["chains"]]

    print(f"alias rows      : {len(auth)}")
    print(f"index chains    : {len(idx)}")
    if not auth or not idx:
        print("FAIL: one side is empty -- the check would be vacuous")
        return 1

    sa, si = set(auth), set(idx)
    missing = sorted(si - sa)
    extra = sorted(sa - si)
    print(f"index chains WITHOUT an alias row : {len(missing)} of {len(si)}")
    print(f"alias rows not in the index       : {len(extra)} of {len(sa)}")

    # the inverse must stay a function, or a label id resolves to two templates again
    seen, collide = {}, []
    for aid, lid in zip(auth, label):
        if lid in seen:
            collide.append((lid, seen[lid], aid))
        seen[lid] = aid
    print(f"label_id collisions               : {len(collide)} (must be 0)")

    n_diff = int(al["differs"].sum())
    print(f"rows where label != auth          : {n_diff}"
          f" ({100 * n_diff / len(auth):.2f}%) -- the silent-wrong-chain population")

    # the mapping is between two NAMES for one polymer, so the counts must agree exactly
    bad_len = int((al["n_res_auth"] != al["n_res_label"]).sum())
    print(f"rows whose two names disagree on length: {bad_len} of {len(auth)} (must be 0)")

    ok = not missing and not collide and bad_len == 0
    for lab, xs in (("missing", missing), ("extra", extra)):
        if xs:
            print(f"  {lab}: " + " ".join(xs[: a.max_report])
                  + (" ..." if len(xs) > a.max_report else ""))
    for lid, x, y in collide[: a.max_report]:
        print(f"  collision {lid} <- {x} and {y}")
    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
