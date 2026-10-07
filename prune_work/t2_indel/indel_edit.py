"""Insert / delete residue stretches in a backbone structure, with a provenance sidecar.

Coordinates are (L, 4, 3) backbone atoms in N, CA, C, O order. Operations are given in NATIVE indices:
  ("ins", after, k)   insert k residues after native residue `after` (-1 = N-terminus, L-1 = C-terminus);
                      the effective spot is relative to the SURVIVING residues, so a site next to a
                      deleted stretch lands at that stretch's seam
  ("del", start, end) delete native residues start..end inclusive
Interior insertion j=1..k sits at r_left + j/(k+1) * (r_right - r_left) for every backbone atom, where
left/right are the nearest SURVIVING residues. A terminal insertion extends the chain outward (only the CA direction is extrapolated; the other atoms copy the end residue) from the
two nearest surviving residues: the end survivor is RIGIDLY translated by j * CA_STEP along the unit CA(r_1) - CA(r_0) direction
(N end: outward from r_0; C end: outward from r_last), so inserted residues copy its internal geometry and the CA-CA step is exactly
CA_STEP (user 10-07: fixed 3.8 A; compare_terminal_methods.py showed per-atom stepping, the earlier rule, explodes the inserted bonds).
Output residue numbering is contiguous 1..L'.
"""
import numpy as np

BB = ("N", "CA", "C", "O")
CA_STEP = 3.8


def deleted_mask(L, ops):
    dele = np.zeros(L, bool)
    for op in ops:
        if op[0] == "del":
            _, s, e = op
            assert 0 <= s <= e < L, f"bad deletion {op}"
            assert not dele[s:e + 1].any(), f"overlapping deletions at {op}"
            dele[s:e + 1] = True
    assert not dele.all(), "every residue deleted"
    return dele


def insertion_locations(L, ops):
    """{loc: k}. loc = number of SURVIVING residues at or before the site, so loc 0 = before the first
    survivor (N end), loc S = after the last survivor (C end). Two sites with the same loc are the same
    physical spot (e.g. either side of a deleted stretch) and are rejected as a double insertion."""
    dele = deleted_mask(L, ops)
    kept_upto = np.cumsum(~dele)
    locs = {}
    for op in ops:
        if op[0] == "ins":
            _, after, k = op
            assert -1 <= after <= L - 1 and k >= 1, f"bad insertion {op}"
            loc = 0 if after == -1 else int(kept_upto[after])
            assert loc not in locs, f"two insertions at the same location (site {after})"
            locs[loc] = k
        elif op[0] != "del":
            raise ValueError(f"unknown op {op}")
    return locs


def validate(L, ops):
    insertion_locations(L, ops)


def edit(coords, ops):
    """Returns (new_coords (L',4,3), orig_idx (L',) with -1 for inserted, native_to_new (L,) with -1 for deleted)."""
    coords = np.asarray(coords, np.float64)
    L = len(coords)
    dele = deleted_mask(L, ops)
    locs = insertion_locations(L, ops)
    surv = np.flatnonzero(~dele)
    S = len(surv)
    new_xyz, orig = [], []

    def step(r0, r1):
        n = np.linalg.norm(r1[1] - r0[1])
        assert n > 0, "coincident CA atoms in the two terminal survivors"
        return (r1[1] - r0[1]) * CA_STEP / n

    def block(loc):
        k = locs[loc]
        if loc in (0, S):
            assert S >= 2, "terminal extrapolation needs at least 2 surviving residues"
        if loc == 0:          # outward from the first two survivors; nearest-to-chain is j=1
            r0, r1 = coords[surv[0]], coords[surv[1]]
            xyz = [r0 - j * step(r0, r1) for j in range(k, 0, -1)]  # step is (3,): translates all atoms
        elif loc == S:
            r0, r1 = coords[surv[-2]], coords[surv[-1]]
            xyz = [r1 + j * step(r0, r1) for j in range(1, k + 1)]
        else:
            rl, rr = coords[surv[loc - 1]], coords[surv[loc]]
            xyz = [rl + j / (k + 1) * (rr - rl) for j in range(1, k + 1)]
        new_xyz.extend(xyz)
        orig.extend([-1] * k)

    if 0 in locs:
        block(0)
    for n, i in enumerate(surv):
        new_xyz.append(coords[i])
        orig.append(int(i))
        if n + 1 in locs:
            block(n + 1)
    orig = np.array(orig)
    native_to_new = np.full(L, -1)
    for new_i, o in enumerate(orig):
        if o >= 0:
            native_to_new[o] = new_i
    return np.array(new_xyz), orig, native_to_new


def write_pdb(path, coords, resnames):
    lines = []
    n = 0
    for i, (res, rn) in enumerate(zip(coords, resnames), start=1):
        for name, xyz in zip(BB, res):
            n += 1
            lines.append(f"ATOM  {n:5d} {name:<4s} {rn:>3s} A{i:4d}    "
                         f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00           {name[0]:>2s}")
    lines.append("END")
    open(path, "w").write("\n".join(lines) + "\n")
