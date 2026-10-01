"""Insert / delete residue stretches in a backbone structure, with a provenance sidecar.

Coordinates are (L, 4, 3) backbone atoms in N, CA, C, O order. Operations are given in NATIVE indices:
  ("ins", after, k)   insert k residues after native residue `after` (-1 = N-terminus, L-1 = C-terminus)
  ("del", start, end) delete native residues start..end inclusive
Interior insertion j=1..k sits at r_left + j/(k+1) * (r_right - r_left) for every backbone atom, where
left/right are the nearest SURVIVING residues. A terminal insertion extrapolates linearly outward from the
two nearest surviving residues: r_0 - j*(r_1 - r_0) (N end) / r_last + j*(r_last - r_prev) (C end).
Output residue numbering is contiguous 1..L'.
"""
import numpy as np

BB = ("N", "CA", "C", "O")


def validate(L, ops):
    dele = np.zeros(L, bool)
    for op in ops:
        if op[0] == "del":
            _, s, e = op
            assert 0 <= s <= e < L, f"bad deletion {op}"
            assert not dele[s:e + 1].any(), f"overlapping deletions at {op}"
            dele[s:e + 1] = True
    assert not dele.all(), "every residue deleted"
    sites = []
    for op in ops:
        if op[0] == "ins":
            _, after, k = op
            assert -1 <= after <= L - 1 and k >= 1, f"bad insertion {op}"
            assert after not in sites, f"two insertions at site {after}"
            sites.append(after)
            if 0 <= after < L - 1:
                assert not (dele[after] and dele[after + 1]), f"insertion site {after} inside a deleted stretch"
        elif op[0] != "del":
            raise ValueError(f"unknown op {op}")
    return dele


def edit(coords, ops):
    """Returns (new_coords (L',4,3), orig_idx (L',) with -1 for inserted, native_to_new (L,) with -1 for deleted)."""
    coords = np.asarray(coords, np.float64)
    L = len(coords)
    dele = validate(L, ops)
    ins = {op[1]: op[2] for op in ops if op[0] == "ins"}
    surv = np.flatnonzero(~dele)
    new_xyz, orig = [], []

    def emit_insertions(site, left, right):
        k = ins.get(site)
        if k is None:
            return
        if left is None:  # N-terminal: outward from the first two survivors
            r0, r1 = coords[surv[0]], coords[surv[1]]
            block = [r0 - j * (r1 - r0) for j in range(k, 0, -1)]
        elif right is None:  # C-terminal
            r0, r1 = coords[surv[-2]], coords[surv[-1]]
            block = [r1 + j * (r1 - r0) for j in range(1, k + 1)]
        else:
            rl, rr = coords[left], coords[right]
            block = [rl + j / (k + 1) * (rr - rl) for j in range(1, k + 1)]
        new_xyz.extend(block)
        orig.extend([-1] * k)

    emit_insertions(-1, None, surv[0])
    for n, i in enumerate(surv):
        new_xyz.append(coords[i])
        orig.append(int(i))
        right = surv[n + 1] if n + 1 < len(surv) else None
        # an insertion site is attached to the native residue it follows, even when that residue
        # was deleted (the boundary case allowed by validate): emit after the last survivor <= site
        sites_here = [s for s in ins if s >= i and (right is None or s < right) and s != -1]
        for s in sorted(sites_here):
            if s == L - 1 and right is None:
                emit_insertions(s, i, None)
            else:
                emit_insertions(s, i, right)
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
