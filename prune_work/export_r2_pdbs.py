"""Export a sample of ROUND-2 npz templates as PDBs, in the layout designability_pipeline.py wants.

Round 2 stores one npz per chain holding all 64 templates (round 1 chose npz over PDB because
`write_coords_to_pdb` costs ~868 ms per 128-template structure -- ~2 TB and 11.3 M files for the
full set). The designability pipeline consumes PDBs, so a SAMPLE is materialised here rather than
the whole pool.

Layout required by designability_pipeline.py::stage_inputs, which matches
`^(?P<model>[^-]+)-epoch(?P<epoch>\\d+)-.*-rewind(?P<rewind>\\d+|None)$` against `parent.parent.name`:

    <out>/<model>-epoch<epoch>-r2-rewind<R>/<chain>/sample_<i>.pdb

so each rung R becomes its own directory and that rung's `seeds_per_rung` samples become
sample_0..sample_{S-1} -- which is exactly the axis the check is about.

⛔ Uses openfold's own `protein.to_pdb` rather than a hand-rolled writer: MPNN reads N/CA/C/O and
the sequence, and getting the atom ordering subtly wrong would be invisible until the NLLs came out
strange.
"""

from __future__ import annotations

import argparse
import zlib
from pathlib import Path

import numpy as np

from openfold.np import protein


def npz_path(root: Path, chain: str) -> Path:
    # ⛔ crc32, not builtin hash(): stable across processes, same rule as generation
    return root / f"shard{zlib.crc32(chain.encode()) % 1000:04d}" / f"{chain}.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--templates-root", required=True)
    ap.add_argument("--index", required=True, help="round-2 index_all.npz, for the chain list")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-chains", type=int, default=30)
    ap.add_argument("--model", default="cc89")
    ap.add_argument("--epoch", default="415")
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--max-length", type=int, default=400,
                    help="skip very long chains: PDB writing and MPNN both scale with L and this "
                         "is a sample, not a census")
    ap.add_argument("--natives-src", default=None,
                    help="sharded natives_train root. ⛔ designability_pipeline.py's stage_inputs "
                         "globs natives_dir FLAT ('*.pdb'), but natives_train is sharded into "
                         "shardNNNN/, so a flat glob there finds NOTHING and the run would "
                         "silently lose its native reference ceiling. Given this, the sampled "
                         "chains' natives are symlinked into --natives-out as a flat dir.")
    ap.add_argument("--natives-out", default=None)
    a = ap.parse_args()
    assert bool(a.natives_src) == bool(a.natives_out), \
        "--natives-src and --natives-out go together"

    z = np.load(a.index, allow_pickle=False)
    chains = [str(c) for c in z["chains"]]
    lengths = z["length"]
    ok = [c for c, L in zip(chains, lengths) if 30 <= int(L) <= a.max_length]
    rng = np.random.default_rng(a.sample_seed)
    pick = [ok[i] for i in rng.choice(len(ok), size=min(a.n_chains, len(ok)), replace=False)]
    print(f"{len(chains)} chains in index, {len(ok)} within length bounds, sampling {len(pick)}",
          flush=True)

    root, out = Path(a.templates_root), Path(a.out)
    written = 0
    skipped = []
    for chain in sorted(pick):
        p = npz_path(root, chain)
        if not p.is_file():
            skipped.append(f"{chain}:missing_npz")
            continue
        d = np.load(p, allow_pickle=False)
        atom_mask = d["atom_mask"]
        L = atom_mask.shape[0]
        n = d["coords"].shape[0]
        full = np.zeros((n, L, 37, 3), np.float32)
        full[:, atom_mask] = d["coords"]
        aatype = d["aatype"].astype(np.int64)
        residue_index = d["residue_index"].astype(np.int64)
        rw = d["rewind_steps"].astype(int)

        # a rung's samples are its seeds; number them 0..S-1 in file order
        for r in sorted(set(rw.tolist()), reverse=True):
            idxs = np.flatnonzero(rw == r)
            dst = out / f"{a.model}-epoch{a.epoch}-r2-rewind{r}" / chain
            dst.mkdir(parents=True, exist_ok=True)
            for s, i in enumerate(idxs):
                prot = protein.Protein(
                    atom_positions=full[i],
                    aatype=aatype,
                    atom_mask=atom_mask.astype(np.float32),
                    residue_index=residue_index,
                    b_factors=np.zeros((L, 37), np.float32),
                    chain_index=np.zeros(L, np.int64),
                )
                (dst / f"sample_{s}.pdb").write_text(protein.to_pdb(prot))
                written += 1
        print(f"  {chain}: L={L} {len(set(rw.tolist()))} rungs -> {n} pdb", flush=True)

    print(f"\nwrote {written} pdb under {out}")

    if a.natives_src:
        src, nout = Path(a.natives_src), Path(a.natives_out)
        nout.mkdir(parents=True, exist_ok=True)
        staged, missing = 0, []
        for chain in sorted(pick):
            hits = list(src.rglob(f"{chain}.pdb"))
            if not hits:
                missing.append(chain)
                continue
            dst = nout / f"{chain}.pdb"
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            dst.symlink_to(hits[0].resolve())
            staged += 1
        print(f"staged {staged} native pdb (flat) under {nout}")
        # the natives are the reference ceiling; losing them silently is the failure this guards
        if missing:
            print(f"⛔ {len(missing)} sampled chains have NO native pdb: " + " ".join(missing))
        assert staged, "no natives staged -- the run would have no reference ceiling"

    # ⛔ a count is not evidence: name every skip
    if skipped:
        print(f"{len(skipped)} chains skipped: " + " ".join(skipped))


if __name__ == "__main__":
    main()
