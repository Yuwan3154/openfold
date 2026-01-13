#!/usr/bin/env python3
"""
Build a PDB30 dataset from an existing HHsuite PDB70 database.

This script is designed for the HHsuite PDB70 layout used by AlphaFold/OpenFold:
  - pdb70_a3m.ffindex / pdb70_a3m.ffdata
  - pdb70_hhm.ffindex / pdb70_hhm.ffdata
  - pdb70_cs219.ffindex / pdb70_cs219.ffdata
  - pdb70_clu.tsv (rep70 -> chain mapping)

Outputs (in --out_dir):
  - pdb70_reps.tsv: rep70_id<TAB>sequence  (query sequence extracted from each A3M entry)
  - pdb70_reps.fasta: FASTA version of the same sequences (for MMseqs2)
  - pdb30_rep_to_pdb70rep.tsv: rep30_id<TAB>rep70_member (MMseqs2 cluster mapping at 30% id)
  - pdb30_{a3m,hhm,cs219}.ffindex + pdb30_{a3m,hhm,cs219}.ffdata (symlinked ffdata)
  - pdb30_200513.tsv: rep30_id<TAB>sequence (requested training TSV)
  - pdb30_clu.tsv: rep30_id<TAB>pdb_chain (rep30 expanded to chains via pdb70_clu.tsv)

No try/except is used by design (per workspace rules).
"""

import argparse
import mmap
import os
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


def read_ffindex_entries(ffindex_path: Path) -> List[Tuple[str, int, int]]:
    entries: List[Tuple[str, int, int]] = []
    with open(ffindex_path, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            name, offset, length = line.split("\t")
            entries.append((name, int(offset), int(length)))
    return entries


def extract_query_sequence_from_a3m_entry(a3m_data: mmap.mmap, offset: int, length: int, entry_id: str) -> str:
    """
    Extract the query sequence (FASTA record whose header starts with >{entry_id})
    from an A3M entry in an ffdata file.

    The ffindex length includes a trailing '\\0' delimiter byte, so we exclude it.
    """
    entry_end = offset + length - 1  # exclude trailing null byte
    entry_id_bytes = entry_id.encode("ascii")

    # Many HHsuite A3Ms contain >ss_dssp / >ss_pred / >ss_conf before the actual query.
    # Find the header line that begins with >{entry_id}.
    header_start = -1
    direct_prefix = b">" + entry_id_bytes
    if offset + len(direct_prefix) <= entry_end and a3m_data[offset : offset + len(direct_prefix)] == direct_prefix:
        header_start = offset
    else:
        header_start = a3m_data.find(b"\n>" + entry_id_bytes, offset, entry_end)
        if header_start != -1:
            header_start += 1  # move from '\n' to '>'

    if header_start == -1:
        return ""

    header_end = a3m_data.find(b"\n", header_start, entry_end)
    if header_end == -1:
        return ""

    next_header = a3m_data.find(b"\n>", header_end + 1, entry_end)
    if next_header == -1:
        seq_block = a3m_data[header_end + 1 : entry_end]
    else:
        seq_block = a3m_data[header_end + 1 : next_header]

    seq_bytes = (
        seq_block.replace(b"\n", b"")
        .replace(b"\r", b"")
        .replace(b" ", b"")
        .replace(b"\t", b"")
        .replace(b"-", b"")
        .replace(b".", b"")
    )
    return seq_bytes.decode("ascii").upper()


def symlink_force(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(str(src), str(dst))


def write_pdb70_reps(pdb70_dir: Path, out_dir: Path, overwrite: bool) -> Tuple[Path, Path]:
    a3m_ffindex = pdb70_dir / "pdb70_a3m.ffindex"
    a3m_ffdata = pdb70_dir / "pdb70_a3m.ffdata"

    out_tsv = out_dir / "pdb70_reps.tsv"
    out_fasta = out_dir / "pdb70_reps.fasta"

    if out_tsv.exists() and out_fasta.exists() and not overwrite:
        return out_tsv, out_fasta

    entries = read_ffindex_entries(a3m_ffindex)

    with open(a3m_ffdata, "rb") as fh:
        a3m_data = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        with open(out_tsv, "w") as tsv_fh, open(out_fasta, "w") as fasta_fh:
            empty = 0
            for name, offset, length in entries:
                seq = extract_query_sequence_from_a3m_entry(a3m_data, offset, length, entry_id=name)
                if not seq:
                    empty += 1
                    continue
                tsv_fh.write(f"{name}\t{seq}\n")
                fasta_fh.write(f">{name}\n{seq}\n")

        a3m_data.close()

    if empty:
        print(f"Warning: {empty} entries had empty query sequences and were skipped.")

    return out_tsv, out_fasta


def run_mmseqs_clustering(
    fasta_path: Path,
    out_dir: Path,
    mmseqs_bin: str,
    seq_id: float,
    coverage: float,
    threads: int,
    overwrite: bool,
) -> Path:
    out_tsv = out_dir / "pdb30_rep_to_pdb70rep.tsv"
    if out_tsv.exists() and not overwrite:
        return out_tsv

    seqdb = out_dir / "pdb70_reps_mmseqs_db"
    clu = out_dir / "pdb30_mmseqs_clu"
    tmp_dir = out_dir / "mmseqs_tmp"

    if tmp_dir.exists() and overwrite:
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    createdb_cmd = [mmseqs_bin, "createdb", str(fasta_path), str(seqdb)]
    cluster_cmd = [
        mmseqs_bin,
        "cluster",
        str(seqdb),
        str(clu),
        str(tmp_dir),
        "--min-seq-id",
        str(seq_id),
        "-c",
        str(coverage),
        "-s",
        "8",
        "--max-seqs",
        "1000",
        "--cluster-mode",
        "1",
        "--threads",
        str(threads),
    ]
    createtsv_cmd = [mmseqs_bin, "createtsv", str(seqdb), str(seqdb), str(clu), str(out_tsv)]

    subprocess.run(createdb_cmd, check=True)
    subprocess.run(cluster_cmd, check=True)
    subprocess.run(createtsv_cmd, check=True)

    return out_tsv


def load_rep_ids(rep_to_member_tsv: Path) -> Set[str]:
    reps: Set[str] = set()
    with open(rep_to_member_tsv, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            rep, _member = line.split("\t", 1)
            reps.add(rep)
    return reps


def filter_ffindex(in_ffindex: Path, out_ffindex: Path, keep_ids: Set[str], overwrite: bool) -> int:
    if out_ffindex.exists() and not overwrite:
        with open(out_ffindex, "r") as f:
            return sum(1 for _ in f)

    kept = 0
    with open(in_ffindex, "r") as in_f, open(out_ffindex, "w") as out_f:
        for line in in_f:
            if not line.strip():
                continue
            name = line.split("\t", 1)[0]
            if name in keep_ids:
                out_f.write(line)
                kept += 1
    return kept


def build_hhsuite_pdb30_subset(pdb70_dir: Path, out_dir: Path, rep_ids: Set[str], overwrite: bool) -> None:
    for suffix in ["a3m", "hhm", "cs219"]:
        src_ffdata = pdb70_dir / f"pdb70_{suffix}.ffdata"
        src_ffindex = pdb70_dir / f"pdb70_{suffix}.ffindex"
        dst_ffdata = out_dir / f"pdb30_{suffix}.ffdata"
        dst_ffindex = out_dir / f"pdb30_{suffix}.ffindex"

        if overwrite or not dst_ffdata.exists():
            symlink_force(src_ffdata, dst_ffdata)

        kept = filter_ffindex(src_ffindex, dst_ffindex, rep_ids, overwrite=overwrite)
        print(f"pdb30_{suffix}: kept {kept} entries")


def write_pdb30_sequences_tsv(pdb70_reps_tsv: Path, rep_ids: Set[str], out_tsv: Path, overwrite: bool) -> int:
    if out_tsv.exists() and not overwrite:
        with open(out_tsv, "r") as f:
            return sum(1 for _ in f)

    kept = 0
    with open(pdb70_reps_tsv, "r") as in_f, open(out_tsv, "w") as out_f:
        for line in in_f:
            line = line.rstrip("\n")
            if not line:
                continue
            seq_id, seq = line.split("\t", 1)
            if seq_id in rep_ids:
                out_f.write(f"{seq_id}\t{seq}\n")
                kept += 1
    return kept


def write_pdb30_clu(
    rep_to_member_tsv: Path,
    pdb70_clu_tsv: Path,
    out_clu_tsv: Path,
    overwrite: bool,
) -> int:
    if out_clu_tsv.exists() and not overwrite:
        with open(out_clu_tsv, "r") as f:
            return sum(1 for _ in f)

    rep70_to_chains: Dict[str, List[str]] = defaultdict(list)
    with open(pdb70_clu_tsv, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            rep70, chain = line.split("\t")
            rep70_to_chains[rep70].append(chain)

    out_lines = 0
    with open(rep_to_member_tsv, "r") as in_f, open(out_clu_tsv, "w") as out_f:
        for line in in_f:
            line = line.rstrip("\n")
            if not line:
                continue
            rep30, rep70_member = line.split("\t")
            for chain in rep70_to_chains.get(rep70_member, []):
                out_f.write(f"{rep30}\t{chain}\n")
                out_lines += 1
    return out_lines


def parse_steps(steps_arg: str) -> List[str]:
    steps = [s.strip() for s in steps_arg.split(",") if s.strip()]
    if not steps:
        return ["all"]
    return steps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pdb70_dir",
        type=Path,
        default=Path("/home/jupyter-chenxi/data/pdb70_200513"),
        help="Path to the PDB70 HHsuite database directory",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("/home/jupyter-chenxi/data/pdb30_200513"),
        help="Output directory for the derived PDB30 dataset",
    )
    parser.add_argument("--seq_id", type=float, default=0.30, help="MMseqs2 min sequence identity threshold")
    parser.add_argument("--coverage", type=float, default=0.80, help="MMseqs2 coverage threshold (-c)")
    parser.add_argument(
        "--mmseqs",
        type=str,
        default=(shutil.which("mmseqs") or "/home/jupyter-chenxi/mmseqs/bin/mmseqs"),
        help="Path to mmseqs binary",
    )
    parser.add_argument("--threads", type=int, default=(os.cpu_count() or 8), help="MMseqs2 threads")
    parser.add_argument(
        "--steps",
        type=str,
        default="all",
        help="Comma-separated steps: extract,cluster,subset,tsv,clu or 'all'",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")

    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    steps = parse_steps(args.steps)
    if "all" in steps:
        steps = ["extract", "cluster", "subset", "tsv", "clu"]

    pdb70_reps_tsv = out_dir / "pdb70_reps.tsv"
    pdb70_reps_fasta = out_dir / "pdb70_reps.fasta"
    rep_to_member_tsv = out_dir / "pdb30_rep_to_pdb70rep.tsv"
    pdb30_tsv = out_dir / "pdb30_200513.tsv"
    pdb30_clu_tsv = out_dir / "pdb30_clu.tsv"

    if "extract" in steps:
        pdb70_reps_tsv, pdb70_reps_fasta = write_pdb70_reps(args.pdb70_dir, out_dir, overwrite=args.overwrite)
        print(f"Wrote: {pdb70_reps_tsv}")
        print(f"Wrote: {pdb70_reps_fasta}")

    if "cluster" in steps:
        rep_to_member_tsv = run_mmseqs_clustering(
            fasta_path=pdb70_reps_fasta,
            out_dir=out_dir,
            mmseqs_bin=args.mmseqs,
            seq_id=args.seq_id,
            coverage=args.coverage,
            threads=args.threads,
            overwrite=args.overwrite,
        )
        print(f"Wrote: {rep_to_member_tsv}")

    rep_ids: Set[str] = set()
    if any(step in steps for step in ["subset", "tsv", "clu"]):
        rep_ids = load_rep_ids(rep_to_member_tsv)
        print(f"Loaded {len(rep_ids)} PDB30 representative IDs")

    if "subset" in steps:
        build_hhsuite_pdb30_subset(args.pdb70_dir, out_dir, rep_ids, overwrite=args.overwrite)

    if "tsv" in steps:
        n = write_pdb30_sequences_tsv(pdb70_reps_tsv, rep_ids, pdb30_tsv, overwrite=args.overwrite)
        print(f"Wrote: {pdb30_tsv} ({n} sequences)")

    if "clu" in steps:
        pdb70_clu = args.pdb70_dir / "pdb70_clu.tsv"
        n = write_pdb30_clu(rep_to_member_tsv, pdb70_clu, pdb30_clu_tsv, overwrite=args.overwrite)
        print(f"Wrote: {pdb30_clu_tsv} ({n} lines)")


if __name__ == "__main__":
    main()


