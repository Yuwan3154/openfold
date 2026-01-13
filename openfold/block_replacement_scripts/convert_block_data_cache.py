#!/usr/bin/env python3
"""
Convert an existing block data cache between on-disk formats.

Why this exists:
  - zipnn-based `.safetensors.znn` currently shows unbounded RSS growth during repeated
    decompression in long training runs (likely inside ZipNN.decompress).
  - Converting the cache to plain `.safetensors` avoids ZipNN decompression at train time.

Notes:
  - This script performs file-by-file conversion and supports multiprocessing.
  - To mitigate ZipNN decompression RSS growth *during conversion*, use a finite
    `--maxtasksperchild` so worker processes are periodically restarted.
"""

from __future__ import annotations

import argparse
import os
from multiprocessing import Pool
from pathlib import Path
from typing import Iterable, Optional, Tuple

from openfold.block_replacement_scripts.block_data_io import load_block_sample, save_block_sample


_G_SRC_CACHE_DIR: Optional[Path] = None
_G_DST_CACHE_DIR: Optional[Path] = None
_G_SRC_EXT: str = "safetensors.znn"
_G_DST_EXT: str = "safetensors"
_G_SAVE_DTYPE: str = "bf16"
_G_QUANTIZATION: str = "none"
_G_DELETE_SRC: bool = False
_G_OVERWRITE: bool = False


def _init_worker(
    src_cache_dir: str,
    dst_cache_dir: Optional[str],
    src_ext: str,
    dst_ext: str,
    save_dtype: str,
    quantization: str,
    delete_src: bool,
    overwrite: bool,
) -> None:
    global _G_SRC_CACHE_DIR
    global _G_DST_CACHE_DIR
    global _G_SRC_EXT
    global _G_DST_EXT
    global _G_SAVE_DTYPE
    global _G_QUANTIZATION
    global _G_DELETE_SRC
    global _G_OVERWRITE

    _G_SRC_CACHE_DIR = Path(src_cache_dir)
    _G_DST_CACHE_DIR = Path(dst_cache_dir) if dst_cache_dir is not None else None
    _G_SRC_EXT = src_ext
    _G_DST_EXT = dst_ext
    _G_SAVE_DTYPE = save_dtype
    _G_QUANTIZATION = quantization
    _G_DELETE_SRC = delete_src
    _G_OVERWRITE = overwrite


def _src_to_dst(src: Path) -> Path:
    if _G_SRC_CACHE_DIR is None:
        raise RuntimeError("worker not initialized (_G_SRC_CACHE_DIR is None)")

    src_suffix = f".{_G_SRC_EXT}"
    if not src.name.endswith(src_suffix):
        raise ValueError(f"Unexpected src filename (expected suffix {src_suffix}): {src}")

    base = src.name[: -len(src_suffix)]
    dst_name = f"{base}.{_G_DST_EXT}"

    rel = src.relative_to(_G_SRC_CACHE_DIR)
    if _G_DST_CACHE_DIR is None:
        return src.with_name(dst_name)
    return _G_DST_CACHE_DIR / rel.parent / dst_name


def _atomic_save(sample: dict, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_tag = f".tmp.{os.getpid()}"
    suffix = f".{_G_DST_EXT}"
    base_name = dst.name[: -len(suffix)] if dst.name.endswith(suffix) else dst.name
    tmp_path = dst.with_name(base_name + tmp_tag + suffix)
    save_block_sample(
        sample,
        tmp_path,
        save_dtype=_G_SAVE_DTYPE,  # type: ignore[arg-type]
        quantization=_G_QUANTIZATION,  # type: ignore[arg-type]
    )
    os.replace(tmp_path, dst)


def _convert_one(src_str: str) -> int:
    if _G_SRC_CACHE_DIR is None:
        raise RuntimeError("worker not initialized (_G_SRC_CACHE_DIR is None)")

    src = Path(src_str)
    dst = _src_to_dst(src)

    if dst.exists() and not _G_OVERWRITE:
        return 0

    sample = load_block_sample(src, map_location="cpu")
    _atomic_save(sample, dst)

    if _G_DELETE_SRC and dst.exists():
        src.unlink()

    return 1


def _iter_src_files(src_cache_dir: Path, src_ext: str) -> Iterable[Path]:
    # Cache layout is expected as: block_{XX}/{seq_id}.{ext}
    pattern = f"*.{src_ext}"
    for block_dir in sorted(src_cache_dir.glob("block_*")):
        if not block_dir.is_dir():
            continue
        yield from sorted(block_dir.glob(pattern))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src_cache_dir", required=True, type=str)
    parser.add_argument("--dst_cache_dir", default=None, type=str)
    parser.add_argument("--src_ext", default="safetensors.znn", type=str)
    parser.add_argument("--dst_ext", default="safetensors", type=str)
    parser.add_argument("--save_dtype", default="bf16", type=str)
    parser.add_argument("--quantization", default="none", type=str)
    parser.add_argument("--delete_src", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--processes", default=8, type=int)
    parser.add_argument("--maxtasksperchild", default=20, type=int)
    parser.add_argument("--limit", default=0, type=int, help="Convert at most N files (0 = no limit)")
    args = parser.parse_args()

    src_cache_dir = Path(args.src_cache_dir)
    dst_cache_dir = Path(args.dst_cache_dir) if args.dst_cache_dir is not None else None

    src_files = list(_iter_src_files(src_cache_dir, args.src_ext))
    if args.limit and args.limit > 0:
        src_files = src_files[: args.limit]

    if len(src_files) == 0:
        print("No source files found.", flush=True)
        return

    init_args: Tuple = (
        str(src_cache_dir),
        str(dst_cache_dir) if dst_cache_dir is not None else None,
        args.src_ext,
        args.dst_ext,
        args.save_dtype,
        args.quantization,
        bool(args.delete_src),
        bool(args.overwrite),
    )

    converted = 0
    skipped = 0

    with Pool(
        processes=int(args.processes),
        initializer=_init_worker,
        initargs=init_args,
        maxtasksperchild=int(args.maxtasksperchild),
    ) as pool:
        for i, did in enumerate(pool.imap_unordered(_convert_one, (str(p) for p in src_files)), start=1):
            if did:
                converted += 1
            else:
                skipped += 1
            if i % 1000 == 0:
                print(f"progress: {i}/{len(src_files)} converted={converted} skipped={skipped}", flush=True)

    print(f"done: total={len(src_files)} converted={converted} skipped={skipped}", flush=True)


if __name__ == "__main__":
    main()

