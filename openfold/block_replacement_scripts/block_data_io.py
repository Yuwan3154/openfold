#!/usr/bin/env python3
"""
Utilities for saving/loading Evoformer block input/output pairs with optional compression.

Supported formats (by file extension):
  - .pt              : torch.save / torch.load (pickle)
  - .pt.gz           : gzip-compressed torch.save (legacy serialization; streamable)
  - .safetensors     : safetensors (flat tensor dict + metadata)
  - .safetensors.gz  : gzip-compressed safetensors bytes
  - .safetensors.znn : ZipNN-compressed safetensors (tensors stored as uint8 + metadata)
  - .df11.safetensors: DFloat11-style lossless compression for BF16 (Huffman-coded exponents + raw sign/mantissa)

Note: DFloat11 here is implemented as a file-level lossless codec for BF16 tensors, not a quantization scheme.
"""

from __future__ import annotations

import math
import gzip
import importlib.util
import os
from pathlib import Path
import json
from typing import Any, Dict, Literal, Mapping, Optional, Tuple

import numpy as np
import torch

from safetensors.torch import load as safetensors_load
from safetensors.torch import safe_open as safetensors_safe_open
from safetensors.torch import save as safetensors_save
from safetensors.torch import save_file as safetensors_save_file


BlockDataFormat = Literal[
    "pt",
    "pt.gz",
    "safetensors",
    "safetensors.gz",
    "safetensors.znn",
    "df11.safetensors",
]
BlockDataQuant = Literal["none"]
BlockDataDType = Literal["float32", "bf16"]


ZIPNN_AVAILABLE = importlib.util.find_spec("zipnn") is not None
if ZIPNN_AVAILABLE:
    from zipnn import ZipNN  # type: ignore
    from zipnn.util_safetensors import (  # type: ignore
        COMPRESSED_DTYPE,
        COMPRESSION_METHOD,
        build_compressed_tensor_info,
        get_compressed_tensors_metadata,
        set_compressed_tensors_metadata,
    )

    _TORCH_DTYPE_TO_ZIPNN_STR: Dict[torch.dtype, str] = {
        torch.float32: "float32",
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float8_e4m3fn: "float8_e4m3fn",
        torch.float8_e5m2: "float8_e5m2",
    }

    # ZipNN object construction appears to leak/accumulate memory when done repeatedly.
    # Cache instances keyed by dtype string and reuse across calls.
    _ZIPNN_BYTE_CODEC_BY_DTYPE: Dict[str, ZipNN] = {}

    def _get_zipnn_byte_codec(dtype_str: str) -> ZipNN:
        znn = _ZIPNN_BYTE_CODEC_BY_DTYPE.get(dtype_str)
        if znn is None:
            znn = ZipNN(input_format="byte", bytearray_dtype=dtype_str, method=COMPRESSION_METHOD)
            _ZIPNN_BYTE_CODEC_BY_DTYPE[dtype_str] = znn
        return znn


DAHUFFMAN_AVAILABLE = importlib.util.find_spec("dahuffman") is not None
if DAHUFFMAN_AVAILABLE:
    from dahuffman import HuffmanCodec  # type: ignore


DFLOAT11_CUPY_AVAILABLE = (
    importlib.util.find_spec("cupy") is not None and importlib.util.find_spec("dfloat11") is not None
)
if DFLOAT11_CUPY_AVAILABLE:
    import cupy as cp  # type: ignore

    _DFLOAT11_SPEC = importlib.util.find_spec("dfloat11")
    if _DFLOAT11_SPEC is None or _DFLOAT11_SPEC.submodule_search_locations is None:
        raise RuntimeError("dfloat11 appears installed but its module spec is missing package locations")
    _DFLOAT11_PKG_DIR = Path(list(_DFLOAT11_SPEC.submodule_search_locations)[0])
    _DFLOAT11_PTX_PATH = _DFLOAT11_PKG_DIR / "decode.ptx"
    if not _DFLOAT11_PTX_PATH.exists():
        raise RuntimeError(f"dfloat11 PTX kernel not found at {_DFLOAT11_PTX_PATH}")
    _DF11_DECODE_KERNEL = cp.RawModule(path=str(_DFLOAT11_PTX_PATH)).get_function("decode")

# NOTE: The upstream DF11 decode kernel (decode.ptx) has shown intermittent CUDA illegal access
# with threads_per_block=512 on some real-world block-cache tensors. Using 256 threads avoids the
# issue in our repro while preserving GPU-side decode.
DF11_THREADS_PER_BLOCK = 256
DF11_BYTES_PER_THREAD = 8


def _dtype_str_to_torch(dtype_str: str) -> torch.dtype:
    # Stored dtype strings in zipnn metadata look like "bfloat16", "float32", etc.
    if dtype_str == "bfloat16":
        return torch.bfloat16
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "float32":
        return torch.float32
    if dtype_str == "float8_e4m3fn":
        return torch.float8_e4m3fn
    if dtype_str == "float8_e5m2":
        return torch.float8_e5m2
    raise ValueError(f"Unsupported dtype in compressed metadata: {dtype_str}")


def sanitize_id(seq_id: str) -> str:
    """Make a safe filename token from an arbitrary sequence id."""
    return (
        seq_id.replace(os.sep, "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )


def cast_floating_tensors(obj: Any, dtype: torch.dtype) -> Any:
    """Recursively cast floating-point torch.Tensors inside nested dict/list/tuple structures."""
    if isinstance(obj, torch.Tensor):
        if torch.is_floating_point(obj) and obj.dtype != dtype:
            return obj.to(dtype=dtype)
        return obj
    if isinstance(obj, dict):
        return {k: cast_floating_tensors(v, dtype) for k, v in obj.items()}
    if isinstance(obj, list):
        return [cast_floating_tensors(v, dtype) for v in obj]
    if isinstance(obj, tuple):
        return tuple(cast_floating_tensors(v, dtype) for v in obj)
    return obj


def quantize_floating_tensors(obj: Any, quant: BlockDataQuant) -> Any:
    """Apply quantization to all floating-point tensors in a nested structure."""
    if quant == "none":
        return obj

    raise ValueError(f"Unsupported quantization: {quant}")


def _flatten_block_sample(sample: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
    return {
        "input.m": sample["input"]["m"],
        "input.z": sample["input"]["z"],
        "output.m": sample["output"]["m"],
        "output.z": sample["output"]["z"],
    }


def _unflatten_block_sample(
    tensors: Mapping[str, torch.Tensor],
    metadata: Optional[Mapping[str, str]] = None,
    fallback_chain_id: Optional[str] = None,
    fallback_block_idx: Optional[int] = None,
) -> Dict[str, Any]:
    chain_id = fallback_chain_id
    block_idx = fallback_block_idx
    if metadata is not None:
        if "chain_id" in metadata:
            chain_id = metadata["chain_id"]
        if "block_idx" in metadata:
            block_idx = int(metadata["block_idx"])

    return {
        "input": {"m": tensors["input.m"], "z": tensors["input.z"]},
        "output": {"m": tensors["output.m"], "z": tensors["output.z"]},
        "chain_id": chain_id,
        "block_idx": block_idx,
    }


def _df11_encode_tensor_bf16(t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, int]]:
    """
    Lossless DFloat11-style encoding for BF16 values.

    We split BF16 into:
      - exponent (8 bits)   -> Huffman-coded bytes
      - sign+mantissa (8b)  -> stored raw as uint8
    """
    if not DAHUFFMAN_AVAILABLE:
        raise RuntimeError("dahuffman is required for df11.safetensors")

    t_bf16 = t.to(dtype=torch.bfloat16).contiguous()
    w = t_bf16.view(torch.int16)

    exponent = ((w >> 7) & 0xFF).to(torch.uint8).flatten()
    sign_mantissa = (((w >> 8) & 0x80) | (w & 0x7F)).to(torch.uint8).flatten()

    vals, freqs = torch.unique(exponent, return_counts=True)
    counter: Dict[int, int] = {int(v.item()): int(f.item()) for v, f in zip(vals, freqs)}

    codec = HuffmanCodec.from_frequencies(counter)
    encoded_bytes = codec.encode(exponent.tolist())
    encoded = torch.from_numpy(np.frombuffer(encoded_bytes, dtype=np.uint8).copy())

    return encoded, sign_mantissa.cpu(), counter


def _df11_decode_tensor_bf16(
    encoded_exponent: torch.Tensor,
    sign_mantissa: torch.Tensor,
    counter: Dict[int, int],
    shape: Tuple[int, ...],
) -> torch.Tensor:
    if not DAHUFFMAN_AVAILABLE:
        raise RuntimeError("dahuffman is required for df11.safetensors")

    codec = HuffmanCodec.from_frequencies(counter)
    exp_list = codec.decode(encoded_exponent.detach().cpu().numpy().tobytes())
    exponent = torch.tensor(exp_list, dtype=torch.int16)

    sm = sign_mantissa.to(torch.int16)
    w = (exponent << 7) | (sm & 0x7F) | ((sm & 0x80) << 8)
    return w.view(torch.bfloat16).reshape(shape)


def _df11_get_32bit_codec(counter: Dict[int, int]) -> Tuple["HuffmanCodec", Dict[int, int], Dict[Any, Tuple[int, int]]]:
    """
    Ensure Huffman codes fit within 32 bits by flattening the tail of the frequency distribution.

    This matches the strategy used in the upstream DFloat11 reference implementation.
    """
    if not DAHUFFMAN_AVAILABLE:
        raise RuntimeError("dahuffman is required for df11.safetensors")

    codec = HuffmanCodec.from_frequencies(counter)
    table = codec.get_code_table()
    max_len = 0
    for _, (l, _) in table.items():
        max_len = max(max_len, l)

    compressed_codec = codec
    compressed_counter = counter

    min_k = 2
    keys = np.array(list(counter.keys()), dtype=np.int64)
    freq = np.array(list(counter.values()), dtype=np.int64)
    while max_len > 32:
        min_indices = np.argpartition(freq, min_k)[:min_k]
        min_k += 1
        min_keys = keys[min_indices]

        compressed_counter = dict(counter)
        for k in min_keys:
            compressed_counter[int(k)] = 1
        compressed_codec = HuffmanCodec.from_frequencies(compressed_counter)
        table = compressed_codec.get_code_table()
        max_len = 0
        for _, (l, _) in table.items():
            max_len = max(max_len, l)

    return compressed_codec, compressed_counter, table


def _df11_build_luts(table: Dict[Any, Tuple[int, int]]) -> torch.Tensor:
    """
    Build hierarchical byte-wise lookup tables for the DF11 GPU decoder kernel.

    Ported from the upstream DFloat11 reference implementation. The kernel expects:
      - luts: uint8 tensor of shape [n_luts, 256]
      - last row contains code lengths for symbols 0..255
      - earlier rows are nested LUTs; entries >= 240 act as indirections
    """
    prefixes = [""]
    for key, (bits, val) in table.items():
        if isinstance(key, int):
            prefix = bin(val)[2:].rjust(bits, "0")[: ((bits - 1) // 8 * 8)]
            if prefix not in prefixes:
                prefixes.append(prefix)

    prefixes.sort(key=len)
    luts = np.zeros((len(prefixes), 256), dtype=np.uint8)

    for pi, p in enumerate(prefixes):
        bytes_dict: Dict[int, int] = {}
        pl = len(p) // 8
        for key, (bits, val) in table.items():
            if isinstance(key, int):
                bin_val = bin(val)[2:].rjust(bits, "0")
                if bin_val.startswith(p):
                    if (bits - 1) // 8 == pl:
                        dict_key = int(bin_val[(pl * 8) :].ljust(8, "0"), 2)
                        dict_value = int(key)
                    else:
                        dict_key = int(bin_val[(pl * 8) : (pl * 8 + 8)], 2)
                        dict_value = 256 - prefixes.index(bin_val[: (pl * 8 + 8)])

                    if dict_key in bytes_dict and bytes_dict[dict_key] != dict_value:
                        raise ValueError(f"Key {dict_key} already exists in {bytes_dict}")
                    bytes_dict[dict_key] = dict_value

        for i in range(256):
            if i in bytes_dict:
                curr_val = bytes_dict[i]
            luts[pi, i] = curr_val

    lens = np.zeros((1, 256), dtype=np.uint8)
    for key, (bits, _) in table.items():
        if isinstance(key, int):
            lens[-1, int(key)] = int(bits)

    return torch.from_numpy(np.concatenate((luts, lens), axis=0))


def _df11_encode_exponents_with_gaps(
    exponent_u8: torch.Tensor,
    codec: "HuffmanCodec",
    *,
    threads_per_block: int,
    bytes_per_thread: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Encode exponent bytes using the DF11 bitstream format and compute the auxiliary arrays
    required for GPU decoding (gaps + output_positions).

    Returns:
      - encoded_exponent (uint8)   : Huffman bitstream bytes
      - gaps (uint8)              : 5-bit gaps packed with np.packbits
      - output_positions (int32)  : output start indices per block (uint32 in kernel)
    """
    if not DAHUFFMAN_AVAILABLE:
        raise RuntimeError("dahuffman is required for df11.safetensors")

    exp_cpu = exponent_u8.detach().contiguous().cpu().numpy().tobytes()

    bits_per_thread = 8 * bytes_per_thread
    bits_per_block = bits_per_thread * threads_per_block

    encoded = bytearray()
    gaps: list[int] = []
    output_positions: list[int] = []

    buffer = 0
    size = 0
    total_size = 0
    element_count = 0

    # Precompute (bits, value) arrays for symbol lookup (0..255) for speed.
    bits_lut = np.zeros(256, dtype=np.uint8)
    val_lut = np.zeros(256, dtype=np.uint32)
    for sym, (b, v) in codec.get_code_table().items():
        if isinstance(sym, int):
            bits_lut[sym] = b
            val_lut[sym] = v

    for s in exp_cpu:
        if total_size // bits_per_thread + 1 > len(gaps):
            gaps.append(total_size - (total_size // bits_per_thread) * bits_per_thread)
        if total_size // bits_per_block + 1 > len(output_positions):
            output_positions.append(element_count)

        b = int(bits_lut[s])
        v = int(val_lut[s])
        buffer = (buffer << b) + v
        size += b
        total_size += b
        element_count += 1

        while size >= 8:
            byte = buffer >> (size - 8)
            encoded.append(byte)
            buffer = buffer - (byte << (size - 8))
            size -= 8

    # Same final-byte handling as dahuffman/DFloat11: append partial EOF bits up to end-of-byte.
    if size > 0:
        # IMPORTANT: Mirror the upstream DFloat11 reference implementation.
        # Before appending EOF bits, ensure gaps/output_positions have entries for the final partial chunk.
        if total_size // bits_per_thread + 1 > len(gaps):
            gaps.append(total_size - (total_size // bits_per_thread) * bits_per_thread)
        if total_size // bits_per_block + 1 > len(output_positions):
            output_positions.append(element_count)

        b_eof, v_eof = codec.get_code_table()[codec._eof]  # type: ignore[attr-defined]
        buffer = (buffer << b_eof) + v_eof
        size += b_eof
        if size >= 8:
            byte = buffer >> (size - 8)
        else:
            byte = buffer << (8 - size)
        encoded.append(int(byte))

    output_positions.append(int(exponent_u8.numel()))

    n_bytes = len(encoded)
    blocks_per_grid = int(math.ceil(n_bytes / (threads_per_block * bytes_per_thread)))

    # Pad gaps to cover all threads (threads_per_block * blocks_per_grid). Each gap is 5 bits.
    total_threads = threads_per_block * blocks_per_grid
    if len(gaps) < total_threads:
        gaps.extend([0] * (total_threads - len(gaps)))
    gaps = gaps[:total_threads]

    binary_str_gaps = [format(gap, "05b") for gap in gaps]
    binary_gaps = [int(bit) for binary in binary_str_gaps for bit in binary]
    gaps_u8 = np.packbits(binary_gaps)

    # torch.from_numpy does not support np.uint32; store as int32 (bitwise identical for non-negative).
    outpos_i32 = np.array(output_positions, dtype=np.int32)

    encoded_t = torch.from_numpy(np.frombuffer(bytes(encoded), dtype=np.uint8).copy())
    gaps_t = torch.from_numpy(gaps_u8.astype(np.uint8, copy=False))
    outpos_t = torch.from_numpy(outpos_i32)
    return encoded_t, gaps_t, outpos_t


def _df11_encode_tensor_bf16_v2(
    t: torch.Tensor,
    *,
    threads_per_block: int,
    bytes_per_thread: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[int, int]]:
    """
    DF11 v2 encoder that produces GPU-decodable components.

    Returns:
      encoded_exponent (uint8), sign_mantissa (uint8), luts (uint8),
      output_positions (int32), gaps (uint8), counter (for reproducibility/debug)
    """
    if not DAHUFFMAN_AVAILABLE:
        raise RuntimeError("dahuffman is required for df11.safetensors")

    t_bf16 = t.to(dtype=torch.bfloat16).contiguous()
    w = t_bf16.view(torch.int16)

    exponent = ((w >> 7) & 0xFF).to(torch.uint8).flatten()
    sign_mantissa = (((w >> 8) & 0x80) | (w & 0x7F)).to(torch.uint8).flatten()

    vals, freqs = torch.unique(exponent, return_counts=True)
    counter: Dict[int, int] = {int(v.item()): int(f.item()) for v, f in zip(vals, freqs)}

    codec, counter32, table = _df11_get_32bit_codec(counter)
    luts = _df11_build_luts(table)
    encoded, gaps, output_positions = _df11_encode_exponents_with_gaps(
        exponent,
        codec,
        threads_per_block=threads_per_block,
        bytes_per_thread=bytes_per_thread,
    )

    # Store the (possibly modified) counter actually used for encoding.
    return (
        encoded,
        sign_mantissa.cpu(),
        luts.cpu(),
        output_positions.cpu(),
        gaps.cpu(),
        counter32,
    )


def _df11_decode_tensor_bf16_v2_gpu(
    *,
    luts: torch.Tensor,
    encoded_exponent: torch.Tensor,
    sign_mantissa: torch.Tensor,
    output_positions: torch.Tensor,
    gaps: torch.Tensor,
    shape: Tuple[int, ...],
    device: torch.device,
    threads_per_block: int,
    bytes_per_thread: int,
) -> torch.Tensor:
    if not DFLOAT11_CUPY_AVAILABLE:
        raise RuntimeError("cupy+dfloat11 is required for DF11 GPU decode")

    if device.type != "cuda":
        raise ValueError(f"DF11 GPU decode requires a CUDA device, got {device}")

    # Validate DF11 v3 payload invariants on CPU *before* any GPU work.
    # This prevents malformed cache entries from triggering CUDA illegal access and
    # crashing the whole DDP job with asynchronous error reporting.
    if luts.dtype != torch.uint8 or luts.ndim != 2 or luts.shape[1] != 256:
        raise RuntimeError(f"DF11 invalid luts: dtype={luts.dtype} shape={tuple(luts.shape)}")
    if encoded_exponent.dtype != torch.uint8 or encoded_exponent.ndim != 1:
        raise RuntimeError(
            f"DF11 invalid encoded_exponent: dtype={encoded_exponent.dtype} shape={tuple(encoded_exponent.shape)}"
        )
    if sign_mantissa.dtype != torch.uint8 or sign_mantissa.ndim != 1:
        raise RuntimeError(
            f"DF11 invalid sign_mantissa: dtype={sign_mantissa.dtype} shape={tuple(sign_mantissa.shape)}"
        )
    if output_positions.ndim != 1 or output_positions.dtype != torch.int32:
        raise RuntimeError(
            f"DF11 invalid output_positions: dtype={output_positions.dtype} shape={tuple(output_positions.shape)}"
        )
    if gaps.dtype != torch.uint8 or gaps.ndim != 1:
        raise RuntimeError(f"DF11 invalid gaps: dtype={gaps.dtype} shape={tuple(gaps.shape)}")

    n_bytes_cpu = int(encoded_exponent.numel())
    n_elements_cpu = int(sign_mantissa.numel())
    outpos_i64 = output_positions.to(dtype=torch.int64)
    if outpos_i64.numel() < 1:
        raise RuntimeError("DF11 invalid output_positions: empty")
    if int(outpos_i64[0].item()) != 0:
        raise RuntimeError(f"DF11 invalid output_positions: first={int(outpos_i64[0].item())} expected=0")
    if outpos_i64.numel() >= 2:
        if bool((outpos_i64[1:] < outpos_i64[:-1]).any().item()):
            raise RuntimeError("DF11 invalid output_positions: not non-decreasing")
    if int(outpos_i64[-1].item()) != n_elements_cpu:
        raise RuntimeError(
            f"DF11 invalid output_positions: last={int(outpos_i64[-1].item())} expected={n_elements_cpu}"
        )

    blocks_cpu = int(math.ceil(n_bytes_cpu / (threads_per_block * bytes_per_thread))) if n_bytes_cpu > 0 else 0
    # gaps is packbits of 5-bit per-thread gap values; encoder pads to threads_per_block*blocks elements.
    expected_gap_bits = threads_per_block * blocks_cpu * 5
    expected_gap_bytes = (expected_gap_bits + 7) // 8
    # IMPORTANT: The upstream decode kernel reads `gaps[(tid*5//8)+1]` unconditionally, so it can
    # read one byte past the nominal packed-bit length. We therefore require (or create) a
    # trailing padding byte.
    required_gap_bytes = int(expected_gap_bytes) + 1
    gaps_n = int(gaps.numel())
    if gaps_n not in (int(expected_gap_bytes), required_gap_bytes):
        raise RuntimeError(
            f"DF11 invalid gaps length: got={gaps_n} expected={int(expected_gap_bytes)} or {required_gap_bytes} "
            f"(tpb={threads_per_block} bpt={bytes_per_thread} n_bytes={n_bytes_cpu} blocks={blocks_cpu} threads={threads_per_block * blocks_cpu})"
        )

    # Some tensors can require >48KB of dynamic shared memory per block for DF11 decode.
    # On Ampere+ GPUs this requires opting-in by setting `max_dynamic_shared_size_bytes`
    # on the CuPy kernel; otherwise the kernel can fail with CUDA illegal access.
    #
    # Cache per-device setting to avoid reconfiguring the kernel on every call.
    global _DF11_KERNEL_MAX_DYN_SHARED_BY_DEVICE
    if "_DF11_KERNEL_MAX_DYN_SHARED_BY_DEVICE" not in globals():
        _DF11_KERNEL_MAX_DYN_SHARED_BY_DEVICE = {}

    # Compute shared-mem sizing on CPU from output_positions to avoid relying on any prior GPU state.
    # Note: outpos_i64 already validated above.
    if outpos_i64.numel() >= 2:
        max_block_elems = int((outpos_i64[1:] - outpos_i64[:-1]).max().item())
    else:
        max_block_elems = int(n_elements_cpu)

    shared_mem_size = threads_per_block * 4 + 4 + max_block_elems * 2

    # Ensure all inputs are on GPU.
    #
    # IMPORTANT: CuPy uses its own stream by default. If we launch the decode kernel on a
    # different stream from PyTorch, it can race with the H2D copies enqueued by PyTorch and
    # crash with CUDA illegal access. We therefore launch the kernel on PyTorch's current stream.
    luts_gpu = luts.to(device=device, dtype=torch.uint8, non_blocking=True).contiguous()
    codes_gpu = encoded_exponent.to(device=device, dtype=torch.uint8, non_blocking=True).contiguous()
    sm_gpu = sign_mantissa.to(device=device, dtype=torch.uint8, non_blocking=True).contiguous()
    outpos_gpu = output_positions.to(device=device, dtype=torch.int32, non_blocking=True).contiguous()
    # Ensure gaps has the required trailing padding byte to satisfy the kernel's `+1` read.
    if gaps_n == int(expected_gap_bytes):
        gaps = torch.cat([gaps, torch.zeros(1, dtype=torch.uint8)], dim=0)
    gaps_gpu = gaps.to(device=device, dtype=torch.uint8, non_blocking=True).contiguous()

    n_luts = int(luts_gpu.shape[0])
    n_bytes = int(codes_gpu.numel())
    n_elements = int(sm_gpu.numel())

    # Empty tensors can appear in some samples; avoid launching the CUDA kernel with grid=(0,).
    if n_elements == 0 or n_bytes == 0:
        return torch.empty(shape, dtype=torch.bfloat16, device=device)

    out = torch.empty(n_elements, dtype=torch.bfloat16, device=device)

    blocks = int(math.ceil(n_bytes / (threads_per_block * bytes_per_thread)))
    if blocks <= 0:
        return out.reshape(shape)
    blocks_per_grid = (blocks,)
    with cp.cuda.Device(device.index):
        torch_stream = torch.cuda.current_stream(device)
        # Inform PyTorch's caching allocator these tensors are used on this stream even though
        # the kernel launch is performed via CuPy.
        luts_gpu.record_stream(torch_stream)
        codes_gpu.record_stream(torch_stream)
        sm_gpu.record_stream(torch_stream)
        outpos_gpu.record_stream(torch_stream)
        gaps_gpu.record_stream(torch_stream)
        out.record_stream(torch_stream)

        cp_stream = cp.cuda.ExternalStream(int(torch_stream.cuda_stream))

        # Opt-in to larger dynamic shared memory when needed (up to device limit).
        dev_props = cp.cuda.runtime.getDeviceProperties(int(device.index))
        optin_max = int(dev_props.get("sharedMemPerBlockOptin", dev_props.get("sharedMemPerBlock", 0)))
        if shared_mem_size > optin_max:
            raise RuntimeError(
                f"DF11 decode requires {shared_mem_size} bytes dynamic shared memory, "
                f"but device max opt-in is {optin_max}. "
                f"Re-encode with smaller df11_threads_per_block/df11_bytes_per_thread."
            )

        # Set the kernel's maximum dynamic shared memory once per device (to the device's opt-in max).
        # This avoids repeatedly mutating the kernel attribute across many calls.
        if _DF11_KERNEL_MAX_DYN_SHARED_BY_DEVICE.get(int(device.index)) is None:
            _DF11_DECODE_KERNEL.max_dynamic_shared_size_bytes = optin_max
            _DF11_KERNEL_MAX_DYN_SHARED_BY_DEVICE[int(device.index)] = optin_max

        with cp_stream:
            _DF11_DECODE_KERNEL(
                grid=blocks_per_grid,
                block=(threads_per_block,),
                shared_mem=shared_mem_size,
                args=[
                    luts_gpu.data_ptr(),
                    codes_gpu.data_ptr(),
                    sm_gpu.data_ptr(),
                    outpos_gpu.data_ptr(),
                    gaps_gpu.data_ptr(),
                    out.data_ptr(),
                    n_luts,
                    n_bytes,
                    n_elements,
                ],
            )
        # Optional debug sync to attribute failures to the correct file/tensor.
        if os.environ.get("OPENFOLD_DF11_SYNC", "") == "1":
            cp.cuda.runtime.deviceSynchronize()

    return out.reshape(shape)

def _torch_save_gzip(sample: Mapping[str, Any], path: Path) -> None:
    # Use legacy serialization to allow streaming writes into gzip.
    with gzip.open(path, "wb") as f:
        torch.save(sample, f, _use_new_zipfile_serialization=False)


def _torch_load_gzip(path: Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    with gzip.open(path, "rb") as f:
        return torch.load(f, map_location=map_location)


def _safetensors_save_bytes(
    sample: Mapping[str, Any], metadata: Dict[str, str]
) -> bytes:
    tensors = _flatten_block_sample(sample)
    return safetensors_save(tensors=tensors, metadata=metadata)


def _safetensors_load_bytes(data: bytes) -> Dict[str, torch.Tensor]:
    return safetensors_load(data)


def _zipnn_safetensors_compress(
    tensors: Mapping[str, torch.Tensor],
    metadata: Dict[str, str],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, str]]:
    if not ZIPNN_AVAILABLE:
        raise RuntimeError("zipnn is not available but safetensors.znn was requested")

    compressed_tensors: Dict[str, torch.Tensor] = {}
    compressed_infos: Dict[str, Any] = {}

    for name, t in tensors.items():
        t_cpu = t.detach().cpu()
        dtype_str = _TORCH_DTYPE_TO_ZIPNN_STR.get(t_cpu.dtype)
        if dtype_str is None:
            raise ValueError(f"zipnn compression unsupported dtype: {t_cpu.dtype}")

        # zipnn's torch-path currently expects torch.uint16 for bf16 (not present in torch),
        # so we compress in BYTE mode over the raw tensor bytes and reconstruct from metadata.
        znn = _get_zipnn_byte_codec(dtype_str)
        raw_bytes = t_cpu.contiguous().view(torch.uint8).numpy().tobytes()
        compressed_bytes = znn.compress(raw_bytes)
        comp_np = np.frombuffer(compressed_bytes, dtype=np.uint8).copy()
        compressed_tensors[name] = torch.from_numpy(comp_np).to(dtype=COMPRESSED_DTYPE)
        compressed_infos[name] = build_compressed_tensor_info(t_cpu)

    set_compressed_tensors_metadata(compressed_infos, metadata)
    return compressed_tensors, metadata


def _zipnn_safetensors_decompress_all(
    path: Path, device: str = "cpu"
) -> Tuple[Dict[str, torch.Tensor], Dict[str, str]]:
    if not ZIPNN_AVAILABLE:
        raise RuntimeError("zipnn is not available but safetensors.znn was requested")

    with safetensors_safe_open(str(path), framework="pt", device=device) as f:
        metadata = f.metadata()
        compressed_meta = get_compressed_tensors_metadata(metadata)
        out: Dict[str, torch.Tensor] = {}
        for name in f.keys():
            t = f.get_tensor(name)
            if name in compressed_meta:
                info = compressed_meta[name]
                dtype_str = info["dtype"]
                shape = json.loads(info["shape"])
                znn = _get_zipnn_byte_codec(dtype_str)
                decom_bytes = znn.decompress(t.contiguous().numpy().tobytes())
                buf = torch.from_numpy(np.frombuffer(decom_bytes, dtype=np.uint8).copy())
                out[name] = buf.view(_dtype_str_to_torch(dtype_str)).reshape(shape)
            else:
                out[name] = t
    return out, metadata


def save_block_sample(
    sample: Mapping[str, Any],
    path: Path,
    *,
    save_dtype: BlockDataDType = "float32",
    quantization: BlockDataQuant = "none",
    make_dirs: bool = True,
) -> None:
    """
    Save a single block sample to `path`, inferring format from the extension.
    """
    if make_dirs:
        path.parent.mkdir(parents=True, exist_ok=True)

    dtype = torch.bfloat16 if save_dtype == "bf16" else torch.float32
    sample2 = cast_floating_tensors(sample, dtype=dtype)
    sample2 = quantize_floating_tensors(sample2, quantization)

    suffix = path.name
    if suffix.endswith(".pt"):
        torch.save(sample2, path)
        return
    if suffix.endswith(".pt.gz"):
        _torch_save_gzip(sample2, path)
        return
    if suffix.endswith(".df11.safetensors"):
        flat = _flatten_block_sample(sample2)
        tensors: Dict[str, torch.Tensor] = {}
        df11_meta: Dict[str, Any] = {}

        for k, t in flat.items():
            enc, sm, luts, outpos, gaps, counter = _df11_encode_tensor_bf16_v2(
                t,
                threads_per_block=DF11_THREADS_PER_BLOCK,
                bytes_per_thread=DF11_BYTES_PER_THREAD,
            )
            tensors[f"{k}.encoded_exponent"] = enc
            tensors[f"{k}.sign_mantissa"] = sm
            tensors[f"{k}.luts"] = luts
            tensors[f"{k}.output_positions"] = outpos
            tensors[f"{k}.gaps"] = gaps
            df11_meta[k] = {"shape": list(t.shape), "counter": counter}

        metadata = {
            "format": "pt",
            # v3: DF11 v2 bitstream + GPU-decodable auxiliaries, with fixed gaps/output_positions encoding.
            # Older v2 files in this repo may have been written with a gaps/output_positions bug that can
            # crash the GPU decoder. We keep v2 *decode* support (CPU path) for backwards compatibility.
            "schema": "openfold_block_data_df11_v3",
            "chain_id": str(sample2.get("chain_id", "")),
            "block_idx": str(sample2.get("block_idx", "")),
            "save_dtype": save_dtype,
            "quantization": quantization,
            "df11_threads_per_block": str(DF11_THREADS_PER_BLOCK),
            "df11_bytes_per_thread": str(DF11_BYTES_PER_THREAD),
            "df11": json.dumps(df11_meta),
        }
        safetensors_save_file(tensors, str(path), metadata=metadata)
        return
    if suffix.endswith(".safetensors"):
        metadata = {
            "format": "pt",
            "schema": "openfold_block_data_v1",
            "chain_id": str(sample2.get("chain_id", "")),
            "block_idx": str(sample2.get("block_idx", "")),
            "save_dtype": save_dtype,
            "quantization": quantization,
        }
        safetensors_save_file(_flatten_block_sample(sample2), str(path), metadata=metadata)
        return
    if suffix.endswith(".safetensors.gz"):
        metadata = {
            "format": "pt",
            "schema": "openfold_block_data_v1",
            "chain_id": str(sample2.get("chain_id", "")),
            "block_idx": str(sample2.get("block_idx", "")),
            "save_dtype": save_dtype,
            "quantization": quantization,
        }
        raw = _safetensors_save_bytes(sample2, metadata=metadata)
        with gzip.open(path, "wb") as f:
            f.write(raw)
        return
    if suffix.endswith(".safetensors.znn"):
        if not ZIPNN_AVAILABLE:
            raise RuntimeError("zipnn is required to write .safetensors.znn")
        metadata = {
            "format": "pt",
            "schema": "openfold_block_data_v1",
            "chain_id": str(sample2.get("chain_id", "")),
            "block_idx": str(sample2.get("block_idx", "")),
            "save_dtype": save_dtype,
            "quantization": quantization,
            "zipnn_method": COMPRESSION_METHOD,
        }
        tensors = _flatten_block_sample(sample2)
        compressed_tensors, metadata = _zipnn_safetensors_compress(tensors, metadata)
        safetensors_save_file(compressed_tensors, str(path), metadata=metadata)
        return

    raise ValueError(f"Unrecognized block data format from path: {path}")


def load_block_sample(path: Path, *, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    """
    Load a block sample saved by `save_block_sample`.
    """
    name = path.name
    if name.endswith(".pt"):
        return torch.load(path, map_location=map_location)
    if name.endswith(".pt.gz"):
        return _torch_load_gzip(path, map_location=map_location)
    if name.endswith(".df11.safetensors"):
        # NOTE: We always open DF11 files on CPU first to read metadata safely. Depending on schema,
        # we may decode on GPU (v3) or on CPU (v2 + legacy).
        with safetensors_safe_open(str(path), framework="pt", device="cpu") as f:
            metadata = f.metadata()
            df11_meta = json.loads(metadata.get("df11", "{}"))
            tensors: Dict[str, torch.Tensor] = {}

            schema = metadata.get("schema", "")
            threads_per_block = int(metadata.get("df11_threads_per_block", str(DF11_THREADS_PER_BLOCK)))
            bytes_per_thread = int(metadata.get("df11_bytes_per_thread", str(DF11_BYTES_PER_THREAD)))
            device = map_location if isinstance(map_location, torch.device) else torch.device(str(map_location))

            # v2 files in this repo may have been written with a gaps/output_positions bug that can
            # trigger CUDA illegal memory access in the GPU decoder. Decode them on CPU for safety.
            allow_gpu_decode = (
                schema == "openfold_block_data_df11_v3"
                and device.type == "cuda"
                and DFLOAT11_CUPY_AVAILABLE
            )

            for k, info in df11_meta.items():
                shape = tuple(int(x) for x in info["shape"])
                enc = f.get_tensor(f"{k}.encoded_exponent")
                sm = f.get_tensor(f"{k}.sign_mantissa")
                if allow_gpu_decode:
                    if os.environ.get("OPENFOLD_DF11_DEBUG", "") == "1":
                        print(f"[df11] decode gpu path={path} tensor={k}", flush=True)
                    luts = f.get_tensor(f"{k}.luts")
                    outpos = f.get_tensor(f"{k}.output_positions")
                    gaps = f.get_tensor(f"{k}.gaps")
                    tensors[k] = _df11_decode_tensor_bf16_v2_gpu(
                        luts=luts,
                        encoded_exponent=enc,
                        sign_mantissa=sm,
                        output_positions=outpos,
                        gaps=gaps,
                        shape=shape,
                        device=device,
                        threads_per_block=threads_per_block,
                        bytes_per_thread=bytes_per_thread,
                    )
                else:
                    counter_raw = info["counter"]
                    counter = {int(kk): int(vv) for kk, vv in counter_raw.items()}
                    tensors[k] = _df11_decode_tensor_bf16(enc, sm, counter, shape)

        return _unflatten_block_sample(tensors, metadata=metadata)
    if name.endswith(".safetensors"):
        with safetensors_safe_open(str(path), framework="pt", device=str(map_location)) as f:
            tensors = {k: f.get_tensor(k) for k in f.keys()}
            metadata = f.metadata()
        return _unflatten_block_sample(tensors, metadata=metadata)
    if name.endswith(".safetensors.gz"):
        with gzip.open(path, "rb") as f:
            raw = f.read()
        tensors = _safetensors_load_bytes(raw)
        # metadata isn't available via load(bytes); recover what we can from filename.
        return _unflatten_block_sample(tensors, metadata=None)
    if name.endswith(".safetensors.znn"):
        tensors, metadata = _zipnn_safetensors_decompress_all(path, device=str(map_location))
        return _unflatten_block_sample(tensors, metadata=metadata)

    raise ValueError(f"Unrecognized block data format from path: {path}")


