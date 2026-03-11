#!/usr/bin/env python3

import argparse
import json
import time
from typing import Dict, Tuple

import torch
import torch._dynamo as dynamo
from torch._dynamo.utils import counters

from openfold.config import model_config
from openfold.model.model import AlphaFold


def _make_batch(seq_len: int, preset: str) -> Dict[str, torch.Tensor]:
    # Reuse the same feature pipeline as training to ensure keys match.
    from openfold.block_replacement_scripts.train_per_block_parallel import PerBlockDataModule

    dm = PerBlockDataModule(
        dataset_path="dummy",
        config_preset=preset,
        batch_size=1,
        num_workers=0,
        min_length=None,
        max_length=None,
        validation_fraction=0.0,
        split_seed=0,
    )
    batch = dm.collate_fn([{"id": "test", "sequence": "A" * seq_len}])
    batch.pop("seq_id")
    batch.pop("seq_length")
    batch = {k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    return batch


def _register_block_capture_hooks(model: AlphaFold) -> Tuple[Dict, list]:
    block_outputs: Dict = {}
    hooks = []

    def make_hook(block_idx: int):
        def hook(module, inputs, outputs):
            m_in, z_in = inputs[0], inputs[1]
            m_out, z_out = outputs[0], outputs[1]

            m_in_single = m_in[..., 0, :, :].detach().clone()
            m_out_single = m_out[..., 0, :, :].detach().clone()

            block_outputs[block_idx] = {
                "input": (m_in_single, z_in.detach().clone()),
                "output": (m_out_single, z_out.detach().clone()),
            }

        return hook

    for idx, block in enumerate(model.evoformer.blocks):
        hooks.append(block.register_forward_hook(make_hook(idx)))

    return block_outputs, hooks


def _run_n_forwards(
    model,
    batch: Dict[str, torch.Tensor],
    block_outputs: Dict,
    iters: int,
    autocast: bool,
) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast):
            for _ in range(iters):
                block_outputs.clear()
                _ = model(batch)
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / float(iters)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark teacher torch.compile with block-capture hooks.")
    parser.add_argument("--preset", type=str, default="model_1_ptm")
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--autocast", action="store_true", default=True)
    parser.add_argument("--no_autocast", action="store_true", default=False)
    parser.add_argument("--backend", type=str, default="inductor")
    parser.add_argument("--mode", type=str, default="reduce-overhead")
    parser.add_argument("--dynamic", action="store_true", default=True)
    parser.add_argument("--static", action="store_true", default=False)
    parser.add_argument("--fullgraph", action="store_true", default=False)
    parser.add_argument("--compile_only", action="store_true", default=False)
    args = parser.parse_args()

    if args.no_autocast:
        autocast = False
    else:
        autocast = bool(args.autocast)

    if args.static:
        dynamic = False
    else:
        dynamic = bool(args.dynamic)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise ValueError("This benchmark requires CUDA.")

    batch = _make_batch(seq_len=args.seq_len, preset=args.preset)

    cfg = model_config(args.preset, train=False, low_prec=False)
    model = AlphaFold(cfg).cuda().eval()
    for p in model.parameters():
        p.requires_grad_(False)

    block_outputs, hooks = _register_block_capture_hooks(model)

    dynamo.config.suppress_errors = True
    counters.clear()

    compile_mode = None if args.mode.lower() in {"none", "null"} else args.mode
    compiled = torch.compile(
        model,
        backend=args.backend,
        mode=compile_mode,
        dynamic=dynamic,
        fullgraph=bool(args.fullgraph),
    )

    # Compilation batch (timed separately; excluded from throughput)
    block_outputs.clear()
    t0 = time.perf_counter()
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast):
            _ = compiled(batch)
    torch.cuda.synchronize()
    compile_time_s = time.perf_counter() - t0
    blocks_captured = int(len(block_outputs))

    if args.compile_only:
        result = {
            "compile_time_s": float(compile_time_s),
            "blocks_captured": blocks_captured,
            "backend": args.backend,
            "mode": compile_mode,
            "dynamic": dynamic,
            "fullgraph": bool(args.fullgraph),
            "dynamo": {
                "frames_ok": int(counters["frames"]["ok"]) if "frames" in counters else 0,
                "unique_graphs": int(counters["stats"]["unique_graphs"]) if "stats" in counters else 0,
            },
        }
        print(json.dumps(result, indent=2))
        for h in hooks:
            h.remove()
        return

    # Baseline eager timing (exclude any compilation)
    eager_avg_ms = _run_n_forwards(
        model=model, batch=batch, block_outputs=block_outputs, iters=args.iters, autocast=autocast
    )

    compiled_avg_ms = _run_n_forwards(
        model=compiled, batch=batch, block_outputs=block_outputs, iters=args.iters, autocast=autocast
    )

    result = {
        "preset": args.preset,
        "seq_len": int(args.seq_len),
        "iters": int(args.iters),
        "autocast_bf16": bool(autocast),
        "backend": args.backend,
        "mode": compile_mode,
        "dynamic": dynamic,
        "fullgraph": bool(args.fullgraph),
        "compile_time_s": float(compile_time_s),
        "blocks_captured": blocks_captured,
        "eager_avg_ms": float(eager_avg_ms),
        "compiled_avg_ms": float(compiled_avg_ms),
        "speedup": float(eager_avg_ms / compiled_avg_ms) if compiled_avg_ms > 0 else None,
        "dynamo": {
            "frames_ok": int(counters["frames"]["ok"]) if "frames" in counters else 0,
            "unique_graphs": int(counters["stats"]["unique_graphs"]) if "stats" in counters else 0,
        },
    }
    print(json.dumps(result, indent=2))

    for h in hooks:
        h.remove()


if __name__ == "__main__":
    main()

