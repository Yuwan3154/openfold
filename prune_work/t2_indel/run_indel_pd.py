"""Partial diffusion over the edited panel inputs: one npz per (model, chain, item).

Items per chain: d00..d63 (the edited inputs) and c00..c63 (native.pdb, the no-indel control with the SAME
per-item seed). Each item is one call with the t* ladder as a per-sample rewind list (tiered, descending) or
one call per rung (grouped). Seed = (crc32(chain) + draw) % 2**31, identical for d<k> and c<k>.
Resumable: an existing output npz is skipped. A cc89 item whose residue-index span exceeds --span-cutoff is
SKIPPED AND RECORDED in skipped.jsonl, never run. No try/except: a runtime failure is a real bug.
Env: protpardelle on a GPU node. Run:
  python run_indel_pd.py --inputs-dir <inputs> --out-root <out> --model cc89 --chains 7du7_A --draws 0 1
"""
import argparse
import json
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch

from atomic_io import atomic_savez

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from generate_templates import MODEL_EPOCH, sampling_kwargs  # noqa: E402
from protpardelle.core.models import load_model  # noqa: E402
from protpardelle.data.pdb_io import load_feats_from_pdb  # noqa: E402
from protpardelle.env import (  # noqa: E402
    MINIMPNN_WEIGHTS,
    PROTPARDELLE_MODEL_CONFIGS,
    PROTPARDELLE_MODEL_WEIGHTS,
)
from protpardelle.utils import seed_everything  # noqa: E402


def run_item(model, pdb, rewinds, schedule, seed):
    feats, _ = load_feats_from_pdb(pdb, include_pos_feats=True)
    n = len(rewinds) if schedule == "tiered" else 1

    def call(rw):
        ridx = torch.tile(feats["residue_index"][None], (n, 1)).cuda()
        cidx = torch.tile(feats["chain_index"][None], (n, 1)).cuda()
        with torch.no_grad():
            return model.sample(
                seq_mask=torch.ones_like(ridx).cuda(), residue_index=ridx, chain_index=cidx, hotspots=None,
                sse_cond=None, adj_cond=None, motif_placements_full=None,
                dummy_fill_mode=model.config.data.dummy_fill_mode, **sampling_kwargs(pdb, rw))

    seed_everything(seed)
    t0 = time.perf_counter()
    if schedule == "tiered":
        aux = call(list(rewinds))
        coords = aux["xt_traj"][-1].numpy().astype(np.float32)
    else:
        outs = [call(r) for r in rewinds]
        coords = np.concatenate([o["xt_traj"][-1].numpy().astype(np.float32) for o in outs], 0)
        aux = outs[-1]
    dt = time.perf_counter() - t0
    atom_mask = aux["atom_mask"][0].cpu().numpy().astype(bool)
    packed = coords.reshape(coords.shape[0], -1, 3)[:, atom_mask.reshape(-1), :]
    return dict(coords=packed, atom_mask=atom_mask, aatype=aux["s"][0].cpu().numpy().astype(np.int8),
                residue_index=feats["residue_index_orig"].numpy().astype(np.int32),
                rewind_steps=np.asarray(list(rewinds), np.int16), seconds=np.float32(dt)), int(feats["aatype"].shape[0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs-dir", required=True)
    p.add_argument("--out-root", required=True)
    p.add_argument("--model", required=True, choices=sorted(MODEL_EPOCH))
    p.add_argument("--rewinds", type=int, nargs="+", default=[375, 300, 250, 200], help="DESCENDING (tiered needs it)")
    p.add_argument("--chains", nargs="*", default=None)
    p.add_argument("--draws", type=int, nargs="*", default=None, help="default: all draws in plans.json")
    p.add_argument("--kinds", nargs="+", default=["indel", "control"], choices=["indel", "control"])
    p.add_argument("--schedule", default="tiered", choices=["tiered", "grouped"])
    p.add_argument("--span-cutoff", type=int, default=484)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    a = p.parse_args()
    assert a.rewinds == sorted(a.rewinds, reverse=True), "rewinds must be descending"

    keys = a.chains if a.chains else sorted(os.listdir(a.inputs_dir))
    keys = keys[a.shard::a.num_shards]
    model = load_model(str(PROTPARDELLE_MODEL_CONFIGS / f"{a.model}.yaml"),
                       str(PROTPARDELLE_MODEL_WEIGHTS / f"{a.model}_epoch{MODEL_EPOCH[a.model]}.pth"))
    model.load_minimpnn(MINIMPNN_WEIGHTS)
    out_root = Path(a.out_root) / a.model
    done = skipped = 0
    t_start = time.perf_counter()
    for key in keys:
        d = Path(a.inputs_dir) / key
        plans = json.load(open(d / "plans.json"))
        draws = a.draws if a.draws is not None else list(range(len(plans["plans"])))
        (out_root / key).mkdir(parents=True, exist_ok=True)
        for k in draws:
            seed = (zlib.crc32(key.encode()) + k) % 2**31
            items = []
            if "indel" in a.kinds:
                items.append((f"d{k:02d}", d / f"d{k:02d}.pdb", plans["plans"][k]["L_new"]))
            if "control" in a.kinds:
                items.append((f"c{k:02d}", d / "native.pdb", plans["L"]))
            for name, pdb, span in items:
                out = out_root / key / f"{name}.npz"
                if out.is_file():
                    continue
                if a.model == "cc89" and span > a.span_cutoff:
                    skipped += 1
                    open(Path(a.out_root) / "skipped.jsonl", "a").write(
                        json.dumps({"model": a.model, "chain": key, "item": name, "span": span}) + "\n")
                    continue
                res, L = run_item(model, str(pdb), a.rewinds, a.schedule, seed)
                assert L == span, (key, name, L, span)
                atomic_savez(out, model=a.model, schedule=a.schedule, seed=np.int64(seed), L_new=np.int32(L), **res)
                done += 1
                print(f"{a.model} {key} {name} L={L} {float(res['seconds']):.1f}s ({a.schedule}) "
                      f"done={done} skipped={skipped}", flush=True)
    print(f"finished: {done} written, {skipped} skipped, {(time.perf_counter() - t_start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
