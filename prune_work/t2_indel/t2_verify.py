"""Per-step correctness checks of the optimised T2 pipeline (run BEFORE any timing test; user 10-09). Env: protpardelle (GPU).
  inmemory : the in-memory partial-diffusion path (pd_inputs, known mask all ones) reproduces the PDB-file path under a fixed seed (max |dx| over atoms).
  padding  : a padded mixed-length batch reproduces each sample run alone, given the SAME initial noisy state per sample (xt_start; the sampler is an ODE afterwards, so any
             difference is padding/batching leakage, not randomness).
  knownmask: unknown atoms (known_mask 0) do not influence the result: their input coordinates are replaced by large random values and the outputs must be identical.
  metrics  : bond medians / DSSP coil fraction / tm_native of stored pool templates equal the old-route values (geom_pools.csv, pool tm_native).
Run: python t2_verify.py --check inmemory|padding|knownmask|metrics --stage-a-dir A --native-pdb P --chain C [--pool-npz F --geom-csv G]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

import indel_pool as ip
import t2_stage_b as sb
from protpardelle.core.models import load_model
from protpardelle.data.atom import dummy_fill_noise_coords, atom37_mask_from_aatype
from protpardelle.data.pdb_io import load_feats_from_pdb
from protpardelle.env import MINIMPNN_WEIGHTS, PROTPARDELLE_MODEL_CONFIGS, PROTPARDELLE_MODEL_WEIGHTS
from protpardelle.utils import seed_everything


def get_model(name="cc89"):
    m = load_model(str(PROTPARDELLE_MODEL_CONFIGS / f"{name}.yaml"), str(PROTPARDELLE_MODEL_WEIGHTS / f"{name}_epoch{sb.MODEL_EPOCH[name]}.pth"))
    m.load_minimpnn(MINIMPNN_WEIGHTS)
    return m


def load_items(stage_a_dir, chain, native_pdb, n):
    sa = json.load(open(os.path.join(stage_a_dir, chain + ".json")))
    feats, _ = load_feats_from_pdb(native_pdb, include_pos_feats=True)
    nat_pos = feats["atom_positions"].float()
    nat_mask = (nat_pos.abs().sum(-1) > 0).float()
    nat_bb = nat_pos[:, sb.BB_IDX].numpy().astype(np.float64)
    return [sb.build_inputs(nat_pos, nat_mask, nat_bb, p) for p in sa["plans"][:n]], nat_pos, feats


def check_inmemory(model, native_pdb, rewind):
    feats, _ = load_feats_from_pdb(native_pdb, include_pos_feats=True)
    pos = feats["atom_positions"].float()
    pos = pos - pos[:, 1].mean(0, keepdim=True)
    aat = feats["aatype"].long()
    L = len(aat)
    full = atom37_mask_from_aatype(aat[None], torch.ones(1, L))[0]
    pb = sb.pad_batch([(pos, aat, full, None, None)] * 2, "cuda")
    seed_everything(5)
    a_mem = sb.run_pd(model, pb, rewind)["xt_traj"][-1]
    kw = sb.sampling_kwargs([rewind] * 2)
    kw["partial_diffusion"]["pdb_file_path"] = native_pdb
    seed_everything(5)
    with torch.no_grad():
        a_file = model.sample(seq_mask=pb["mask"], residue_index=pb["ridx"], chain_index=torch.zeros_like(pb["ridx"]), hotspots=None, sse_cond=None, adj_cond=None,
                              motif_placements_full=None, dummy_fill_mode=model.config.data.dummy_fill_mode, **kw)["xt_traj"][-1]
    print(f"inmemory vs file: max |dx| = {float((a_mem - a_file).abs().max()):.3e} A over {tuple(a_mem.shape)}")


def initial_state(model, pb, rewind, seeds):
    """Per-sample initial noisy state, identical whether the sample is alone or in a padded batch (same per-sample generator seed)."""
    kw = sb.sampling_kwargs([rewind])
    ts = torch.linspace(1, 0, int(kw["num_steps"]) + 1)
    sigma = float(model.sampling_noise_schedule_default(ts[int(kw["num_steps"]) - rewind]))
    rows = []
    for b in range(pb["aat"].shape[0]):
        torch.manual_seed(seeds[b])
        n = int(pb["mask"][b].sum())   # noise drawn at the sample's own length, then zero-padded: identical alone and in a batch
        m = atom37_mask_from_aatype(pb["aat"][b:b + 1, :n], pb["mask"][b:b + 1, :n]) * pb["known"][b:b + 1, :n]
        x = dummy_fill_noise_coords(pb["pos"][b:b + 1, :n], m, noise_level=torch.tensor([sigma], device=pb["pos"].device), dummy_fill_mode=model.config.data.dummy_fill_mode)
        rows.append(torch.nn.functional.pad(x, (0, 0, 0, 0, 0, pb["mask"].shape[1] - n)))
    return torch.cat(rows)


def check_padding(model, items, rewind):
    pb = sb.pad_batch(items, "cuda")
    seeds = list(range(100, 100 + len(items)))
    xt = initial_state(model, pb, rewind, seeds)
    batch = sb.run_pd(model, pb, rewind, xt_start=xt)["xt_traj"][-1]
    worst = 0.0
    for b, it in enumerate(items):
        n = len(it[1])
        one = sb.pad_batch([it], "cuda")
        x1 = initial_state(model, one, rewind, [seeds[b]])
        alone = sb.run_pd(model, one, rewind, xt_start=x1)["xt_traj"][-1]
        d = float((batch[b, :n] - alone[0, :n]).abs().max())
        worst = max(worst, d)
        print(f"  sample {b} L'={n}: max |dx| batched-vs-alone = {d:.3e} A")
    print(f"padding: worst max |dx| = {worst:.3e} A over {len(items)} samples (lengths {[len(it[1]) for it in items]})")


def check_knownmask(model, items, rewind):
    pb = sb.pad_batch(items, "cuda")
    seed_everything(7)
    a = sb.run_pd(model, pb, rewind)["xt_traj"][-1]
    junk = pb["pos"].clone()
    unk = (pb["known"] == 0) & (pb["mask"][:, :, None] > 0)
    junk[unk] = torch.randn_like(junk)[unk] * 50.0
    pb2 = dict(pb, pos=junk)
    seed_everything(7)
    b = sb.run_pd(model, pb2, rewind)["xt_traj"][-1]
    print(f"knownmask: unknown-atom coordinates replaced by N(0, 50 A): max |dx| = {float((a - b).abs().max()):.3e} A ({int(unk.sum())} unknown atoms)")


def check_metrics(pool_npz, geom_csv, chain, native_pdb, n):
    g = pd.read_csv(geom_csv)
    g = g[g.chain == chain]
    z = np.load(pool_npz)
    worst = dict(cn_med=0.0, nca_med=0.0, cac_med=0.0, tm_native=0.0)
    for i in range(min(n, int(z["n_templates"]))):
        t = ip.read_template(pool_npz, i)
        bb = ip.atom37_coords(t)[:, sb.BB_IDX].astype(np.float64)
        seq = "".join(ip.AA_ORDER[int(x)] for x in t["aatype"])
        r = sb.score_one((bb, native_pdb, np.asarray(t["orig_idx"]), seq, np.zeros(t["L_native"] - 1, bool)))
        old = g[(g.draw == t["draw"]) & (g.set.str.endswith(t["arm"]))]
        assert len(old) == 1, (chain, t["draw"], t["arm"], len(old))
        for k in ("cn_med", "nca_med", "cac_med"):
            worst[k] = max(worst[k], abs(r[k] - float(old[k].iloc[0])))
        worst["tm_native"] = max(worst["tm_native"], abs(r["tm_native"] - t["tm_native"]))
    print(f"metrics over {min(n, int(z['n_templates']))} stored templates of {chain}: max |new - old| = {worst}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", required=True, choices=["inmemory", "padding", "knownmask", "metrics"])
    ap.add_argument("--stage-a-dir")
    ap.add_argument("--chain", required=True)
    ap.add_argument("--native-pdb", required=True)
    ap.add_argument("--rewind", type=int, default=250)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--pool-npz")
    ap.add_argument("--geom-csv")
    a = ap.parse_args()
    if a.check == "metrics":
        check_metrics(a.pool_npz, a.geom_csv, a.chain, a.native_pdb, a.n)
        return
    model = get_model()
    if a.check == "inmemory":
        check_inmemory(model, a.native_pdb, a.rewind)
        return
    items, _, _ = load_items(a.stage_a_dir, a.chain, a.native_pdb, a.n)
    {"padding": check_padding, "knownmask": check_knownmask}[a.check](model, items, a.rewind)


if __name__ == "__main__":
    main()
