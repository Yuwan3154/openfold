"""Refinement variants for the indel templates (user 10-06: more local / more global Langevin refinement).

--variant forms (all cc89, ODE unless stated; the production sampler is step_scale 1.0, s_churn 0):
  refine:r=<rewind>:n=<cycles>  starting from an EXISTING output (--base-root, rung taken from its rewind_steps),
                                re-noise to rewind r (sigma from the measured table) and denoise, n times
                                = predictor-corrector refinement at that noise level (low r = local, high r = global)
  fresh:churn=<s_churn>:recur=<k>  a new partial diffusion from the edited INPUT at the rung with the sampler's own
                                stochastic churn (gamma = s_churn/num_steps) and k self-recurrence passes per step
Output mirrors the sweep layout: <out-root>/<variant-tag>/<chain>/dNN.npz with coords for each --rungs entry.
Resumable (existing npz skipped). Env: protpardelle on a GPU node.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from generate_templates import MODEL_EPOCH, sampling_kwargs  # noqa: E402
from make_indel_inputs import ATOM37, write_pdb  # noqa: E402
from protpardelle.common import residue_constants as rc  # noqa: E402
from protpardelle.core.models import load_model  # noqa: E402
from protpardelle.data.pdb_io import load_feats_from_pdb  # noqa: E402
from protpardelle.env import MINIMPNN_WEIGHTS, PROTPARDELLE_MODEL_CONFIGS, PROTPARDELLE_MODEL_WEIGHTS  # noqa: E402
from protpardelle.utils import seed_everything  # noqa: E402

ONE2THREE = {"A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN", "E": "GLU", "G": "GLY",
             "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO", "S": "SER",
             "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL"}


def parse(variant):
    kind, *kv = variant.split(":")
    return kind, {k: float(v) for k, v in (x.split("=") for x in kv)}


def sample_once(model, pdb, rewind, churn=0.0, recur=1):
    feats, _ = load_feats_from_pdb(pdb, include_pos_feats=True)
    ridx = torch.tile(feats["residue_index"][None], (1, 1)).cuda()
    cidx = torch.tile(feats["chain_index"][None], (1, 1)).cuda()
    kw = sampling_kwargs(pdb, int(rewind))
    kw["s_churn"] = churn
    kw["conditional_cfg"]["num_recurrence_steps"] = int(recur)
    with torch.no_grad():
        aux = model.sample(seq_mask=torch.ones_like(ridx).cuda(), residue_index=ridx, chain_index=cidx, hotspots=None,
                           sse_cond=None, adj_cond=None, motif_placements_full=None,
                           dummy_fill_mode=model.config.data.dummy_fill_mode, **kw)
    return (aux["xt_traj"][-1].numpy().astype(np.float32)[0], aux["atom_mask"][0].cpu().numpy().astype(bool),
            aux["s"][0].cpu().numpy().astype(np.int8), feats["residue_index_orig"].numpy().astype(np.int32))


def write_from(path, coords, atom_mask, aatype):
    resnames = [ONE2THREE[rc.restypes[int(a)]] for a in aatype]
    write_pdb(path, resnames, coords.astype(np.float64), atom_mask)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs-dir", required=True)
    p.add_argument("--out-root", required=True)
    p.add_argument("--variant", required=True)
    p.add_argument("--base-root", default=None, help="sweep root holding <model>/<chain>/dNN.npz (refine variants)")
    p.add_argument("--base-model", default="cc89")
    p.add_argument("--rungs", type=int, nargs="+", default=[300, 250])
    p.add_argument("--chains", nargs="+", required=True)
    p.add_argument("--draws", type=int, nargs="+", required=True)
    p.add_argument("--tmp-dir", required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    a = p.parse_args()
    kind, par = parse(a.variant)
    tag = a.variant.replace(":", "_").replace("=", "")
    model = load_model(str(PROTPARDELLE_MODEL_CONFIGS / "cc89.yaml"),
                       str(PROTPARDELLE_MODEL_WEIGHTS / f"cc89_epoch{MODEL_EPOCH['cc89']}.pth"))
    model.load_minimpnn(MINIMPNN_WEIGHTS)
    os.makedirs(a.tmp_dir, exist_ok=True)
    jobs = [(c, k) for c in a.chains for k in a.draws][a.shard::a.num_shards]
    t0, done = time.perf_counter(), 0
    for key, k in jobs:
        out = Path(a.out_root) / tag / key / f"d{k:02d}.npz"
        if out.is_file():
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        pdb = os.path.join(a.inputs_dir, key, f"d{k:02d}.pdb")
        seed = (sum(map(ord, key)) * 1000 + k) % 2**31
        coords_all = []
        for r in a.rungs:
            seed_everything(seed + r)
            if kind == "fresh":
                c, am, aa, ri = sample_once(model, pdb, r, churn=par.get("churn", 0.0), recur=par.get("recur", 1))
            else:
                z = np.load(os.path.join(a.base_root, a.base_model, key, f"d{k:02d}.npz"))
                i = z["rewind_steps"].tolist().index(r)
                mask = z["atom_mask"]
                full = np.zeros((mask.size, 3), np.float32)
                full[mask.reshape(-1)] = z["coords"][i]
                c, am, aa, ri = full.reshape(mask.shape[0], 37, 3), mask, z["aatype"], z["residue_index"]
                for cyc in range(int(par["n"])):
                    tmp = os.path.join(a.tmp_dir, f"{tag}_{key}_{k}_{r}_{cyc}.pdb")
                    write_from(tmp, c, am, aa)
                    c, am, aa, ri = sample_once(model, tmp, par["r"])
                    os.remove(tmp)
            coords_all.append(c.reshape(-1, 3)[am.reshape(-1)])
        np.savez(out, coords=np.stack(coords_all), atom_mask=am, aatype=aa, residue_index=ri,
                 rewind_steps=np.asarray(a.rungs, np.int16), L_new=np.int32(am.shape[0]), model=tag)
        done += 1
        print(f"{tag} {key} d{k:02d} done={done}/{len(jobs)} {time.perf_counter() - t0:.0f}s", flush=True)
    print(f"finished {tag}: {done} written", flush=True)


if __name__ == "__main__":
    main()
