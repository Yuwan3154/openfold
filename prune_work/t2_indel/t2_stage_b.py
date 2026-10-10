"""Stage B of the T2 template pipeline (GPU, protpardelle env): batched in-memory partial diffusion of the edited variants of a chain + the template filters.

Input = Stage A's <chain>.json (plans + ESMC-filled sequences) and the chain's native all-atom PDB. For every chain ALL draws go through ONE cc89 call (same rewind for all,
constant batch, padded to the longest variant; per-sample seq_mask / residue_index), instead of one PDB-file item per call:
  * the edited input of each draw is built in memory: unmutated survivors keep their native atom37 coordinates, inserted and mutated residues carry only the backbone
    (N, CA, C, O; the rigid insertion of indel_edit.edit) and their side chains are DUMMY-FILLED and noised by the sampler (protpardelle patch 0002, pd_inputs.known_mask) instead of
    being rebuilt by cg2all;
  * after the call each template gets: sequence-independent TM to the native (USalign default mode, normalised by the NATIVE length = the pool's tm_native), the geometry gate
    (median C-N, N-CA, CA-C inside the native envelope +- tol, user 10-09: 0.05 A) the DSSP loop gate (pydssp with the proline donor mask, T8 rule; coil fraction increase vs the native < 0.15, absolute, user 10-09), the backbone-break gate (at most one CA-CA step > 4.0 A
    where the native is continuous, T8 handoff) and the TM window gate (0.4-0.9, user 10-09). Refolding is NOT run (user 10-09: skip the refold filter).
Writes <out>/<chain>.npz (indel_pool layout, via pack_chain) and <out>/<chain>.metrics.csv. Env: protpardelle + mdtraj + USalign at ~/.local/bin/USalign.
Run: python t2_stage_b.py --stage-a-dir A --chains-file chains.tsv --out-dir O --geom-ref geom_pools.csv [--rewind 250 --chunk 64 --tol 0.05]
"""
import argparse
import json
import os
import sys
import tempfile
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

if "--fast" in sys.argv:   # read by protpardelle at import (patch 0003): fused attention + fp16 autocast of the denoiser
    os.environ["T2_SDPA"] = "1"
    os.environ["T2_AUTOCAST"] = "fp16"

import hydra
import numpy as np
import pandas as pd
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import indel_pool as ip
from indel_edit import edit
from protpardelle.common import residue_constants as rc
from protpardelle.core.models import load_model
from protpardelle.data.atom import atom37_mask_from_aatype
from protpardelle.data.pdb_io import load_feats_from_pdb
from protpardelle.env import MINIMPNN_WEIGHTS, PROTPARDELLE_MODEL_CONFIGS, PROTPARDELLE_MODEL_WEIGHTS, PROTPARDELLE_RUNNING_CONFIGS
from protpardelle.utils import apply_dotdict_recursively, seed_everything
from score_indel import usalign, write_ca_pdb
from t2_dssp import loop_fraction
from t2_graph import GraphedDenoiser
from t2_native import load_npz, native_ref, shard_path

MODEL_EPOCH = {"cc89": "415", "cc91": "383", "cc94": "3100"}
BB_IDX = [0, 1, 2, 4]                       # atom37 N, CA, C, O
CFG = "sampling_partial_diffusion_allatom"


def sampling_kwargs(rewinds, num_steps=None):
    """As prune_work/generate_templates.py:sampling_kwargs (ODE, no churn), with no PDB path (inputs come from pd_inputs)."""
    with initialize_config_dir(config_dir=str(PROTPARDELLE_RUNNING_CONFIGS), version_base="1.3.2"):
        c = compose(config_name=CFG)
    c = OmegaConf.to_container(hydra.utils.call(c), resolve=True)
    s = c["sampling"]
    s["step_scale"], s["s_churn"] = 1.0, 0
    s["conditional_cfg"]["crop_conditional_guidance"]["start"] = 0.0
    s["partial_diffusion"]["pdb_file_path"] = None
    if num_steps is not None:
        s["num_steps"] = num_steps   # total discretisation steps; rewind is the number of them taken (entry noise level = rewind / num_steps)
    s["partial_diffusion"]["num_steps"] = rewinds
    s["motif_file_path"] = "test_dir/empty.pdb"
    s.update(apply_dotdict_recursively(s.pop("allatom_cfg")))
    s.pop("stage2_cfg")
    return s


def build_inputs(nat_pos, nat_mask, nat_bb, plan):
    """In-memory edited input of one draw: positions (L',37,3) centred on the mean CA, aatype (L',), known mask (L',37), orig_idx, backbone (L',4,3)."""
    new_bb, orig, _ = edit(nat_bb, [tuple(o) for o in plan["ops"]])
    Lp = len(orig)
    aat = torch.tensor([rc.restype_order[c] for c in plan["seq"]], dtype=torch.long)
    full = atom37_mask_from_aatype(aat[None], torch.ones(1, Lp))[0]
    mutated = np.zeros(Lp, bool)
    mutated[plan["mut_idx"]] = True
    keep = (orig >= 0) & ~mutated                      # native residue type and atoms retained
    pos = torch.zeros(Lp, 37, 3)
    known = torch.zeros(Lp, 37)
    ki = torch.from_numpy(np.flatnonzero(keep))
    pos[ki] = nat_pos[torch.from_numpy(orig[keep])]
    known[ki] = full[ki] * nat_mask[torch.from_numpy(orig[keep])]   # native atoms that were not resolved stay unknown (dummy-filled by the sampler)
    oi = torch.from_numpy(np.flatnonzero(~keep))
    pos[oi[:, None], torch.tensor(BB_IDX)[None, :]] = torch.from_numpy(new_bb[~keep]).float()
    known[oi[:, None], torch.tensor(BB_IDX)[None, :]] = 1.0
    pos = pos - pos[:, 1].mean(0, keepdim=True)
    return pos, aat, known, orig, new_bb


def pad_batch(items, device, multiple=1):
    B = len(items)
    Lm = -(-max(len(it[1]) for it in items) // multiple) * multiple   # padding is exact-neutral (checked: padded batch == alone), so a length bucket is free
    pos, aat, known = torch.zeros(B, Lm, 37, 3), torch.zeros(B, Lm, dtype=torch.long), torch.zeros(B, Lm, 37)
    mask, ridx = torch.zeros(B, Lm), torch.zeros(B, Lm, dtype=torch.long)
    for b, (p, a, k, _, _) in enumerate(items):
        n = len(a)
        pos[b, :n], aat[b, :n], known[b, :n] = p, a, k
        mask[b, :n], ridx[b, :n] = 1, torch.arange(1, n + 1)
    return dict(pos=pos.to(device), aat=aat.to(device), known=known.to(device), mask=mask.to(device), ridx=ridx.to(device))


@torch.no_grad()
def run_pd(model, pb, rewind, xt_start=None, num_steps=None):
    B = pb["aat"].shape[0]
    return model.sample(seq_mask=pb["mask"], residue_index=pb["ridx"], chain_index=torch.zeros_like(pb["ridx"]), hotspots=None, sse_cond=None, adj_cond=None,
                        motif_placements_full=None, dummy_fill_mode=model.config.data.dummy_fill_mode, xt_start=xt_start,
                        pd_inputs=dict(aatype=pb["aat"], atom_positions=pb["pos"], known_mask=pb["known"]), **sampling_kwargs([rewind] * B, num_steps))


def bond_medians(bb):
    cn = np.linalg.norm(bb[1:, 0] - bb[:-1, 2], axis=1)
    return float(np.median(cn)), float(np.median(np.linalg.norm(bb[:, 1] - bb[:, 0], axis=1))), float(np.median(np.linalg.norm(bb[:, 2] - bb[:, 1], axis=1)))


def n_broken_steps(ca, orig, nat_break):
    """Template CA-CA steps > 4.0 A, except steps joining two native-adjacent survivors whose native step is itself broken (T8 handoff: a broken step is one where the native is continuous)."""
    d = np.linalg.norm(ca[1:] - ca[:-1], axis=1)
    adj = (orig[1:] >= 0) & (orig[:-1] >= 0) & (orig[1:] == orig[:-1] + 1)
    native_broken = np.zeros(len(d), bool)
    native_broken[adj] = nat_break[orig[:-1][adj]]
    return int(((d > 4.0) & ~native_broken).sum())


def score_one(task):
    """(bb (L,4,3), native pdb path, orig_idx, template sequence, native CA-CA break flags) -> tm_native, tm_template, bond medians, loop fraction, broken steps."""
    bb, nat_pdb, orig, seq, nat_break = task
    with tempfile.TemporaryDirectory() as td:
        tpl = os.path.join(td, "t.pdb")
        write_ca_pdb(tpl, bb[:, 1])
        tm1, tm2, _, _ = usalign(tpl, nat_pdb)
    cn, nca, cac = bond_medians(bb)
    return dict(tm_template=tm1, tm_native=tm2, cn_med=cn, nca_med=nca, cac_med=cac, loop_frac=loop_fraction(bb, np.array([c == "P" for c in seq])),
                n_broken=n_broken_steps(bb[:, 1], orig, nat_break))


def native_envelope(geom_csv, tol):
    n = pd.read_csv(geom_csv)
    n = n[n.set == "native"]
    assert len(n) >= 10, "native rows missing in the geometry reference"
    return {m: (n[m].min() - tol, n[m].max() + tol) for m in ("cn_med", "nca_med", "cac_med")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-a-dir", required=True)
    ap.add_argument("--chains-file", required=True, help="one chain id per line (natives from --natives-dir) or TSV chain<TAB>native.pdb")
    ap.add_argument("--natives-dir", default=None, help="t2_extract_natives.py output: <dir>/<id[1:3]>/<id>.npz")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--geom-ref", required=True, help="geom_pools.csv: its native rows give the bond-length envelope")
    ap.add_argument("--model", default="cc89", choices=sorted(MODEL_EPOCH))
    ap.add_argument("--rewind", type=int, default=250)
    ap.add_argument("--num-steps", type=int, default=None, help="total ODE steps of the schedule (default: the config value, 500); entry noise level = rewind / num-steps, so --num-steps 250 --rewind 125 = the same noise level with half the denoiser calls")
    ap.add_argument("--chunk", type=int, default=64, help="draws per sampler call")
    ap.add_argument("--span-cutoff", type=int, default=484)
    ap.add_argument("--tol", type=float, default=0.05, help="bond-length tolerance, A (user 10-09)")
    ap.add_argument("--loop-max", type=float, default=0.15, help="max coil-fraction increase vs the native, absolute (user 10-09)")
    ap.add_argument("--max-broken", type=int, default=1, help="max broken backbone steps (CA-CA > 4.0 A where the native is continuous; T8 handoff: more than one is dropped)")
    ap.add_argument("--tm-lo", type=float, default=0.4)
    ap.add_argument("--tm-hi", type=float, default=0.9)
    ap.add_argument("--store", default="all", choices=["all", "passing"], help="templates written to the npz: all 64 (metrics.csv flags the passing ones) or only those passing every gate; metrics.csv always has every row")
    ap.add_argument("--graph-per-chain", action="store_true", help="recapture the CUDA graphs for every chain (the denoiser's memoised relative-position / rotary tensors are constants of the graph and belong to the chain that captured it)")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--seed-offset", type=int, default=0, help="added to the per-chain noise seed (crc32 of the id): a second draw of the same edits, to measure the seed-to-seed spread")
    ap.add_argument("--fast", action="store_true", help="fused attention + fp16 autocast + torch.compile + CUDA-graph replay of the denoiser (RAW 140: 3.7-5.9x faster; coordinates deviate ~0.1 A from fp32)")
    ap.add_argument("--pad-multiple", type=int, default=None, help="pad the batch length to a multiple of this (default 16 with --fast, else 1): bounds the number of compiled/captured shapes")
    a = ap.parse_args()
    assert torch.cuda.is_available(), "stage B runs on the GPU"
    os.makedirs(a.out_dir, exist_ok=True)
    env = native_envelope(a.geom_ref, a.tol)
    pool = Pool(a.procs)   # forked before the model touches CUDA
    model = load_model(str(PROTPARDELLE_MODEL_CONFIGS / f"{a.model}.yaml"), str(PROTPARDELLE_MODEL_WEIGHTS / f"{a.model}_epoch{MODEL_EPOCH[a.model]}.pth"))
    model.load_minimpnn(MINIMPNN_WEIGHTS)
    mult = a.pad_multiple or (16 if a.fast else 1)
    if a.fast:
        torch._dynamo.config.cache_size_limit = 64   # one compiled graph per padded length bucket
        model.struct_model = GraphedDenoiser(torch.compile(model.struct_model, dynamic=False))
    last_bucket = None
    pending = None
    tot = dict(build=0.0, pd=0.0, score=0.0, wait=0.0)
    def finalize(job):
        if job is None:
            return
        chain, plans, items, kept, nat_bb, nat_loop, out = (job[k] for k in ("chain", "plans", "items", "kept", "nat_bb", "nat_loop", "out"))
        t2 = time.perf_counter()
        rows = job["rows"].get()
        job["tmp"].cleanup()
        tot["score"] += time.perf_counter() - t2   # only the wait that was NOT hidden behind the GPU
        coords, masks, seqs = job["coords"], job["masks"], job["seqs"]
        pool_items, recs = [], []
        for j, c, m, s, r in zip(kept, coords, masks, seqs, rows):
            p, it = plans[j], items[j]
            r["pass_tm"] = bool(a.tm_lo <= r["tm_native"] <= a.tm_hi)
            r["pass_bond"] = all(env[k][0] <= r[k] <= env[k][1] for k in env)
            r["loop_increase"] = r["loop_frac"] - nat_loop
            r["pass_loop"] = bool(r["loop_increase"] < a.loop_max)
            r["pass_break"] = r["n_broken"] <= a.max_broken
            r["pass_all"] = r["pass_tm"] and r["pass_bond"] and r["pass_loop"] and r["pass_break"]
            recs.append(dict(chain=chain, draw=p["draw"], L=len(s), L_native=len(nat_bb), n_ins=p["n_ins"], n_del=p["n_del"], n_mut=p["n_mut"], **r))
            if a.store == "passing" and not r["pass_all"]:
                continue
            pool_items.append(dict(arm="t2pipe", model=a.model, rewind=a.rewind, draw=p["draw"], coords=c[m].astype(np.float32), atom_mask=m, aatype=s, residue_index_orig=np.arange(1, len(s) + 1),
                                   orig_idx=it[3], ops=p["ops"], tm_native=r["tm_native"], tm_template=r["tm_template"], L_native=len(nat_bb)))
        if pool_items:   # a chain without survivors has no npz; its metrics.csv records why
            np.savez(out, **ip.pack_chain(chain, pool_items))
        csv = shard_path(a.out_dir, chain, ".metrics.csv")
        pd.DataFrame(recs).to_csv(csv + ".tmp", index=False)
        os.replace(csv + ".tmp", csv)
        print(f"{chain} L={len(nat_bb)} n={len(plans)} Lmax={max(len(c) for c in coords)} pd {job['pd_s']:.2f}s score-wait {time.perf_counter() - t2:.2f}s "
              f"pass_all {np.mean([x['pass_all'] for x in recs]):.2f} tm {np.mean([x['tm_native'] for x in recs]):.3f} peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)

    def prepare(ln):
        """CPU-only preparation of one chain (runs in a worker thread while the previous chain is on the GPU); None = skipped (recorded) or already done."""
        chain, nat_path = native_ref(ln, a.natives_dir)
        out = shard_path(a.out_dir, chain, ".npz")
        pa = shard_path(a.stage_a_dir, chain, ".json")
        if os.path.isfile(shard_path(a.out_dir, chain, ".metrics.csv")) or not os.path.isfile(pa):   # metrics.csv is written last = the chain-done marker
            return None
        os.makedirs(os.path.dirname(out), exist_ok=True)
        sa = json.load(open(pa))
        plans = [p for p in sa["plans"] if len(p["orig_idx"]) <= a.span_cutoff or a.model != "cc89"]
        if len(plans) < len(sa["plans"]):
            open(os.path.join(a.out_dir, "skipped.jsonl"), "a").write(json.dumps(dict(chain=chain, model=a.model, n_skipped=len(sa["plans"]) - len(plans), why="span > cutoff")) + "\n")
        if not plans:
            return None
        t0 = time.perf_counter()
        if nat_path.endswith(".npz"):
            nd = load_npz(nat_path)
            nat_pos, nat_mask, nat_bb, nat_complete = torch.from_numpy(nd["pos"]), torch.from_numpy(nd["mask"]).float(), nd["bb"], nd["complete"]
        else:
            feats, _ = load_feats_from_pdb(nat_path, include_pos_feats=True)
            nat_pos = feats["atom_positions"].float()
            nat_mask = (nat_pos.abs().sum(-1) > 0).float()
            nat_bb, nat_complete = nat_pos[:, BB_IDX].numpy().astype(np.float64), np.ones(len(nat_pos), bool)
        if len(nat_pos) != sa["L"]:
            open(os.path.join(a.out_dir, "skipped.jsonl"), "a").write(json.dumps(dict(chain=chain, why="native length differs between stage A and B", L_a=sa["L"], L_b=len(nat_pos))) + "\n")
            return None
        nat_is_pro = np.array([c == "P" for c in sa["names"]])
        nat_loop = loop_fraction(nat_bb, nat_is_pro, nat_complete)
        nat_break = np.linalg.norm(nat_bb[1:, 1] - nat_bb[:-1, 1], axis=1) > 4.0
        tmp = tempfile.TemporaryDirectory()
        nat_pdb = nat_path
        if nat_path.endswith(".npz"):
            nat_pdb = os.path.join(tmp.name, "native_ca.pdb")
            write_ca_pdb(nat_pdb, nat_bb[:, 1])
        items = [build_inputs(nat_pos, nat_mask, nat_bb, p) for p in plans]
        t1 = time.perf_counter()
        return dict(chain=chain, out=out, plans=plans, items=items, nat_bb=nat_bb, nat_loop=nat_loop, nat_break=nat_break, nat_pdb=nat_pdb, tmp=tmp, build_s=t1 - t0)

    chain_lines = list(open(a.chains_file))
    ex = ThreadPoolExecutor(max_workers=1)
    nxt = ex.submit(prepare, chain_lines[0]) if chain_lines else None
    for n_line in range(len(chain_lines)):
        tw = time.perf_counter()
        job = nxt.result()
        tot["wait"] += time.perf_counter() - tw   # GPU idle while the next chain is still being prepared
        nxt = ex.submit(prepare, chain_lines[n_line + 1]) if n_line + 1 < len(chain_lines) else None
        if job is None:
            continue
        chain, out, plans, items, nat_bb, nat_loop, nat_break, nat_pdb, tmp = (job[k] for k in ("chain", "out", "plans", "items", "nat_bb", "nat_loop", "nat_break", "nat_pdb", "tmp"))
        t1 = time.perf_counter()
        seed_everything((zlib.crc32(chain.encode()) + a.seed_offset) % 2**31)
        coords, masks, seqs, kept = [], [], [], []
        for i in range(0, len(items), a.chunk):
            sub = items[i:i + a.chunk]
            pb = pad_batch(sub, "cuda", mult)
            if a.fast and (pb["aat"].shape[1] != last_bucket or (a.graph_per_chain and i == 0)):   # graphs of a finished length bucket are dropped (compiled code is kept)
                model.struct_model.clear()
                last_bucket = pb["aat"].shape[1]
            aux = run_pd(model, pb, a.rewind, num_steps=a.num_steps)
            x, m, s = aux["xt_traj"][-1].numpy(), aux["atom_mask"].cpu().numpy().astype(bool), aux["s"].cpu().numpy()
            for b, it in enumerate(sub):
                n = len(it[1])
                if not (s[b, :n] == it[1].numpy()).all() or not np.isfinite(x[b, :n]).all():
                    open(os.path.join(a.out_dir, "skipped.jsonl"), "a").write(json.dumps(dict(chain=chain, draw=plans[i + b]["draw"], why="sequence changed or non-finite coordinates")) + "\n")
                    continue
                coords.append(x[b, :n])
                masks.append(m[b, :n])
                seqs.append(s[b, :n])
                kept.append(i + b)
        if not kept:   # every draw was recorded in skipped.jsonl above
            tmp.cleanup()
            continue
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        tot["build"] += job["build_s"]
        tot["pd"] += t2 - t1
        finalize(pending)   # the previous chain's scoring ran on the CPU pool while this chain was on the GPU
        pending = dict(chain=chain, plans=plans, items=items, kept=kept, coords=coords, masks=masks, seqs=seqs, nat_bb=nat_bb, nat_loop=nat_loop, tmp=tmp, out=out, pd_s=t2 - t1,
                       rows=pool.map_async(score_one, [(c[:, BB_IDX].astype(np.float64), nat_pdb, items[j][3], plans[j]["seq"], nat_break) for c, j in zip(coords, kept)]))
    finalize(pending)
    print(f"totals: build (overlapped with the GPU) {tot['build']:.1f}s, GPU idle waiting for the next chain {tot['wait']:.1f}s, pd {tot['pd']:.1f}s, score wait {tot['score']:.1f}s")


if __name__ == "__main__":
    main()
