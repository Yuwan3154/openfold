"""Stage A of the T2 template pipeline (GPU, esmfold2 env): edit plans + ESMC fill/mutation for MANY chains in ONE process, batched, no per-draw files.

Replaces make_mild_inputs.py (plans) + esmc_fill_mut.py (fill) of the file-per-draw route, whose cost was the CPU ESMC forward (xeon-p8 node, no GPU) and the per-file
launches around it. Logic is imported, not re-implemented: draw_plan / edit / sample_positions are the same functions, so the plans (ops, orig_idx, mutation sites) and the
sampling given the logits are identical to the old route; only the ESMC forward is batched (padded token matrix, pad-masked by the model) and run on the GPU.
Per chain it writes <out>/<id[1:3]>/<chain>.json = {chain, L, names, plans: [{draw, ops, orig_idx, mut_idx, seq, n_ins, n_del, n_mut}]}. A chain whose edit fractions round to zero
residues (too short) is SKIPPED AND RECORDED in <out>/skipped.jsonl.
Edit sizes: insertion, deletion and point-mutation fractions each U(frac_lo, frac_hi) of L (user 10-08: 5-10 %), ESMC T 0.7, top-p 1.0, native residue excluded at mutated sites.
Run: python t2_stage_a.py --chains-file chains.tsv(chain<TAB>native.pdb) --out-dir D --n-draws 64 [--verify-dir inputs_mf --verify-seqs mf_seqs_300m.json]
"""
import argparse
import json
import os
import time
import zlib

import numpy as np
import torch
from esm.models.esmc import ESMC
from esm.tokenization import get_esmc_model_tokenizers

from esmc_fill_pilot import AA, THREE2ONE, sample_positions
from indel_edit import edit
from regen_gly_inputs import backbone, native_residues
from sample_indels import draw_plan
from t2_native import STANDARD, load_npz, native_ref, shard_path


ESMC_DIMS = {"esmc_300m": (960, 15, 30), "esmc_600m": (1152, 18, 36)}


def load_esmc(name, dev, pth):
    """from_pretrained, or (--esmc-pth) the legacy .pth checkpoint when the installed esm cannot read the hub layout (Engaging's esm 3.3.0)."""
    if pth is None:
        return ESMC.from_pretrained(name, device=torch.device(dev)).eval()
    d, h, n = ESMC_DIMS[name]
    model = ESMC(d_model=d, n_heads=h, n_layers=n, tokenizer=get_esmc_model_tokenizers(), use_flash_attn=False).eval()   # fp32 without flash attention: the same numerics as the old CPU route
    model.load_state_dict(torch.load(pth, map_location="cpu"))
    return model.to(dev)


def load_native(pdb):
    if pdb.endswith(".npz"):
        d = load_npz(pdb)
        return d["bb"], d["names"]
    res = native_residues(pdb)
    names = "".join(THREE2ONE[res[i][0][17:20].strip()] for i in sorted(res))
    return backbone(res), names


def make_plans(chain, nat_bb, n_draws, seed, frac_lo, frac_hi, mut_lo, mut_hi):
    """Same sampling as make_mild_inputs.py (draw_plan + the mutation draw rng [seed, crc32(chain), draw, 5])."""
    L = len(nat_bb)
    plans = []
    for k in range(n_draws):
        rec = draw_plan(L, chain, k, seed, frac_lo, frac_hi)
        if rec["ins"]["T"] < 1 or rec["del"]["T"] < 1:
            return None
        _, orig, _ = edit(nat_bb, [tuple(o) for o in rec["ops"]])
        rng_m = np.random.default_rng([seed, zlib.crc32(chain.encode()), k, 5])
        n_mut = int(round(float(rng_m.uniform(mut_lo, mut_hi)) * L))
        surv = np.flatnonzero(orig >= 0)
        if not 1 <= n_mut <= len(surv):
            return None
        mut = sorted(rng_m.choice(surv, size=n_mut, replace=False).tolist())
        plans.append(dict(draw=k, ops=rec["ops"], orig_idx=orig.tolist(), mut_idx=mut, n_ins=int((orig < 0).sum()), n_del=int(L - (orig >= 0).sum()), n_mut=len(mut)))
    return plans


@torch.no_grad()
def esmc_logits(model, tok, ids_of, names, plans, dev, batch_size):
    """One masked forward per plan, padded and batched; returns a list of (L', 20) fp32 logits (AA order) on the CPU."""
    seqs = []
    for p in plans:
        orig = np.asarray(p["orig_idx"])
        masked = set(np.union1d(np.flatnonzero(orig < 0), np.asarray(p["mut_idx"], dtype=int)).tolist())
        seqs.append([tok.cls_token_id] + [tok.mask_token_id if j in masked else ids_of[names[o]] for j, o in enumerate(orig)] + [tok.eos_token_id])
    cols = [ids_of[x] for x in AA]
    out = []
    for i in range(0, len(seqs), batch_size):
        chunk = seqs[i:i + batch_size]
        w = max(len(s) for s in chunk)
        ids = torch.full((len(chunk), w), tok.pad_token_id, dtype=torch.long)
        for r, s in enumerate(chunk):
            ids[r, :len(s)] = torch.tensor(s)
        lg = model(sequence_tokens=ids.to(dev)).sequence_logits.float()[:, :, cols].cpu()
        out.extend(lg[r, 1:len(s) - 1] for r, s in enumerate(chunk))
    return out


def sample_sequences(names, chain, plans, logits, temp, top_p):
    seqs = []
    for p, lg in zip(plans, logits):
        orig = np.asarray(p["orig_idx"])
        mut = np.asarray(p["mut_idx"], dtype=int)
        lg = lg.clone()
        for j in mut:
            lg[j, AA.index(names[orig[j]])] = -float("inf")
        masked = np.union1d(np.flatnonzero(orig < 0), mut)
        rng = np.random.default_rng([zlib.crc32(chain.encode()), p["draw"], 93])
        pick = sample_positions(lg[masked], temp, top_p, rng)
        seq = [names[o] if o >= 0 else None for o in orig]
        for j, i in zip(masked, pick):
            seq[j] = AA[i]
        assert all(s is not None for s in seq) and all(seq[j] != names[orig[j]] for j in mut), (chain, p["draw"])
        seqs.append("".join(seq))
    return seqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains-file", required=True, help="one chain id per line (natives from --natives-dir) or TSV chain<TAB>native.pdb")
    ap.add_argument("--natives-dir", default=None, help="t2_extract_natives.py output: <dir>/<id[1:3]>/<id>.npz")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-draws", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frac-lo", type=float, default=0.05)
    ap.add_argument("--frac-hi", type=float, default=0.10)
    ap.add_argument("--mut-lo", type=float, default=0.05)
    ap.add_argument("--mut-hi", type=float, default=0.10)
    ap.add_argument("--esmc", default="esmc_300m", choices=["esmc_300m", "esmc_600m"])
    ap.add_argument("--esmc-pth", default=None, help="legacy esmc_*.pth checkpoint (use when from_pretrained cannot read the hub layout)")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--verify-dir", default=None, help="old-route inputs dir (plans.json per chain): plans must match")
    ap.add_argument("--verify-seqs", default=None, help="old-route esmc_fill_mut.py JSON: report the share of identical sequences")
    a = ap.parse_args()
    dev = "cuda"
    assert torch.cuda.is_available(), "stage A runs on the GPU (user 10-09: not on CPU)"
    os.makedirs(a.out_dir, exist_ok=True)
    model = load_esmc(a.esmc, dev, a.esmc_pth)
    tok = model.tokenizer
    ids_of = {c: tok.convert_tokens_to_ids(c) for c in AA}
    old_seqs = {(r["chain"], r["draw"]): r["seq"] for r in json.load(open(a.verify_seqs))} if a.verify_seqs else {}
    n_same = n_tot = 0
    t_plan = t_fwd = t_samp = 0.0
    for ln in open(a.chains_file):
        chain, pdb = native_ref(ln, a.natives_dir)
        out = shard_path(a.out_dir, chain, ".json")
        if os.path.isfile(out):
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        t0 = time.perf_counter()
        nat_bb, names = load_native(pdb)
        if set(names) - STANDARD:
            open(os.path.join(a.out_dir, "skipped.jsonl"), "a").write(json.dumps(dict(chain=chain, L=len(nat_bb), why="non-standard residue type in the native sequence")) + "\n")
            continue
        plans = make_plans(chain, nat_bb, a.n_draws, a.seed, a.frac_lo, a.frac_hi, a.mut_lo, a.mut_hi)
        if plans is None:
            open(os.path.join(a.out_dir, "skipped.jsonl"), "a").write(json.dumps(dict(chain=chain, L=len(nat_bb), why="edit fraction rounds to zero residues")) + "\n")
            continue
        if a.verify_dir:
            old = json.load(open(os.path.join(a.verify_dir, chain, "plans.json")))["plans"]
            for p in plans[:len(old)]:
                assert p["orig_idx"] == old[p["draw"]]["orig_idx"] and p["ops"] == old[p["draw"]]["ops"] and p["mut_idx"] == old[p["draw"]]["mut"]["new_idx"], (chain, p["draw"], "plan differs from the old route")
        t1 = time.perf_counter()
        logits = esmc_logits(model, tok, ids_of, names, plans, dev, a.batch_size)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        for p, s in zip(plans, sample_sequences(names, chain, plans, logits, a.temp, a.top_p)):
            p["seq"] = s
            if (chain, p["draw"]) in old_seqs:
                n_tot += 1
                n_same += old_seqs[(chain, p["draw"])] == s
        t3 = time.perf_counter()
        json.dump(dict(chain=chain, L=len(nat_bb), names=names, plans=plans), open(out, "w"))
        t_plan, t_fwd, t_samp = t_plan + t1 - t0, t_fwd + t2 - t1, t_samp + t3 - t2
        print(f"{chain} L={len(nat_bb)} draws={len(plans)} plan {t1 - t0:.2f}s esmc {t2 - t1:.2f}s sample {t3 - t2:.2f}s", flush=True)
    print(f"totals: plan {t_plan:.1f}s, esmc forward {t_fwd:.1f}s, sampling {t_samp:.1f}s")
    if a.verify_seqs:
        assert n_tot > 0, "no overlapping (chain, draw) with --verify-seqs"
        print(f"identical to the old route: {n_same} of {n_tot} sequences")


if __name__ == "__main__":
    main()
