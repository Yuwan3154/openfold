"""Noise-ladder ("temperature") sweep: how diverse are the samples, and at what cost in quality?

Answers the empirical question behind the replica-exchange Run C design (user directive 2026-08-19):
the ladder's rungs should be CHOSEN from measured diversity-vs-quality curves, not guessed.

⭐⭐ WHY EVAL MODE IS THE CLEAN EXPERIMENT, AND WHY THE LIVE RUN CANNOT ANSWER THIS.
`model.eval()` disables dropout, so `z_0` is the ONLY randomness in the forward and every bit of
observed spread is attributable to the injected noise. In TRAINING mode dropout also varies across the
K samples -- which is why `explore/loss_spread` from the live run measures noise+dropout together and
cannot isolate the noise level.

⛔⛔ EMA WEIGHTS, NOT `state_dict`. train_openfold.py:487-494 swaps the EMA params in at the start of
every validation epoch, so every val/lddt_ca this project has reported came from EMA weights. Use
prune_work/strip_ckpt-style inputs (`{"ema_params": ...}`); loading `state_dict` would measure a
different model than the curves describe.

⛔ A multiplicative noise scale is a NO-OP without --contractive_recycling: the plain-additive path
LayerNorms z_prev and LayerNorm is scale-invariant (measured: scale=4 and scale=100 deviate from
scale=1 by the same 7.8e-3, i.e. eps, not signal). This script therefore ASSERTS use_contractive.

⛔ V100 has no bf16 (compute capability 7.0), so this runs fp32. That is a different numeric path from
the bf16 training run -- deliberate and recorded, not an oversight.

⚠️ Features are built ONCE per target and reused across every (tau, seed). Besides being faster, this
sidesteps `random_crop_to_size` not being seed-reproducible: rebuilding per sample could hand different
taus different residue windows, and every TM below would then be comparing different problems.
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openfold.config import model_config
from openfold.model.model import AlphaFold
from openfold.utils.tm_score import REFERENCE_KWARGS, tm_score, tm_score_ca
from pda_dataset import PDASingleSeqDataset


def build_config(args):
    """Mirror train_openfold.py's config construction for the single-seq + tricks recipe.

    ⚠️ Deliberately NOT a refactor of train_openfold: the guard against divergence is the assertion
    block in `main` (state-dict keys must match the checkpoint EXACTLY, MSA depth must be 1, the extra
    track must be inert), which catches a wrong architecture or data config immediately instead of
    letting it produce plausible numbers.
    """
    config = model_config(args.config_preset, train=True, low_prec=False)
    config.data.common.max_recycling_iters = args.max_recycling_iters
    # --enable_single_seq_mode
    config.data.common.max_extra_msa = 1
    config.data.common.max_msa_clusters = 1
    config.data.train.max_extra_msa = 1
    config.data.train.max_msa_clusters = 1
    # ⛔⛔ TEMPLATES STAY ENABLED IN THE CONFIG, even for the template-free arm. The checkpoints were
    # trained with --single_seq_keep_templates, so they CONTAIN template_embedder weights (129
    # tensors) -- building the model with template.enabled=False omits those modules and the load
    # fails. Template-free evaluation is a RUNTIME gate, exactly as --validate_without_templates does
    # it (train_openfold.py:556-558): build with templates, load, THEN flip
    # model.config.template.enabled=False, which short-circuits the template branch at model.py:345.
    # ⭐ This mismatch is what the state-dict key assert caught on the first GPU run -- 0 missing,
    # 129 unexpected -- instead of it silently becoming a differently-architected model.
    # --contractive_recycling --gaussian_pair_init
    config.model.recycling_embedder.use_contractive = True
    config.model.recycling_embedder.use_gaussian_pair_init = True
    return config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ema_ckpt", required=True, help="output of strip_ckpt.py ({'ema_params': ...})")
    p.add_argument("--arch", choices=["stock", "ws5"], required=True)
    p.add_argument("--arm", choices=["pda_templatefree"], default="pda_templatefree")
    p.add_argument("--manifest", required=True)
    p.add_argument("--cif_dir", required=True)
    p.add_argument("--config_preset", default="finetuning_ptm")
    # ⛔ No defaults on the experimental knobs -- the sweep grid and the recycle count are choices,
    # and a silent default is exactly how an ungrounded value enters a result.
    p.add_argument("--taus", required=True, help="comma-separated noise scales, e.g. 0,0.25,0.5,1,2,4,8,16")
    p.add_argument("--seeds", type=int, required=True)
    p.add_argument("--max_recycling_iters", type=int, required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--n_shards", type=int, default=1)
    p.add_argument("--out", required=True)
    p.add_argument("--per_sample_out", default=None,
                   help="CSV with ONE ROW PER FORWARD (tau, seed, tm, ptm, plddt, pae) so selection "
                        "strategies can be simulated offline instead of re-running the model. Without "
                        "this only per-tau aggregates survive, and the oracle-vs-confidence question "
                        "cannot be answered from the output at all.")
    p.add_argument("--limit", type=int, default=0, help="smoke test: only this many targets")
    args = p.parse_args()

    taus = [float(x) for x in args.taus.split(",")]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    config = build_config(args)

    ck = torch.load(args.ema_ckpt, map_location="cpu", weights_only=False)
    assert "ema_params" in ck, f"{args.ema_ckpt} is not a stripped EMA checkpoint"
    # ⛔ AlphaFold takes the FULL config, not config.model: its __init__ reads BOTH `config.globals`
    # (model.py:81) and `config.model`. Passing config.model raises KeyError('globals') -- which is
    # exactly how the first smoke test died, after 6 minutes of imports.
    model = AlphaFold(config)
    if args.arch == "ws5":
        # ⛔⛔ --prune_evoformer is NOT a config flag -- it is structural surgery on the built model
        # (train_openfold.py:1046-1048 calls prune_blocks AFTER the weight load), which deletes
        # msa_att_col and replaces tri_att_start/end with no-ops. It must run BEFORE loading these
        # params, because train_openfold REBUILDS the EMA after pruning, so the stored EMA keys are
        # the PRUNED model's keys. Setting a config field instead would silently do nothing.
        from openfold.block_replacement_scripts.pruned_evoformer import prune_blocks
        prune_blocks(model.evoformer)
        print("arch=ws5: pruned the Evoformer (dropped column + triangle attention)")
    missing, unexpected = model.load_state_dict(ck["ema_params"], strict=False)
    # ⭐ The architecture guard: a wrong --arch or a missing trick shows up here as key drift rather
    # than as plausible-but-wrong numbers further down.
    assert not missing and not unexpected, (
        f"state-dict mismatch -- wrong --arch or trick flags?\n"
        f"  missing({len(missing)}): {list(missing)[:5]}\n"
        f"  unexpected({len(unexpected)}): {list(unexpected)[:5]}")
    assert config.model.recycling_embedder.use_contractive, "a noise scale is a no-op without contractive"
    if args.arm == "pda_templatefree":
        # the runtime gate, applied AFTER the load -- see build_config's note
        model.config.template.enabled = False
        print("template-free arm: model.config.template.enabled = False (post-load runtime gate)")
    model = model.to(dev).eval()
    print(f"loaded {args.ema_ckpt} (epoch={ck.get('epoch')}) arch={args.arch} on {dev}, "
          f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    ds = PDASingleSeqDataset(args.manifest, args.cif_dir, config.data, mode="eval")
    manifest = json.load(open(args.manifest))
    idx = list(range(len(ds)))[args.shard::args.n_shards]
    if args.limit:
        idx = idx[: args.limit]
    print(f"{len(idx)} targets (shard {args.shard}/{args.n_shards}), "
          f"{len(taus)} taus x {args.seeds} seeds = {len(idx)*len(taus)*args.seeds} forwards")

    fh = open(args.out, "w", newline="")
    w = csv.writer(fh)
    ps_fh = ps_w = None
    if args.per_sample_out:
        ps_fh = open(args.per_sample_out, "w", newline="")
        ps_w = csv.writer(ps_fh)
        ps_w.writerow(["pdb", "chain", "length", "stock_fail", "tau", "seed",
                       "tm_native", "ptm", "plddt_mean", "pae_mean"])
    w.writerow(["pdb", "chain", "length", "stock_fail", "tau", "n_samples",
                "mean_pairwise_tm", "mean_tm_native", "best_tm_native", "oracle_gain",
                "mean_ptm", "ptm_tm_spearman_within", "sec_per_forward", "peak_mem_gb"])

    for t_i, i in enumerate(idx):
        entry = manifest[i]
        feats = ds[i]
        batch = {k: (v.unsqueeze(0).to(dev) if torch.is_tensor(v) else v)
                 for k, v in feats.items()}
        L = int(batch["seq_mask"][..., -1].sum().item())

        # one-time feature sanity, on the REAL batch rather than on assumptions
        if t_i == 0:
            # ⛔⛔ CHECK THE MASK, NOT THE SHAPE. `make_fixed_size` PADS the MSA axis out to
            # max_msa_clusters (128 here) with zeros and marks the padding in `msa_mask`, so the
            # tensor is 128 deep even for a genuinely single-sequence input -- depth is not a bug.
            # My first version asserted msa_feat.shape[-3] == 1, which is wrong twice over: that axis
            # is N_RES, not the MSA depth, and the depth would legitimately be 128 anyway. This is the
            # same padding-blindness that produced 12 phantom "violations" in the Run B feature audit.
            _mm = batch["msa_mask"][..., -1] if batch["msa_mask"].dim() == 4 else batch["msa_mask"]
            _depth = float(_mm.sum(dim=-2).max())        # real (unmasked) rows, worst residue
            assert _depth == 1.0, f"expected a single-sequence MSA, got depth {_depth}"
            if "extra_msa_mask" in batch:
                assert float(batch["extra_msa_mask"].max()) == 0.0, "extra track is not inert"
            print(f"  msa depth (unmasked rows) = {_depth}; msa_feat {tuple(batch['msa_feat'].shape)}")
            for k, v in batch.items():
                if torch.is_tensor(v) and v.dtype.is_floating_point:
                    assert torch.isfinite(v).all(), f"non-finite feature: {k}"
            print(f"  feature check OK (msa depth 1, extra track inert, all finite); L={L}")

        native = batch["all_atom_positions"][..., -1] if batch["all_atom_positions"].dim() == 5 \
            else batch["all_atom_positions"]
        nat_mask = batch["all_atom_mask"][..., -1] if batch["all_atom_mask"].dim() == 4 \
            else batch["all_atom_mask"]

        for tau in taus:
            config.model.recycling_embedder.gaussian_pair_init_scale = tau
            preds, ptms, confs = [], [], []
            # cost instrumentation: peak ALLOCATED bytes is a function of shapes/dtypes, not of the
            # device, so these numbers transfer to any GPU that can hold them.
            if dev == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            t_start = time.perf_counter()
            with torch.no_grad():
                for s in range(args.seeds):
                    torch.manual_seed(1000 * s + 7)
                    if dev == "cuda":
                        torch.cuda.manual_seed_all(1000 * s + 7)
                    out = model(batch)
                    preds.append(out["final_atom_positions"].float())
                    ptms.append(float(out["ptm_score"].mean()) if "ptm_score" in out else float("nan"))
                    if ps_w is not None:
                        # ⛔ MASKED reductions: plddt is [*, N_res] and pae [*, N_res, N_res], both
                        # padded out to the fixed size, so an unmasked mean averages in the padding.
                        _sm = batch["seq_mask"][..., -1] if batch["seq_mask"].dim() == 3 \
                            else batch["seq_mask"]
                        _n = _sm.sum().clamp(min=1)
                        _pl = float("nan")
                        if "plddt" in out:
                            _pl = float((out["plddt"].float() * _sm).sum() / _n)
                        _pae = float("nan")
                        if "predicted_aligned_error" in out:
                            _pm = _sm[..., :, None] * _sm[..., None, :]
                            _pae = float((out["predicted_aligned_error"].float() * _pm).sum()
                                         / _pm.sum().clamp(min=1))
                        confs.append((_pl, _pae))

            if dev == "cuda":
                torch.cuda.synchronize()
            sec_fwd = (time.perf_counter() - t_start) / max(1, args.seeds)
            peak_gb = (torch.cuda.max_memory_allocated() / 2**30) if dev == "cuda" else float("nan")

            tm_nat = [float(tm_score_ca(p, native, nat_mask, **REFERENCE_KWARGS).mean())
                      for p in preds]
            if ps_w is not None:
                for _s, (_tm, _pt) in enumerate(zip(tm_nat, ptms)):
                    _pl, _pae = confs[_s] if _s < len(confs) else (float("nan"), float("nan"))
                    ps_w.writerow([entry["pdb"], entry["chain_id"], L, entry.get("stock_fail"),
                                   tau, _s, round(_tm, 6), round(_pt, 6),
                                   round(_pl, 6), round(_pae, 6)])
                ps_fh.flush()
            # sample-vs-sample: same chain, same numbering, so correspondence is fixed and the
            # normalisation is the shared coverage (there is no "native" in this pair)
            sm = batch["seq_mask"][..., -1] if batch["seq_mask"].dim() == 3 else batch["seq_mask"]
            pw = [float(tm_score(preds[a][:, :, 1, :], preds[b][:, :, 1, :], mask=sm,
                                 norm_mask=sm, **REFERENCE_KWARGS).mean())
                  for a in range(len(preds)) for b in range(a + 1, len(preds))]

            fin = [(x, y) for x, y in zip(ptms, tm_nat) if np.isfinite(x)]
            rho = float("nan")
            if len(fin) >= 3 and len(set(x for x, _ in fin)) > 1:
                from scipy.stats import spearmanr
                rho = float(spearmanr([x for x, _ in fin], [y for _, y in fin]).statistic)

            w.writerow([entry["pdb"], entry["chain_id"], L, entry.get("stock_fail"), tau,
                        len(preds), round(float(np.mean(pw)), 5) if pw else "",
                        round(float(np.mean(tm_nat)), 5), round(float(max(tm_nat)), 5),
                        round(float(max(tm_nat) - np.mean(tm_nat)), 5),
                        round(float(np.nanmean(ptms)), 5), round(rho, 5),
                        round(sec_fwd, 4), round(peak_gb, 3)])
            fh.flush()
        print(f"  [{t_i+1}/{len(idx)}] {entry['pdb']}_{entry['chain_id']} L={L} done")
    fh.close()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
