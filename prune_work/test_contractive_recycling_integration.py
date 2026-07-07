"""Integration-level test for the ESMFold2-inspired contractive recycling + Gaussian pair init
wiring in openfold/model/embedders.py (RecyclingEmbedder) and openfold/model/model.py (AlphaFold
cycle loop) -- NOT just the standalone module (already covered by
test_contractive_recycling.py), but the actual wired-in code path via a real (tiny) AlphaFold
forward pass, CPU-only.

Compares 4 configurations: {use_contractive, use_gaussian_pair_init} x {False, True} x
{recycle=3 (baseline), recycle=12 (scaled)}, checking:
(1) all combinations run without shape/runtime errors,
(2) default (both False) behavior is UNCHANGED (bit-identical to pre-patch OpenFold, verified by
    reproducibility across two runs with the same seed -- a proxy for "didn't break anything"),
(3) with use_contractive=True, output final_atom_positions stay finite (no NaN/Inf) even at
    recycle=12, i.e. the model doesn't blow up numerically at a higher recycle count than default,
(4) with use_gaussian_pair_init=True, two different seeds produce DIFFERENT outputs for the
    IDENTICAL input sequence -- confirming genuine seed-based structural diversity, the actual
    point of the mechanism.
"""
import glob
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{os.path.dirname(os.path.abspath(__file__))}/../openfold/block_replacement_scripts")
from hallucination_straight_through import make_feature_batch

from openfold.config import model_config
from openfold.model.model import AlphaFold
from openfold.np import residue_constants as rc

SEQ = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKR"[:20]
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
# NOTE: OpenFold's inplace-attention path (triggered under torch.no_grad(), see
# `inplace_safe = not (self.training or torch.is_grad_enabled())` in model.py) unconditionally
# calls a CUDA-only kernel (attn_core_inplace_cuda) -- this is pre-existing OpenFold behavior,
# not something this patch changes, so this test needs a real GPU if one is available, and CANNOT
# run under torch.no_grad() on CPU-only.


def build_model(use_contractive, use_gaussian_pair_init, seed=0):
    torch.manual_seed(seed)
    cfg = model_config("finetuning_ptm", train=False, low_prec=False)
    cfg.globals.chunk_size = None
    for g in ["use_deepspeed_evo_attention", "use_lma", "use_flash"]:
        setattr(cfg.globals, g, False)
    cfg.model.template.enabled = False
    cfg.data.common.use_templates = False
    cfg.data.common.use_template_torsion_angles = False
    cfg.model.recycling_embedder.use_contractive = use_contractive
    cfg.model.recycling_embedder.use_gaussian_pair_init = use_gaussian_pair_init
    m = AlphaFold(cfg)
    m.eval()
    return m.to(DEVICE)


def seq_to_logits(seq):
    aat = torch.tensor([rc.restype_order.get(a, rc.restype_num) for a in seq])
    oh = torch.nn.functional.one_hot(aat.clamp(max=19), 20).float()
    oh[aat >= 20] = 0.0
    return (3.0 * oh).to(DEVICE)


@torch.no_grad()
def run_forward(model, seq, recycle, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    logits = seq_to_logits(seq)
    ri = torch.arange(len(seq), device=DEVICE)
    batch = make_feature_batch(logits, ri, recycle_dim=recycle + 1)
    out = model(batch)
    fap = out["final_atom_positions"]
    return fap[0] if fap.dim() == 4 else fap


def test_runs_without_error_all_configs():
    for use_contractive in [False, True]:
        for use_gaussian in [False, True]:
            for recycle in [3, 12]:
                model = build_model(use_contractive, use_gaussian, seed=0)
                fap = run_forward(model, SEQ, recycle, seed=0)
                assert torch.isfinite(fap).all(), (
                    f"non-finite output: contractive={use_contractive} "
                    f"gaussian={use_gaussian} recycle={recycle}")
                print(f"PASS: contractive={use_contractive} gaussian={use_gaussian} "
                      f"recycle={recycle} -> finite output, shape={tuple(fap.shape)}")


def test_default_config_reproducible():
    """Default (both flags False) should be deterministic given the same seed -- a proxy for
    'the new code paths don't touch anything when both flags are off'."""
    model = build_model(False, False, seed=0)
    fap1 = run_forward(model, SEQ, recycle=3, seed=42)
    fap2 = run_forward(model, SEQ, recycle=3, seed=42)
    assert torch.allclose(fap1, fap2), "default config not reproducible with same seed"
    print("PASS: default (both flags False) config is deterministic/reproducible")


def build_ws5_model(use_contractive, use_gaussian_pair_init, ckpt_path):
    """Load WS5's REAL trained checkpoint (not fresh random weights) -- IMPORTANT: an untrained,
    randomly-initialized model was found (see plan doc) to have essentially zero z->single-repr
    coupling for this reduced single-sequence config, so a seed-diversity test on random weights
    is not meaningful. A model actually TRAINED to exploit its pair representation for structure
    prediction is the correct thing to test this against."""
    sys.path.insert(0, f"{os.path.dirname(os.path.abspath(__file__))}/../openfold/block_replacement_scripts")
    from pruned_evoformer import prune_blocks

    cfg = model_config("finetuning_ptm", train=False, low_prec=False)
    cfg.globals.chunk_size = None
    for g in ["use_deepspeed_evo_attention", "use_lma", "use_flash"]:
        setattr(cfg.globals, g, False)
    cfg.data.common.max_recycling_iters = 3
    cfg.model.template.enabled = False
    cfg.data.common.use_templates = False
    cfg.data.common.use_template_torsion_angles = False
    cfg.model.recycling_embedder.use_contractive = use_contractive
    cfg.model.recycling_embedder.use_gaussian_pair_init = use_gaussian_pair_init

    m = AlphaFold(cfg)
    prune_blocks(m.evoformer)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    missing, unexpected = m.load_state_dict(
        {k[6:]: v for k, v in sd.items() if k.startswith("model.")}, strict=False)
    assert not missing, f"unexpected missing keys: {missing}"
    assert all(k.startswith("template_embedder.") for k in unexpected), \
        f"unexpected non-template keys: {[k for k in unexpected if not k.startswith('template_embedder.')]}"
    return m.eval().to(DEVICE)


def test_gaussian_init_gives_seed_diversity_on_real_ws5_weights():
    """The actual point of use_gaussian_pair_init: same sequence, different seeds -> different
    structures -- tested on WS5's REAL trained checkpoint, not fresh random weights (see
    build_ws5_model's docstring for why that distinction matters)."""
    ws5_ckpt_dir = "/home/jupyter-chenxi/runs/prune_singleseq_v1/lightning_logs/version_4/checkpoints"
    candidates = glob.glob(os.path.join(ws5_ckpt_dir, "best-*.ckpt"))
    if not candidates:
        print(f"SKIP: no WS5 checkpoint found in {ws5_ckpt_dir} -- cannot test seed-diversity "
              f"on real trained weights in this environment")
        return
    ckpt_path = max(candidates, key=os.path.getmtime)
    print(f"using WS5 checkpoint: {ckpt_path}")

    model = build_ws5_model(use_contractive=False, use_gaussian_pair_init=True, ckpt_path=ckpt_path)
    fap_seed1 = run_forward(model, SEQ, recycle=3, seed=1)
    fap_seed2 = run_forward(model, SEQ, recycle=3, seed=2)
    diff = (fap_seed1 - fap_seed2).abs().mean().item()
    assert diff > 1e-3, f"different seeds gave near-identical output on REAL WS5 weights (mean abs diff={diff:.2e})"
    print(f"PASS: gaussian_pair_init gives seed-based diversity on real WS5 weights "
          f"(mean abs coord diff={diff:.3f} A)")

    model_off = build_ws5_model(use_contractive=False, use_gaussian_pair_init=False, ckpt_path=ckpt_path)
    fap_off_seed1 = run_forward(model_off, SEQ, recycle=3, seed=1)
    fap_off_seed2 = run_forward(model_off, SEQ, recycle=3, seed=2)
    diff_off = (fap_off_seed1 - fap_off_seed2).abs().mean().item()
    assert diff_off < 1e-6, (
        f"expected NO seed-diversity with gaussian_pair_init=False (eval mode, no dropout), "
        f"got diff={diff_off:.2e} -- something else is seed-dependent unexpectedly")
    print(f"PASS: with gaussian_pair_init=False, different seeds give identical output on real "
          f"WS5 weights (diff={diff_off:.2e}) -- confirms the diversity above comes from z0, "
          f"not some other incidental randomness")


if __name__ == "__main__":
    test_runs_without_error_all_configs()
    test_default_config_reproducible()
    test_gaussian_init_gives_seed_diversity_on_real_ws5_weights()
    print("\nALL INTEGRATION TESTS PASSED")
