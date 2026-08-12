"""Controlled precision/methodology check: evaluate T1's OWN checkpoint through the EXACT SAME
reference methodology pda_baseline_full.py used to produce the 0.728 lDDT "Stock AF2" baseline
(fp32, no autocast, TF32 disabled, crop 256, low_prec=False) -- reusing that script's own
build_cfg_stock/run_eval code UNMODIFIED, only swapping the weight source from raw jax params to
T1's checkpoint. T1's checkpoint has had only ~1500 training steps from the same jax init, so if
this result lands close to 0.728, precision+harness-methodology (not weights/architecture)
explains the original gap; T1's own real-harness number (bf16, uncropped, already logged: 0.7094
lDDT at this same checkpoint) is the other side of the comparison -- not reproduced here, already
have it from the real training run.
"""
import os
import sys

sys.path.insert(0, "/home/jupyter-chenxi/openfold-esmfold2-recycling/prune_work")
sys.path.insert(0, "/home/jupyter-chenxi/openfold-esmfold2-recycling")

import torch

from pda_baseline_full import build_cfg_stock, run_eval, shard, MANIFEST, CIF_CACHE_DIR, DEVICE, OUT_DIR
from pda_dataset import PDASingleSeqDataset
from openfold.model.model import AlphaFold

T1_CKPT = os.environ["T1_CKPT"]
RUN_TAG = os.environ.get("RUN_TAG", "t1ckpt_refmethod")

# Match the reference script's implicit precision behavior exactly: it never touches these, so
# they sit at PyTorch's plain defaults. Set explicitly here so this run cannot silently inherit
# TF32/reduced-precision settings from anything else that may have touched process-global state.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def load_t1_ckpt(cfg):
    # cfg (build_cfg_stock) sets template.enabled=False -> AlphaFold(cfg) never builds a
    # template_embedder submodule at all. T1's real checkpoint DOES carry template_embedder.*
    # weights (trained with --single_seq_keep_templates). Discarding those keys here matches
    # this SAME script's own established convention for load_ws5() above (see its own
    # `assert all(k.startswith("template_embedder.") ...)`) -- not a new judgment call.
    m = AlphaFold(cfg)
    sd = torch.load(T1_CKPT, map_location="cpu", weights_only=False)["state_dict"]
    sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not missing, f"unexpected missing keys: {missing}"
    assert all(k.startswith("template_embedder.") for k in unexpected), \
        f"unexpected non-template keys: {[k for k in unexpected if not k.startswith('template_embedder.')]}"
    return m.to(DEVICE).eval()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cfg = build_cfg_stock()  # UNMODIFIED reference config: fp32, crop 256, low_prec=False
    ds = PDASingleSeqDataset(manifest_path=MANIFEST, cif_cache_dir=CIF_CACHE_DIR,
                              config=cfg.data, mode="eval")
    ds.manifest = shard(ds.manifest)
    print(f"n={len(ds)} crop={cfg.data.eval.crop_size} eps={cfg.globals.eps} "
          f"ckpt={T1_CKPT}", flush=True)
    model = load_t1_ckpt(cfg)
    rows = run_eval(model, ds, RUN_TAG)
    n = len(rows)
    mean_lddt = sum(r["lddt_ca"] for r in rows) / n
    mean_ptm = sum(r["ptm"] for r in rows) / n
    n_succ = sum(r["success_2A"] for r in rows)
    print(f"\n[RESULT] n={n} mean_lddt={mean_lddt:.4f} mean_ptm={mean_ptm:.4f} "
          f"recall@2A={n_succ}/{n} ({n_succ/n:.3f})", flush=True)


if __name__ == "__main__":
    main()
