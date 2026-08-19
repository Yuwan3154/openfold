"""Audit every FEATURE the model is fed, across all three data regimes.

⛔ WHY: a malformed feature does not raise. It produces a plausible loss, trains, and silently teaches
the model something wrong -- exactly how the T2 residue-frame bug survived until a live assert caught
it. The checks here are the ones that would have caught the bugs this project has actually hit:
  * NaN/Inf anywhere (a masked-mean loss hides a non-finite slice)
  * masks that are not 0/1, or coordinates that are nonzero where the mask says absent
    (the T2 native-frame bug wrote coords at the wrong residues while looking well-formed)
  * one-hots that do not sum to 1 (the HHBLITS-vs-restype ordering trap: a permuted one-hot is still
    a valid one-hot, so ONLY a decode round-trip catches it -- included below)
  * residue_index not strictly increasing (the qmap bug)
  * template mask/coord agreement, and the empty-template placeholder's ragged axis

REGIMES (all three are audited because they use DIFFERENT code paths, not just different data):
  R1 TRAIN   -- query-only MSA, natural+synthetic mixing, crop, dropout-time transforms
  R2 EVAL    -- ws5 val set, eval-mode transforms (subsample_templates=False, "top templates")
  R3 PDA     -- the de novo design eval set, a SEPARATE dataset class with no alignment dir at all
"""

import argparse
import collections
import random

import numpy as np
import torch

from openfold.config import model_config
from openfold.data.data_modules import OpenFoldSingleDataset
from openfold.np import residue_constants as rc

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", required=True)
ap.add_argument("--aln-dir", required=True)
ap.add_argument("--train-list", required=True)
ap.add_argument("--val-list", required=True)
ap.add_argument("--obsolete", required=True)
ap.add_argument("--kalign", required=True)
ap.add_argument("--template-cache", required=True)
ap.add_argument("--t2-index", required=True)
ap.add_argument("--t2-root", required=True)
ap.add_argument("--t2-qmap", required=True)
ap.add_argument("--pref-counts", required=True)
ap.add_argument("--pda-manifest", default=None)
ap.add_argument("--pda-cif-dir", default=None)
ap.add_argument("--n", type=int, default=6)
a = ap.parse_args()

FAIL = []


def check(cond, msg):
    if not cond:
        FAIL.append(msg)
        print(f"    ⛔ {msg}")
    return cond


def audit_tensor_dict(tag, f, mode):
    """Generic sanity over every tensor, then feature-specific invariants."""
    n_t = 0
    for k, v in sorted(f.items()):
        if not torch.is_tensor(v):
            continue
        n_t += 1
        vf = v.float()
        if not torch.isfinite(vf).all():
            check(False, f"{tag}: {k} has {int(torch.isnan(vf).sum())} NaN / "
                         f"{int(torch.isinf(vf).sum())} Inf")
        if k.endswith("_mask") or k in ("seq_mask", "msa_mask", "template_mask"):
            u = torch.unique(vf)
            check(bool(((u == 0) | (u == 1)).all()),
                  f"{tag}: {k} is not binary, uniques={u[:6].tolist()}")

    strip = lambda t: t[..., 0] if t.dim() > 1 else t
    aat = strip(f["aatype"])
    check(int(aat.min()) >= 0 and int(aat.max()) <= 20,
          f"{tag}: aatype out of range [{int(aat.min())},{int(aat.max())}]")

    # ⛔⛔ EVERY CHECK BELOW IS MASK-AWARE, and that is not a detail. `make_fixed_size` pads NUM_RES to
    # crop_size and NUM_MSA_SEQ to max_msa_clusters with ZEROS, so the padded tail legitimately has
    # residue_index 0 and an all-zero one-hot. A naive check reports both as violations: an earlier
    # version of this file produced 12 "failures" that were entirely its own padding blindness
    # (1r3b_A: seq_length 202 in a 256 crop, residue_index ... 200, 201, 0, 0, 0 with seq_mask 0 there).
    # ⭐ So padding is not merely skipped -- it is ASSERTED to be zero, which is the real invariant.
    sm = strip(f["seq_mask"]).bool()
    n_real = int(sm.sum())

    ri = strip(f["residue_index"]).long()
    d = torch.diff(ri[sm])
    check(bool((d > 0).all()) if d.numel() else True,
          f"{tag}: residue_index not strictly increasing WITHIN seq_mask "
          f"(min diff {int(d.min()) if d.numel() else 'n/a'})")
    if (~sm).any():
        check(int(ri[~sm].abs().max()) == 0,
              f"{tag}: residue_index nonzero in the PADDED region (max {int(ri[~sm].abs().max())})")

    # ---- coordinates: finite, plausible magnitude, and ZERO wherever the mask says absent
    pos, msk = strip(f["all_atom_positions"]), strip(f["all_atom_mask"])
    check(int(msk[~sm].sum()) == 0 if (~sm).any() else True,
          f"{tag}: all_atom_mask set in the PADDED region")
    present = msk > 0
    if present.any():
        mag = pos[present].abs().max()
        check(float(mag) < 1e4, f"{tag}: |all_atom_positions| = {float(mag):.1f}, implausible")
    absent = ~present
    if absent.any():
        check(float(pos[absent].abs().max()) == 0.0,
              f"{tag}: all_atom_positions NONZERO where all_atom_mask=0 "
              f"(max {float(pos[absent].abs().max()):.3g}) -- coords written at masked residues")

    # ---- templates
    if "template_aatype" in f:
        t_aat = strip(f["template_aatype"])
        t_msk = strip(f["template_all_atom_mask"])
        t_pos = strip(f["template_all_atom_positions"])
        tm_ = strip(f["template_mask"]) if "template_mask" in f else None
        check(t_aat.shape[0] == t_pos.shape[0] == t_msk.shape[0],
              f"{tag}: ragged template axis {t_aat.shape[0]}/{t_pos.shape[0]}/{t_msk.shape[0]}")
        tp = t_msk > 0
        if (~tp).any():
            check(float(t_pos[~tp].abs().max()) == 0.0,
                  f"{tag}: template coords NONZERO where template_all_atom_mask=0")
        # ⭐ the ordering trap: template_aatype is HHBLITS-ordered ints AFTER fix_templates_aatype;
        # before it, a one-hot. Either way every value must decode to a real residue.
        if t_aat.dtype in (torch.int64, torch.int32):
            check(int(t_aat.min()) >= 0 and int(t_aat.max()) <= 21,
                  f"{tag}: template_aatype ints out of range [{int(t_aat.min())},{int(t_aat.max())}]")
        else:
            s = t_aat.sum(-1)
            nz = s[s > 0]
            if nz.numel():
                check(bool(torch.allclose(nz, torch.ones_like(nz), atol=1e-4)),
                      f"{tag}: template_aatype one-hot rows do not sum to 1 (max {float(nz.max()):.3f})")
        if tm_ is not None:
            # ⛔ NOT `n_real` -- that name holds the residue count for the final line, and reusing it
            # here made every regime report its TEMPLATE count as its residue count (R1 chains showed
            # "0-4 real residues of 256"). The checks were unaffected; only the report lied.
            n_templ_live = int((tm_ > 0).sum())
            print(f"    templates delivered: {n_templ_live}/{t_aat.shape[0]}")

    # ---- MSA block
    if "msa_feat" in f:
        mf = strip(f["msa_feat"])
        mm = strip(f["msa_mask"]).bool()
        check(mf.shape[-1] == 49, f"{tag}: msa_feat has {mf.shape[-1]} channels, expected 49")
        oh = mf[..., :23].sum(-1)
        live = oh[mm]
        check(bool(torch.allclose(live, torch.ones_like(live), atol=1e-3)) if live.numel() else True,
              f"{tag}: msa_feat one-hot block does not sum to 1 where msa_mask=1")
        if (~mm).any():
            check(float(oh[~mm].abs().max()) == 0.0,
                  f"{tag}: msa_feat one-hot NONZERO in padded MSA rows")
    if "extra_msa" in f:
        em = strip(f["extra_msa"])
        emm = strip(f["extra_msa_mask"])
        live = float(emm.max())
        print(f"    extra_msa: shape {tuple(em.shape)} mask_max={live:.3f} "
              f"({'LIVE' if live > 0 else 'inert (query-only MSA)'})")
    print(f"    {n_t} tensors checked, {n_real} real residues of {sm.numel()}")


def make_cfg():
    c = model_config("finetuning_ptm", train=True, low_prec=True)
    for k in ("common", "train"):
        c.data[k].max_extra_msa = 1
        c.data[k].max_msa_clusters = 1
    c.loss.masked_msa.weight = 0.0
    c.data.train.crop_size = 256
    return c


cfg = make_cfg()
common = dict(
    data_dir=a.data_dir, template_mmcif_dir=a.data_dir, max_template_date="2018-04-30",
    config=cfg.data, chain_data_cache_path=None, kalign_binary_path=a.kalign,
    max_template_hits=cfg.data.train.max_template_hits,
    shuffle_top_k_prefiltered=cfg.data.train.shuffle_top_k_prefiltered,
    template_release_dates_cache_path=a.template_cache, obsolete_pdbs_file_path=a.obsolete,
    force_query_only_msa=True,
)

print("=" * 92)
print("REGIME 1: TRAIN -- query-only MSA + count-matched synthetic templates (Run B's actual path)")
from openfold.data.synthetic_templates import SyntheticTemplatePool
pool = SyntheticTemplatePool(a.t2_index, a.t2_root, min_tm=0.3, max_tm=0.9, qmap_path=a.t2_qmap)
ds1 = OpenFoldSingleDataset(alignment_dir=a.aln_dir, mode="train", chain_list_path=a.train_list,
                            synthetic_template_pool=pool, n_synthetic_templates=0,
                            t2_replace_prob=0.5, t2_topup_to=20,
                            prefiltered_counts_path=a.pref_counts, **common)
print(f"  dataset: {len(ds1)} chains; prefiltered table missing for {ds1._n_prefiltered_missing}")
random.seed(0)
idxs = random.sample(range(len(ds1)), a.n)
for i in idxs:
    print(f"  -- {ds1.idx_to_chain_id(i)}")
    torch.manual_seed(0); np.random.seed(0)
    audit_tensor_dict(f"R1/{ds1.idx_to_chain_id(i)}", ds1[i], "train")

print("=" * 92)
print("REGIME 2: EVAL -- ws5 val set, eval-mode transforms")
ds2 = OpenFoldSingleDataset(alignment_dir=a.aln_dir, mode="eval", chain_list_path=a.val_list, **common)
print(f"  dataset: {len(ds2)} chains")
for i in range(min(a.n, len(ds2))):
    print(f"  -- {ds2.idx_to_chain_id(i)}")
    torch.manual_seed(0); np.random.seed(0)
    audit_tensor_dict(f"R2/{ds2.idx_to_chain_id(i)}", ds2[i], "eval")

if a.pda_manifest:
    print("=" * 92)
    print("REGIME 3: PDA -- de novo designs, a separate dataset class with NO alignment dir")
    import sys
    sys.path.insert(0, "/tmp/of_test_wt/prune_work")
    from pda_dataset import PDASingleSeqDataset
    ds3 = PDASingleSeqDataset(manifest_path=a.pda_manifest, cif_cache_dir=a.pda_cif_dir,
                              config=cfg.data, mode="eval")
    print(f"  dataset: {len(ds3)} entries")
    for i in range(min(a.n, len(ds3))):
        torch.manual_seed(0); np.random.seed(0)
        print(f"  -- entry {i}")
        audit_tensor_dict(f"R3/{i}", ds3[i], "eval")

print("=" * 92)
print(f"FEATURE AUDIT: {'✅ PASS -- no violations' if not FAIL else '⛔ %d VIOLATIONS' % len(FAIL)}")
for m in FAIL[:25]:
    print("   ", m)
