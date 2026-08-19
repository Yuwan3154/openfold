"""Audit gradient FLOW through the real Run B model: who gets gradient, how big, and is it finite.

⛔ WHY, specifically for this configuration:
  * `--freeze_non_evoformer` freezes everything except the Evoformer, and `freeze_all_except_evoformer`
    was patched to ALSO keep the contractive params. If that patch ever regressed, those 640 brand-new
    randomly-initialised parameters would sit frozen at their random init forever -- no error, no log
    line, and the ESMFold2 contractive trick would be silently inert while appearing enabled.
  * A FROZEN module still passes gradient THROUGH to the trunk. So "frozen" must mean grad is None on
    its own parameters, NOT that the confidence losses stop shaping the Evoformer -- both are checked.
  * bf16 autocast + a fresh trick can produce non-finite or vanishing gradients that a masked-mean loss
    will happily average away.

Runs on CPU with a tiny crop so it cannot disturb the live run.
"""

import argparse
import collections

import torch

from openfold.config import model_config
from openfold.model.model import AlphaFold
from openfold.block_replacement_scripts.pruned_evoformer import freeze_all_except_evoformer
from openfold.utils.loss import AlphaFoldLoss

ap = argparse.ArgumentParser()
ap.add_argument("--crop", type=int, default=32)
ap.add_argument("--n-templ", type=int, default=4)
ap.add_argument("--autocast", action="store_true", help="run under bf16 autocast, as training does")
a = ap.parse_args()

cfg = model_config("finetuning_ptm", train=True, low_prec=True)
for k in ("common", "train"):
    cfg.data[k].max_extra_msa = 1
    cfg.data[k].max_msa_clusters = 1
cfg.loss.masked_msa.weight = 0.0
cfg.model.recycling_embedder.use_contractive = True
cfg.model.recycling_embedder.use_gaussian_pair_init = True
cfg.globals.blocks_per_ckpt = None          # no activation checkpointing, so grads are direct

L, T, R = a.crop, a.n_templ, cfg.data.common.max_recycling_iters + 1
model = AlphaFold(cfg)
freeze_all_except_evoformer(model)
model.train()
loss_fn = AlphaFoldLoss(cfg.loss)

torch.manual_seed(0)
B = 1


def rnd(*s):
    return torch.randn(*s)


aat = torch.randint(0, 20, (B, L))
batch = {
    "aatype": aat,
    "residue_index": torch.arange(L).unsqueeze(0).expand(B, L).contiguous(),
    "seq_mask": torch.ones(B, L),
    "seq_length": torch.full((B,), L),
    "msa_feat": rnd(B, 1, L, 49),
    "msa_mask": torch.ones(B, 1, L),
    "msa_row_mask": torch.ones(B, 1),
    "extra_msa": torch.randint(0, 21, (B, 1, L)),
    "extra_msa_mask": torch.zeros(B, 1, L),         # query-only MSA => inert, as in Run B
    "extra_msa_row_mask": torch.zeros(B, 1),
    # ⛔ feats.build_extra_msa_feat reads these EXACT names (feats.py:197-198); the
    # extra_msa_* spelling is a different, later feature and produced a KeyError here
    "extra_deletion_value": torch.zeros(B, 1, L),
    "extra_has_deletion": torch.zeros(B, 1, L),
    "target_feat": rnd(B, L, 22),
    "template_aatype": torch.randint(0, 21, (B, T, L)),
    "template_all_atom_positions": rnd(B, T, L, 37, 3),
    "template_all_atom_mask": torch.ones(B, T, L, 37),
    "template_mask": torch.ones(B, T),
    "template_pseudo_beta": rnd(B, T, L, 3),
    "template_pseudo_beta_mask": torch.ones(B, T, L),
    "template_torsion_angles_sin_cos": rnd(B, T, L, 7, 2),
    "template_alt_torsion_angles_sin_cos": rnd(B, T, L, 7, 2),
    "template_torsion_angles_mask": torch.ones(B, T, L, 7),
    "atom14_atom_exists": torch.ones(B, L, 14),
    "atom37_atom_exists": torch.ones(B, L, 37),
    "residx_atom14_to_atom37": torch.zeros(B, L, 14, dtype=torch.long),
    "residx_atom37_to_atom14": torch.zeros(B, L, 37, dtype=torch.long),
    "no_recycling_iters": torch.tensor([R - 1] * B),
}
batch = {k: (v.unsqueeze(-1).expand(*v.shape, R).contiguous() if torch.is_tensor(v) else v)
         for k, v in batch.items()}

ctx = (torch.autocast(device_type="cpu", dtype=torch.bfloat16) if a.autocast
       else torch.autocast(device_type="cpu", enabled=False))
print(f"forward: L={L} templates={T} recycles={R} autocast={a.autocast}")
with ctx:
    out = model(batch)
print("  forward OK; outputs:", sorted(k for k in out if not k.startswith("sm"))[:8])

b1 = {k: v[..., -1] for k, v in batch.items()}
b1.update({
    "all_atom_positions": rnd(B, L, 37, 3),
    "all_atom_mask": torch.ones(B, L, 37),
    "resolution": torch.full((B,), 2.0),
    "use_clamped_fape": torch.zeros(B),
    "backbone_rigid_tensor": torch.eye(4).view(1, 1, 4, 4).expand(B, L, 4, 4).contiguous(),
    "backbone_rigid_mask": torch.ones(B, L),
    "chi_angles_sin_cos": rnd(B, L, 4, 2),
    "chi_mask": torch.ones(B, L, 4),
    "rigidgroups_gt_frames": torch.eye(4).view(1, 1, 1, 4, 4).expand(B, L, 8, 4, 4).contiguous(),
    "rigidgroups_alt_gt_frames": torch.eye(4).view(1, 1, 1, 4, 4).expand(B, L, 8, 4, 4).contiguous(),
    "rigidgroups_gt_exists": torch.ones(B, L, 8),
    "atom14_gt_positions": rnd(B, L, 14, 3),
    "atom14_alt_gt_positions": rnd(B, L, 14, 3),
    "atom14_gt_exists": torch.ones(B, L, 14),
    "atom14_alt_gt_exists": torch.ones(B, L, 14),
    "atom14_atom_is_ambiguous": torch.zeros(B, L, 14),
    "pseudo_beta": rnd(B, L, 3),
    "pseudo_beta_mask": torch.ones(B, L),
    "true_msa": torch.randint(0, 21, (B, 1, L)),
    "bert_mask": torch.zeros(B, 1, L),
})
loss, bd = loss_fn(out, b1, _return_breakdown=True)
print(f"\nloss = {float(loss):.4f}")
print("  breakdown:", {k: round(float(v), 4) for k, v in sorted(bd.items())})
assert torch.isfinite(loss), "LOSS IS NON-FINITE"
loss.backward()

groups = collections.defaultdict(lambda: [0, 0, 0.0, 0])   # n, n_with_grad, sumsq, n_nonfinite
for n, p in model.named_parameters():
    g = n.split(".")[0] + ("." + n.split(".")[1] if n.startswith("aux_heads") else "")
    e = groups[g]
    e[0] += p.numel()
    if p.grad is not None:
        e[1] += p.numel()
        gf = p.grad.float()
        e[2] += float(gf.pow(2).sum())
        e[3] += int((~torch.isfinite(gf)).sum())

print(f"\n{'group':34s} {'params':>12} {'with grad':>12} {'‖grad‖':>12} {'nonfinite':>10}  expected")
bad = []
for g in sorted(groups):
    n, ng, sq, nf = groups[g]
    should_train = g in ("evoformer", "recycling_embedder")
    ok = (ng > 0) if should_train else (ng == 0)
    print(f"{g:34s} {n:>12,} {ng:>12,} {sq**0.5:>12.4e} {nf:>10}  "
          f"{'TRAIN' if should_train else 'frozen'}{'' if ok else '   ⛔ MISMATCH'}")
    if not ok:
        bad.append(g)
    if nf:
        bad.append(g + " (nonfinite grad)")
    if should_train and sq == 0.0:
        bad.append(g + " (ZERO gradient -- receives no learning signal)")

# the contractive params specifically: brand new, and easy to silently freeze
print("\n=== contractive params (new, random init, live outside model.evoformer) ===")
for n, p in model.named_parameters():
    if "contractive" in n:
        gn = float(p.grad.float().norm()) if p.grad is not None else None
        print(f"  {n:52s} requires_grad={p.requires_grad} "
              f"‖grad‖={'None' if gn is None else f'{gn:.4e}'}")
        if p.requires_grad and (p.grad is None or gn == 0.0):
            bad.append(f"{n} has no/zero gradient")

print("\n" + ("✅ GRADIENT AUDIT PASS" if not bad else "⛔ GRADIENT AUDIT FAILURES: " + str(bad)))
