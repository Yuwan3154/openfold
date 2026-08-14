"""T4 self-distillation gate: does the model's own prediction beat the template it was handed?

The idea (user, 2026-08-13): when the model, given a template of quality q_in, predicts a structure
BETTER than that template, the prediction is a better template than the one we have -- promote it
into the pool so later epochs train on a template distribution that improves with the model, rather
than one frozen at Protpardelle's output quality.

This module is the GATE only (measurement + decision). Writing promotions into a pool and sampling
them back is deliberately separate: it shares the chain->template index that T2's template-consuming
path needs and that does not exist yet, so building it here would build it twice.

Acceptance criterion (user, 2026-08-13): TM with delta = 0.05.

⛔ Both TMs are computed IN-LOOP on the CROPPED tensors, never against a precomputed full-chain
number. `random_crop_to_size` applies the same residue window to `all_atom_*` and every `template_*`
field, so inside a batch the three structures are residue-aligned -- but a TM precomputed over the
full chain has a different L_norm, a different d0, and a different residue subset, and is simply not
comparable to the prediction's cropped TM. Comparing them would silently bias every promotion
decision by whatever the crop happened to select.
"""

from __future__ import annotations

import torch

from openfold.utils.tm_score import FAST_KWARGS, tm_score

CA = 1  # atom37 index of the alpha carbon


def template_gate_metrics(
    outputs: dict,
    batch: dict,
    delta: float = 0.05,
    min_tm: float = 0.0,
    tm_kwargs: dict | None = None,
) -> dict:
    """Compare the prediction to the best template the model was given, both vs the native.

    Call AFTER the recycling dimension has been stripped from `batch`.

    Args:
        outputs: model output; uses `final_atom_positions` (B,L,37,3).
        batch: cropped features; uses `all_atom_positions`, `all_atom_mask` and, when templates are
            enabled, `template_all_atom_positions`, `template_all_atom_mask`, `template_mask`.
        delta: promotion margin in TM.
        min_tm: absolute floor on the prediction's TM; 0.0 disables it.

    Returns:
        dict of (B,) tensors: `tm_pred`, `tm_template`, `has_template`, `promote`.
    """
    kw = FAST_KWARGS if tm_kwargs is None else tm_kwargs
    with torch.no_grad():
        native = batch["all_atom_positions"]                       # (B,L,37,3)
        native_ca = batch["all_atom_mask"][..., CA] > 0             # (B,L)
        B, L = native_ca.shape

        tm_pred = tm_score(
            outputs["final_atom_positions"][:, :, CA, :], native[:, :, CA, :],
            mask=native_ca.float(), norm_mask=native_ca.float(), **kw,
        )

        tmpl_pos = batch.get("template_all_atom_positions")
        tmpl_mask = batch.get("template_mask")
        if tmpl_pos is None or tmpl_mask is None or tmpl_pos.shape[1] == 0:
            tm_tmpl = torch.zeros_like(tm_pred)
            has_tmpl = torch.zeros_like(tm_pred)
        else:
            T = tmpl_pos.shape[1]
            tmpl_ca_ok = batch["template_all_atom_mask"][..., CA] > 0        # (B,T,L)
            # a template slot that is masked off contributes no residues and so scores 0
            pair_ok = tmpl_ca_ok & native_ca[:, None, :] & (tmpl_mask > 0)[..., None]
            tm_flat = tm_score(
                tmpl_pos[:, :, :, CA, :].reshape(B * T, L, 3),
                native[:, None, :, CA, :].expand(B, T, L, 3).reshape(B * T, L, 3),
                mask=pair_ok.reshape(B * T, L).float(),
                # ⭐ normalized by the NATIVE, so a partially-covering template is penalized for its
                # gaps rather than flattered by scoring only where it happens to exist
                norm_mask=native_ca[:, None, :].expand(B, T, L).reshape(B * T, L).float(),
                **kw,
            )
            # "beat the BEST template it was handed" -- the model could exploit any of them
            tm_tmpl = tm_flat.reshape(B, T).max(dim=1).values
            has_tmpl = (tmpl_mask > 0).any(dim=1).float()

        promote = (
            (tm_pred > tm_tmpl + delta) & (has_tmpl > 0) & (tm_pred >= min_tm)
        ).float()
    return {
        "tm_pred": tm_pred,
        "tm_template": tm_tmpl,
        "has_template": has_tmpl,
        "promote": promote,
    }
