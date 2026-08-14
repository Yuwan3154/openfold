"""Gate tests for T4 self-distillation's promotion decision (openfold/utils/t4_self_distill.py).

Builds synthetic batches where the right answer is known by construction, so a regression in the
decision logic (rather than in TM itself, which test_tm_score.py covers) is caught.
"""

import torch

from openfold.utils.t4_self_distill import template_gate_metrics
from tests.test_tm_score import _helix, _walk


def _batch(native_ca, pred_ca, tmpl_cas=None, tmpl_cov=None):
    """Assemble the minimal feature/output dicts the gate reads. All args are (B,L,3) / list."""
    B, L, _ = native_ca.shape
    native = torch.zeros(B, L, 37, 3)
    native[:, :, 1] = native_ca
    nat_mask = torch.zeros(B, L, 37)
    nat_mask[:, :, 1] = 1
    pred = torch.zeros(B, L, 37, 3)
    pred[:, :, 1] = pred_ca

    batch = {"all_atom_positions": native, "all_atom_mask": nat_mask}
    if tmpl_cas is not None:
        T = len(tmpl_cas)
        tp = torch.zeros(B, T, L, 37, 3)
        tm_ = torch.zeros(B, T, L, 37)
        for i, c in enumerate(tmpl_cas):
            tp[:, i, :, 1] = c
            tm_[:, i, :, 1] = 1
            if tmpl_cov is not None and tmpl_cov[i] is not None:
                tm_[:, i, tmpl_cov[i]:, 1] = 0
        batch["template_all_atom_positions"] = tp
        batch["template_all_atom_mask"] = tm_
        batch["template_mask"] = torch.ones(B, T)
    return {"final_atom_positions": pred}, batch


def test_good_prediction_bad_template_promotes():
    nat = _helix(100)[None]
    bad = _walk(100, 99)[None]                      # unrelated fold
    out, b = _batch(nat, nat.clone(), tmpl_cas=[bad])
    m = template_gate_metrics(out, b)
    assert float(m["tm_pred"]) > 0.99
    assert float(m["tm_template"]) < 0.5
    assert float(m["promote"]) == 1.0


def test_perfect_template_blocks_promotion():
    nat = _helix(100)[None]
    bad_pred = _walk(100, 99)[None]
    out, b = _batch(nat, bad_pred, tmpl_cas=[nat.clone()])
    m = template_gate_metrics(out, b)
    assert float(m["tm_template"]) > 0.99
    assert float(m["promote"]) == 0.0


def test_margin_is_respected():
    """A prediction that beats its template by less than delta must NOT promote."""
    nat = _helix(120)[None]
    other = _walk(120, 42)[None]
    tmpl = 0.80 * nat + 0.20 * other
    pred = 0.79 * nat + 0.21 * other                 # very slightly worse-blended, near-identical TM
    out, b = _batch(nat, pred, tmpl_cas=[tmpl])
    m = template_gate_metrics(out, b, delta=0.05)
    assert abs(float(m["tm_pred"]) - float(m["tm_template"])) < 0.05
    assert float(m["promote"]) == 0.0
    # and with delta=0 the sign of the (tiny) margin decides instead
    m0 = template_gate_metrics(out, b, delta=0.0)
    assert float(m0["promote"]) == float(float(m["tm_pred"]) > float(m["tm_template"]))


def test_no_template_never_promotes():
    """With no template there is no baseline to beat -- promoting would be an ungated loop."""
    nat = _helix(80)[None]
    out, b = _batch(nat, nat.clone())
    m = template_gate_metrics(out, b)
    assert float(m["has_template"]) == 0.0
    assert float(m["promote"]) == 0.0


def test_masked_off_template_slot_is_ignored():
    nat = _helix(100)[None]
    out, b = _batch(nat, _walk(100, 99)[None], tmpl_cas=[nat.clone()])
    b["template_mask"] = torch.zeros(1, 1)           # the one good template is switched off
    m = template_gate_metrics(out, b)
    assert float(m["tm_template"]) == 0.0
    assert float(m["has_template"]) == 0.0
    assert float(m["promote"]) == 0.0


def test_best_of_several_templates_is_the_baseline():
    """The bar is the BEST template handed to the model, not the first or the mean."""
    nat = _helix(100)[None]
    out, b = _batch(nat, nat.clone(),
                    tmpl_cas=[_walk(100, 99)[None], nat.clone(), _walk(100, 77)[None]])
    m = template_gate_metrics(out, b)
    assert float(m["tm_template"]) > 0.99
    assert float(m["promote"]) == 0.0


def test_partial_template_coverage_is_penalized():
    """A template covering half the crop perfectly must not out-rank a full-coverage prediction."""
    nat = _helix(100)[None]
    out, b = _batch(nat, nat.clone(), tmpl_cas=[nat.clone()], tmpl_cov=[50])
    m = template_gate_metrics(out, b)
    assert 0.45 < float(m["tm_template"]) < 0.55
    assert float(m["promote"]) == 1.0


def test_min_tm_floor_blocks_low_quality_promotion():
    """Beating a terrible template is not enough if the prediction is itself terrible."""
    nat = _helix(120)[None]
    pred = 0.35 * nat + 0.65 * _walk(120, 5)[None]
    out, b = _batch(nat, pred, tmpl_cas=[_walk(120, 99)[None]])
    free = template_gate_metrics(out, b, min_tm=0.0)
    gated = template_gate_metrics(out, b, min_tm=float(free["tm_pred"]) + 0.01)
    assert float(free["promote"]) == 1.0
    assert float(gated["promote"]) == 0.0


def test_per_sample_decisions_in_a_batch():
    """Decisions must be per-sample; a batched gate that leaks would promote all-or-nothing."""
    nat = torch.cat([_helix(100, 1)[None], _walk(100, 2)[None]])
    pred = torch.cat([nat[0:1].clone(), _walk(100, 88)[None]])       # sample 0 perfect, 1 bad
    tmpl = torch.cat([_walk(100, 99)[None], nat[1:2].clone()])       # 0 bad tmpl, 1 perfect tmpl
    out, b = _batch(nat, pred, tmpl_cas=[tmpl])
    m = template_gate_metrics(out, b)
    assert m["promote"].tolist() == [1.0, 0.0]
