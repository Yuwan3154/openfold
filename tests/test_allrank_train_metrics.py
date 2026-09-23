"""t4_all/* and explore_all/*: the rank-0-only t4/* and explore/* scalars, summed on every rank and reduced
with ONE all_reduce(SUM) per epoch.

⛔ WHAT THE OLD TAGS ARE (window audit A, 2026-09-23): every t4/* and explore/* scalar is unsynced, so the
logged epoch value is rank 0's 750 steps of 3000; and at batch size 1 t4/tm_template, margin and
promote_rate count a step without a template as 0 while t4/tm_pred averages every step, so comparing them
mixes populations. The same-name t4_all/* keep the old definitions (continuity); *_on_templated /
*_on_untemplated are the like-for-like versions.

The per-step definitions are checked against the expressions in train_openfold.py itself (parsed with ast:
importing train_openfold needs a GPU).
"""

import ast
import os

import pytest
import torch

from openfold.utils import allrank_metrics
from openfold.utils.allrank_metrics import TrainEpochSums, train_epoch_metrics
from tests.gloo_harness import record_collectives, run_ranks

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_OPENFOLD = os.path.join(REPO_ROOT, "train_openfold.py")
WORLD = 4                  # the A6000's DDP world size
LADDER = [0.0, 8.0, 16.0, 32.0]  # Run C v2's --explore_noise_ladder (tau 0/8/16/32)
STEPS = 6


def _training_step_logs():
    """{tag: value expression} for every constant-named self.log in training_step, plus the n_t assignment."""
    tree = ast.parse(open(TRAIN_OPENFOLD).read())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "training_step")
    exprs = {}
    for node in ast.walk(fn):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "log"
                and node.args and isinstance(node.args[0], ast.Constant)):
            exprs[node.args[0].value] = node.args[1]
        if isinstance(node, ast.Assign) and [ast.unparse(t) for t in node.targets] == ["n_t"]:
            exprs["n_t"] = node.value
    return exprs


def _eval(expr, ns):
    return float(eval(compile(ast.Expression(expr), TRAIN_OPENFOLD, "eval"), dict(ns, torch=torch)))


def _t4_step(rank, s):
    """Batch-size-1 gate outputs; rank 3 never has a template. Dyadic, so every sum is exact."""
    has = 0.0 if rank == 3 else float((s + rank) % 3 != 0)
    tp = (8 + s + 4 * rank) / 64
    tt = (4 + s + 2 * rank) / 64 * has
    promote = float(has > 0 and tp > tt + 0.05)
    return [torch.tensor([v]) for v in (tp, tt, has, promote)]


def _explore_step(rank, s):
    losses = [(16 + ((3 * j + s + rank) % 5)) / 8 for j in range(len(LADDER))]
    confs = [(j + s) % 4 / 16 for j in range(len(LADDER))]
    best = min(range(len(losses)), key=lambda j: losses[j])
    return (s + rank) % len(LADDER), best, losses, confs


def _rank_sums(rank):
    sums = TrainEpochSums()
    for s in range(STEPS):
        sums.add_t4_step(*_t4_step(rank, s))
        pick, best, losses, confs = _explore_step(rank, s)
        sums.add_explore_step(pick, best, losses, confs, True, LADDER[pick])
    sums.add_t4_promoted(len(LADDER))
    return sums


def test_imports_are_this_checkout():
    print("allrank_metrics:", allrank_metrics.__file__, "train_openfold parsed:", TRAIN_OPENFOLD)
    assert allrank_metrics.__file__.startswith(REPO_ROOT), (allrank_metrics.__file__, REPO_ROOT)


@pytest.mark.parametrize("batch", [
    [_t4_step(0, 1)], [_t4_step(0, 0)],                        # one step with, one without a template
    [[torch.cat(x) for x in zip(*(_t4_step(r, 1) for r in range(3)))]],  # batch size 3, 2 of 3 templated
])
def test_t4_step_values_are_train_openfolds_own_expressions(batch):
    exprs = _training_step_logs()
    (tp, tt, has, pr), = batch
    m = {"tm_pred": tp, "tm_template": tt, "has_template": has, "promote": pr}
    ns = {"m": m, "n_t": torch.tensor(_eval(exprs["n_t"], {"m": m}))}
    sums = TrainEpochSums()
    sums.add_t4_step(tp, tt, has, pr)
    for k in ("tm_pred", "tm_template", "margin", "promote_rate", "has_template"):
        print(k, _eval(exprs[f"t4/{k}"], ns), sums.sums[f"t4/step_{k}"])
        assert sums.sums[f"t4/step_{k}"] == pytest.approx(_eval(exprs[f"t4/{k}"], ns), abs=1e-7), k


@pytest.mark.parametrize("sel", ["loss", "ptm"])
def test_explore_step_values_are_train_openfolds_own_expressions(sel):
    exprs = _training_step_logs()
    pick, best, losses, confs = _explore_step(1, 2)
    ns = {"_pick": pick, "_best_loss": best, "_losses": losses, "_confs": confs, "_K": len(losses),
          "_sel": sel, "_ladder": LADDER}
    sums = TrainEpochSums()
    sums.add_explore_step(pick, best, losses, confs, sel == "loss", LADDER[pick])
    for k in ("conf_picks_loss_argmin", "loss_spread", "loss_gain_vs_mean", "regret_vs_best", "conf_spread",
              "using_true_loss", "selected_rung", "selected_tau"):
        assert sums.sums[f"explore/{k}"] == _eval(exprs[f"explore/{k}"], ns), k


def test_known_bad_control_the_all_step_template_mean_is_diluted_by_no_template_steps():
    """Old definition vs like-for-like on the same steps: they differ exactly by the templated fraction."""
    out = train_epoch_metrics(_rank_sums(0).all_reduce(False, "cpu"))
    print({k: v for k, v in out.items() if k.startswith("t4_all/")})
    frac = out["t4_all/n_templated"] / out["t4_all/n_items"]
    assert 0 < frac < 1
    assert out["t4_all/tm_template"] == pytest.approx(frac * out["t4_all/tm_template_on_templated"])
    assert out["t4_all/tm_template"] < out["t4_all/tm_template_on_templated"]
    assert out["t4_all/margin"] == pytest.approx(frac * out["t4_all/margin_on_templated"])


def test_empty_counts_omit_their_ratios_but_keep_the_counts():
    sums = TrainEpochSums()
    sums.add_t4_step(*_t4_step(3, 0))                          # no template
    sums.add_explore_step(0, 0, [1.0, 2.0], [0.5, 0.25], False)  # no ladder
    out = train_epoch_metrics(sums.all_reduce(False, "cpu"))
    assert out["t4_all/n_templated"] == 0 and "t4_all/tm_template_on_templated" not in out
    assert out["t4_all/tm_pred_on_untemplated"] == out["t4_all/tm_pred"]
    assert "t4_all/promoted_per_step" not in out and "explore_all/selected_rung" not in out
    assert out["explore_all/n_steps"] == 1
    assert train_epoch_metrics(TrainEpochSums().all_reduce(False, "cpu")) == {}


def _pooled_expected():
    """The all-rank values straight from every rank's raw steps, without the helper."""
    t4 = [[float(x) for x in _t4_step(r, s)] for r in range(WORLD) for s in range(STEPS)]
    ex = [_explore_step(r, s) for r in range(WORLD) for s in range(STEPS)]
    n, templ = len(t4), [x for x in t4 if x[2]]
    untempl = [x for x in t4 if not x[2]]
    return n, {
        "t4_all/tm_pred": sum(x[0] for x in t4) / n,
        "t4_all/tm_template": sum(x[1] for x in t4) / n,
        "t4_all/tm_template_on_templated": sum(x[1] for x in templ) / len(templ),
        "t4_all/tm_pred_on_templated": sum(x[0] for x in templ) / len(templ),
        "t4_all/margin_on_templated": (sum(x[0] for x in templ) - sum(x[1] for x in templ)) / len(templ),
        "t4_all/tm_pred_on_untemplated": sum(x[0] for x in untempl) / len(untempl),
        "t4_all/n_templated": float(len(templ)),
        "explore_all/regret_vs_best": sum(l[p] - l[b] for p, b, l, _ in ex) / n,
        "explore_all/selected_tau": sum(LADDER[p] for p, _, _, _ in ex) / n,
    }


def test_gloo_4_ranks_issue_one_all_reduce_and_get_the_exact_pooled_values(tmp_path):
    def fn(rank, world):
        sums = _rank_sums(rank)
        calls = record_collectives()
        out = train_epoch_metrics(sums.all_reduce(True, "cpu"))
        return {"calls": list(calls), "out": out}

    res = run_ranks(fn, WORLD, tmp_path / "store")
    n, expected = _pooled_expected()
    print("calls:", {r: res[r]["calls"] for r in range(WORLD)})
    assert all(res[r]["calls"] == ["all_reduce"] for r in range(WORLD))
    assert all(res[r]["out"] == res[0]["out"] for r in range(WORLD)), "ranks disagree"
    out = res[0]["out"]
    assert out["t4_all/n_steps"] == out["explore_all/n_steps"] == n
    assert out["t4_all/n_promote_steps"] == WORLD and out["t4_all/promoted_per_step"] == len(LADDER)
    for k, v in expected.items():
        assert out[k] == v, (k, out[k], v)


def test_known_bad_control_rank_0_alone_is_not_the_all_rank_value():
    """What the old tags logged (rank 0's steps only) differs from the pooled value on these data."""
    rank0 = train_epoch_metrics(_rank_sums(0).all_reduce(False, "cpu"))
    _, pooled = _pooled_expected()
    for k in ("t4_all/tm_template_on_templated", "t4_all/tm_pred", "explore_all/selected_tau"):
        print(k, "rank 0", rank0[k], "all ranks", pooled[k])
        assert rank0[k] != pooled[k], k
