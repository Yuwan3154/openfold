"""The rung PICK DISTRIBUTION must be measurable at full n, not from a rank-0 subsample.

⛔⛔ WHY INDICATORS AND NOT `sync_dist` ON THE RUNG INDEX (the trap this pins):
`self.log("explore/selected_rung", float(_pick), sync_dist=True)` AVERAGES the rung index across ranks.
The mean of picks {0,3,1,2} is 1.5 -- not a rung, and a per-step mean of indices cannot be inverted
back into a histogram. K binary indicators are the correct estimator: each one's epoch mean IS
P(rung r picked), and averaging an indicator across ranks is exactly what you want.

⚠️ Also pins on_step=False/on_epoch=True: with on_step=True Lightning would emit ~22k extra per-step
points into TB for no benefit, since only the epoch aggregate is the quantity of interest.
"""

import ast
import inspect
import textwrap

from train_openfold import OpenFoldWrapper


def _training_step():
    return ast.parse(textwrap.dedent(inspect.getsource(OpenFoldWrapper.training_step))).body[0]


def _logs(name_contains):
    """Every self.log(...) whose first arg mentions `name_contains`, as {kwarg: source}."""
    out = []
    for node in ast.walk(_training_step()):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "log" and node.args):
            continue
        tag = ast.unparse(node.args[0])
        if name_contains in tag:
            out.append((tag, {k.arg: ast.unparse(k.value) for k in node.keywords}))
    return out


def test_the_indicator_logs_exist():
    """Negative control: on the pre-fix version this FAILS (no picked_rung logs at all)."""
    assert _logs("picked_rung"), "no explore/picked_rung* logging found"


def test_indicators_are_ddp_synced():
    """Without sync_dist the epoch value is rank 0's alone -- a quarter of the data."""
    for tag, kw in _logs("picked_rung") + _logs("best_loss_rung"):
        assert kw.get("sync_dist") == "True", f"{tag} is not sync_dist=True"


def test_indicators_are_epoch_accumulated_not_step_sampled():
    """on_epoch=True accumulates EVERY step; on_step=True would add ~22k useless TB points."""
    for tag, kw in _logs("picked_rung") + _logs("best_loss_rung"):
        assert kw.get("on_epoch") == "True", f"{tag} must be on_epoch=True"
        assert kw.get("on_step") == "False", f"{tag} must be on_step=False"


def test_the_raw_index_series_is_NOT_sync_dist():
    """⭐ The load-bearing one. Averaging a rung INDEX across ranks yields a non-rung (mean of
    {0,3,1,2} = 1.5) and silently destroys the histogram. selected_rung/selected_tau must stay
    rank-local per-step traces."""
    for tag, kw in _logs("selected_rung") + _logs("selected_tau"):
        assert kw.get("sync_dist") in (None, "False"), (
            f"{tag} uses sync_dist -- that averages the rung INDEX across ranks and is meaningless")


def test_one_indicator_per_rung_driven_by_K():
    """Hardcoding 4 would silently under-report if --explore_k ever changed."""
    src = textwrap.dedent(inspect.getsource(OpenFoldWrapper.training_step))
    assert "for _r in range(_K):" in src, "indicators must be emitted per rung from _K"
    assert 'f"explore/picked_rung{_r}"' in src


def test_oracle_and_selector_use_the_same_estimator():
    """They are only comparable if measured identically -- same n, same reduction."""
    p = dict(_logs("picked_rung"))
    b = dict(_logs("best_loss_rung"))
    assert p and b
    (_, pk), (_, bk) = list(p.items())[0], list(b.items())[0]
    for key in ("sync_dist", "on_epoch", "on_step"):
        assert pk.get(key) == bk.get(key), f"selector/oracle differ on {key}"
