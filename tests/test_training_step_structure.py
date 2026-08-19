"""Structural guard on train_openfold.py's training_step.

⛔⛔ WHY THIS EXISTS. The best-of-K helpers were first inserted into the MIDDLE of `training_step`'s
body at method indentation (4 spaces inside an 8-space block). Python accepted it: the dedent simply
ENDED `training_step` early and re-parented the rest of its body onto `_explore_confidence`. Nothing
failed to import, nothing failed to parse, every unit test still passed -- and the run died 18 seconds
in with Lightning's "Skipping the training_step by returning None in distributed training is not
supported", because the truncated `training_step` fell off the end and returned None.

⭐ The general hazard: a string-insertion edit can silently RESTRUCTURE Python via indentation, and
neither `ast.parse` nor an import will complain. Anything that edits this file by text substitution
needs a structural assertion, not just a syntax check.
"""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "train_openfold.py"


@pytest.fixture(scope="module")
def wrapper_methods():
    tree = ast.parse(SRC.read_text())
    for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
        fns = {f.name: f for f in cls.body if isinstance(f, ast.FunctionDef)}
        if "training_step" in fns and "_rng_snapshot" in fns:
            return fns
    pytest.fail("no class defining both training_step and _rng_snapshot")


def test_training_step_ends_in_a_return(wrapper_methods):
    """Lightning treats a None return as 'skip this step', which is a hard error under DDP."""
    body = wrapper_methods["training_step"].body
    assert isinstance(body[-1], ast.Return), (
        "training_step must end with `return loss`; falling off the end returns None and DDP raises"
    )


def test_training_step_still_contains_its_whole_body(wrapper_methods):
    src = ast.unparse(wrapper_methods["training_step"])
    for needle in ("outputs = self(batch)", "self.loss(", "loss_breakdown", "_explore",
                   "gt_features"):
        assert needle in src, f"training_step lost `{needle}` -- likely re-parented by a bad edit"


def test_helpers_did_not_swallow_the_training_body(wrapper_methods):
    """The exact failure that happened: training_step's tail was re-parented onto this helper."""
    for name in ("_explore_confidence", "_rng_snapshot", "_rng_restore"):
        src = ast.unparse(wrapper_methods[name])
        assert "gt_features" not in src and "loss_breakdown" not in src, (
            f"{name} contains training_step's body -- the methods were inserted mid-function"
        )


def test_helpers_are_methods_of_the_same_class(wrapper_methods):
    for name in ("_rng_snapshot", "_rng_restore", "_explore_confidence"):
        assert wrapper_methods[name].args.args[0].arg == "self"
