"""The noise ladder must not leak its scale out of training_step.

⛔⛔ THE BUG THIS PINS (found 2026-08-22, cost a full epoch of misread results):
`training_step` sets `model.config.recycling_embedder.gaussian_pair_init_scale = _ladder[_j]` for each
sample and again to `_ladder[_pick]` for the grad-carrying replay. `AlphaFold.forward` reads that field
on EVERY call (model.py:286), and the config object outlives the step -- so the LAST training step of an
epoch handed its winning rung's scale straight to VALIDATION. Run C's epoch-0 val therefore measured the
model at tau in {0, 8, 16, 32} instead of the configured 1.0, came in 0.047 lDDT low, and read as a real
regression. The damage concentrated on short chains, which have the fewest pair elements to average the
noise over -- which is what made it look like a genuine length-dependent model failure.

These tests are deliberately written against the SOURCE rather than a live training step: constructing a
real AlphaFold + DDP batch is far out of unit-test reach, but the invariant ("every mutation of that
field inside training_step is followed by a restore before the method returns") is exactly checkable.
"""

import ast
import inspect
import textwrap

import train_openfold
from train_openfold import OpenFoldWrapper

FIELD = "gaussian_pair_init_scale"


def _training_step_ast():
    src = textwrap.dedent(inspect.getsource(OpenFoldWrapper.training_step))
    return ast.parse(src).body[0]


def _scale_assignments(node):
    """Every `....gaussian_pair_init_scale = <rhs>` in the tree, as (lineno, rhs_source)."""
    out = []
    for n in ast.walk(node):
        if not isinstance(n, ast.Assign):
            continue
        for t in n.targets:
            if isinstance(t, ast.Attribute) and t.attr == FIELD:
                out.append((n.lineno, ast.unparse(n.value)))
    return out


def test_the_field_is_mutated_at_all():
    """Negative control: if this fails the test below is vacuous."""
    assert _scale_assignments(_training_step_ast()), \
        "training_step no longer touches the scale; these guards need rewriting"


def test_the_original_scale_is_captured_before_the_ladder_runs():
    src = textwrap.dedent(inspect.getsource(OpenFoldWrapper.training_step))
    assert "_scale0" in src, "training_step must snapshot the pre-ladder scale"
    tree = _training_step_ast()
    capture = [n.lineno for n in ast.walk(tree)
               if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "_scale0" for t in n.targets)]
    assert capture, "no `_scale0 = ...` capture found"
    muts = [ln for ln, _ in _scale_assignments(tree)]
    assert min(capture) < min(muts), \
        "the snapshot must be taken BEFORE the first ladder mutation"


def test_the_last_assignment_restores_the_snapshot_not_a_rung():
    """⭐ The load-bearing one. The FINAL write to the field on the way out of training_step must be
    the restore. On the buggy version the last write was `_ladder[_pick]`, so this FAILS there."""
    muts = _scale_assignments(_training_step_ast())
    assert muts, "no assignments found"
    last_line, last_rhs = max(muts, key=lambda x: x[0])
    assert last_rhs == "_scale0", (
        f"the last write to {FIELD} in training_step is `{last_rhs}`, not `_scale0` -- the ladder's "
        f"scale leaks into validation")


def test_every_ladder_rung_write_is_followed_by_the_restore():
    muts = _scale_assignments(_training_step_ast())
    rung_writes = [ln for ln, rhs in muts if "_ladder" in rhs]
    restores = [ln for ln, rhs in muts if rhs == "_scale0"]
    assert rung_writes, "expected the ladder to write the field"
    assert restores, "expected at least one restore"
    assert max(restores) > max(rung_writes), \
        "a ladder rung is written after the final restore"


def test_restore_is_guarded_by_the_same_condition_as_the_mutation():
    """The restore must be inside `if _ladder is not None:` -- an unconditional restore would clobber
    the configured scale on the non-ladder path (Run B's semantics), and an unguarded mutation with a
    guarded restore would leak."""
    tree = _training_step_ast()
    guarded = 0
    for n in ast.walk(tree):
        if not isinstance(n, ast.If):
            continue
        test_src = ast.unparse(n.test)
        if "_ladder" not in test_src:
            continue
        for a in _scale_assignments(n):
            if a[1] == "_scale0":
                guarded += 1
    assert guarded >= 1, "the restore is not inside an `if _ladder ...` guard"


def test_module_defines_the_scale_default_as_one():
    """Sanity: the configured default the restore returns to."""
    import openfold.config as cfg
    c = cfg.model_config("finetuning_ptm", train=True, low_prec=False)
    assert float(getattr(c.model.recycling_embedder, FIELD, 1.0)) == 1.0


def test_training_step_still_ends_in_a_return():
    """⛔ A string-insertion edit can silently re-parent the tail of a method via indentation, and
    neither a syntax check nor an import catches it (this project has been bitten). Re-assert the
    structural invariant after touching training_step."""
    tree = _training_step_ast()
    assert isinstance(tree, ast.FunctionDef)
    assert isinstance(tree.body[-1], ast.Return), \
        "training_step no longer ends in a Return -- indentation may have re-parented its tail"
    assert hasattr(train_openfold.OpenFoldWrapper, "_explore_confidence")
