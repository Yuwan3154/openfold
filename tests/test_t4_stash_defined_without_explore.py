"""T4 without exploration (--explore_k 1, or epochs before explore_after_epoch) must not hit an unbound `_stash`.

train_openfold needs a GPU to import, so this reads training_step's AST: `_stash` must be bound at the
function's top level before the `if _explore:` block, since the T4 block reads it on every step.
Known-bad control: the pre-fix source (927c68d) binds it only inside `if _explore:`.
"""

import ast
import os
import subprocess

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _stash_bound_before_explore(src):
    fn = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "training_step")
    for stmt in fn.body:
        if isinstance(stmt, ast.If) and isinstance(stmt.test, ast.Name) and stmt.test.id == "_explore":
            return False
        if isinstance(stmt, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_stash" for t in stmt.targets):
            return True
    raise AssertionError("no `if _explore:` block found in training_step")


def test_stash_bound_before_explore_branch():
    with open(os.path.join(REPO_ROOT, "train_openfold.py")) as fh:
        assert _stash_bound_before_explore(fh.read())


def test_known_bad_control_pre_fix_source():
    old = subprocess.run(["git", "show", "927c68d:train_openfold.py"], cwd=REPO_ROOT, capture_output=True,
                         text=True, check=True).stdout
    assert not _stash_bound_before_explore(old)
