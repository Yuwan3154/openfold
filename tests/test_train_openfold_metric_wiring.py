"""Static wiring checks on train_openfold.py (parsed with ast: importing it needs a GPU).

⛔ The invariant behind the val-metric cross-pairing (RAW Phase M §6(3)) and the E4 hang (T-1): every DDP rank
must issue the same collectives in the same order. A synced self.log key is a collective created on its first
log, so its NAME must never depend on batch content, and a new collective must never sit in a rank- or
data-dependent branch.
"""

import ast
import os

import pytest

from tests.test_val_population_metrics import GROUPS, SOURCE_NAMES

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_OPENFOLD = os.path.join(REPO_ROOT, "train_openfold.py")
# every synced key's name template: loss terms, the unconditional val/{metric} (incl. the checkpoint monitor
# val/lddt_ca) and the rung indicators -- none depends on which entries a rank saw
SYNCED_NAMES = {"f'{phase}/{loss_name}'", "f'{phase}/{loss_name}_epoch'", "f'{phase}/{k}'",
                "f'explore/picked_rung{_r}'", "f'explore/best_loss_rung{_r}'"}


@pytest.fixture(scope="module")
def tree():
    print("parsed:", TRAIN_OPENFOLD)
    return ast.parse(open(TRAIN_OPENFOLD).read())


@pytest.fixture(scope="module")
def hooks(tree):
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "OpenFoldWrapper")
    return {f.name: f for f in cls.body if isinstance(f, ast.FunctionDef)}


def _self_logs(tree):
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "log"
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "self" and node.args):
            yield node


def test_parsing_this_checkout():
    assert TRAIN_OPENFOLD.startswith(REPO_ROOT) and os.path.isfile(TRAIN_OPENFOLD)


def test_synced_keys_are_exactly_the_batch_independent_ones(tree):
    synced = set()
    for call in _self_logs(tree):
        kw = {k.arg: k.value for k in call.keywords}
        if "sync_dist" in kw and not (isinstance(kw["sync_dist"], ast.Constant) and kw["sync_dist"].value is False):
            synced.add(ast.unparse(call.args[0]))
    print("synced name templates:", sorted(synced))
    assert synced == SYNCED_NAMES


def test_the_per_population_val_keys_are_gone(tree):
    names = [ast.unparse(c.args[0]) for c in _self_logs(tree)]
    assert not [n for n in names if "{suffix}" in n or "_src_" in n], names


def test_new_helpers_are_called_in_their_hooks(hooks):
    required = {
        "validation_step": ["population_records(batch, _metrics, VAL_SOURCE_NAMES)"],
        "on_validation_epoch_start": ["self._val_pop_records = []"],
        "on_validation_epoch_end": ["gather_records(", "population_means(", "f'val_pop/{k}'"],
        "training_step": ["self._train_sums.add_t4_step(", "self._train_sums.add_explore_step(",
                          "self._train_sums.add_t4_promoted(", "picked=_j == _pick", "picked=True"],
        "on_train_epoch_start": ["self._train_sums.reset()"],
        "on_train_epoch_end": ["self._train_sums.all_reduce(", "train_epoch_metrics("],
        "_log": ["return other_metrics"],
    }
    for hook, needles in required.items():
        assert hook in hooks, f"{hook} missing"
        src = ast.unparse(hooks[hook])
        for needle in needles:
            assert needle in src, f"{hook} does not contain `{needle}`"
    assert ast.unparse(hooks["training_step"]).count("has_template=bool(m['has_template'][i])") == 2


@pytest.mark.parametrize("hook,call", [("on_validation_epoch_end", "gather_records("),
                                       ("on_train_epoch_end", "self._train_sums.all_reduce(")])
def test_new_collectives_run_unconditionally_on_every_rank(hooks, hook, call):
    top = [s for s in hooks[hook].body if call in ast.unparse(s)]
    nested = [n for n in ast.walk(hooks[hook]) if isinstance(n, (ast.If, ast.For, ast.While, ast.With, ast.Try))
              and call in ast.unparse(n)]
    assert len(top) == 1 and not nested, f"{call} must be a top-level statement of {hook}"
    # the only switch is the world size, which every rank shares
    assert "self.trainer.world_size > 1" in ast.unparse(top[0])


def test_worker_sharing_replaces_file_system(tree):
    src = ast.unparse(tree)
    assert "set_sharing_strategy('file_system')" not in src
    module_calls = [ast.unparse(s) for s in tree.body if isinstance(s, (ast.Assign, ast.Expr))]
    assert "_WORKER_SHARING = configure_worker_sharing()" in module_calls
    assert any("worker sharing:" in c and c.startswith("rank_zero_info(") for c in module_calls)


def test_val_pop_groups_and_source_names_match_the_helper_tests(tree):
    ns = {}
    for s in tree.body:
        if isinstance(s, ast.Assign) and [ast.unparse(t) for t in s.targets] in (["VAL_SOURCE_NAMES"],
                                                                                 ["VAL_POP_GROUPS"]):
            exec(compile(ast.Module([s], []), TRAIN_OPENFOLD, "exec"), ns)
    assert ns["VAL_SOURCE_NAMES"] == SOURCE_NAMES
    assert ns["VAL_POP_GROUPS"] == GROUPS
