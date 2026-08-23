"""--checkpoint_save_top_k must actually reach ModelCheckpoint.

⛔⛔ THE BUG THIS PINS (2026-08-23): the flag was declared at the argparse layer and NEVER read;
`save_top_k=1` was hardcoded. Every launcher passing `--checkpoint_save_top_k 5` was a silent no-op, and
the startup banner printed a literal "[1]" that looked like a coincidence rather than the truth.

With top_k=1 each new best DELETES the previous one, so a monitor that disagrees with the benchmark can
irrecoverably discard the better model. That happened: best-000 (PDA 0.7619) was replaced by best-001
(PDA 0.7613) because the monitored 906-entry mean rose while PDA fell.

Source-level tests: constructing a Trainer + callbacks needs the full arg surface, but the invariant
("save_top_k is derived from the flag, and the banner reports the derived value") is exactly checkable.
"""

import ast
import inspect
import textwrap

import train_openfold


def _main_src():
    return textwrap.dedent(inspect.getsource(train_openfold.main))


def test_the_flag_is_declared():
    """Negative control: if the flag vanished the rest of this file is vacuous."""
    src = inspect.getsource(train_openfold)
    assert '"--checkpoint_save_top_k"' in src


def test_save_top_k_is_not_a_hardcoded_literal():
    """⭐ The load-bearing test. On the buggy version `save_top_k=1` appears as a literal in the
    ModelCheckpoint call for the BEST checkpoint, and this FAILS."""
    tree = ast.parse(_main_src())
    offenders = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "id", "") == "ModelCheckpoint"):
            continue
        kw = {k.arg: k.value for k in node.keywords}
        # the periodic/rolling callback legitimately hardcodes save_top_k=0 with save_last=True
        last = kw.get("save_last")
        if isinstance(last, ast.Constant) and last.value is True:
            continue
        v = kw.get("save_top_k")
        assert v is not None, "the best-checkpoint callback must set save_top_k"
        if isinstance(v, ast.Constant):
            offenders.append(v.value)
    assert not offenders, (
        f"best-checkpoint save_top_k is a hardcoded literal {offenders}; "
        f"--checkpoint_save_top_k cannot take effect")


def test_the_flag_is_actually_read_in_main():
    src = _main_src()
    assert "checkpoint_save_top_k" in src, \
        "main() never reads the flag, so it is a no-op no matter what a launcher passes"


def test_default_none_maps_to_one_preserving_old_behaviour():
    """⚠️ Every existing launcher that does NOT pass the flag must be byte-identical to before."""
    src = _main_src()
    assert "_top_k = 1 if _top_k is None else int(_top_k)" in src, \
        "the None default must map to 1, or unrelated launchers change behaviour"


def test_banner_reports_the_derived_value_not_a_literal():
    """The banner printed a literal '[1]' for three launches while I passed 5 and never noticed.
    It must report what was actually configured."""
    src = _main_src()
    assert "[top_k={_top_k}]" in src, "the startup banner must report the derived top_k"
    assert "({monitor_mode}) [1]" not in src, "banner still prints a hardcoded [1]"


def test_monitor_is_still_configurable():
    """Guard against a fix to one flag breaking its neighbour."""
    src = _main_src()
    assert "checkpoint_monitor" in src
