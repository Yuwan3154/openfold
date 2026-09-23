"""E3 hardening: DataLoader workers share tensors by `file_descriptor`, with the soft fd limit raised to hard.

The known-bad control is the reason `file_system` was chosen in the first place: `file_descriptor` at the
A6000's measured default soft limit (1024) runs out of fds once the parent holds more live shared storages
than that. The same loader must then complete after configure_worker_sharing(), which proves the raise is
load-bearing. Every configuration runs in its own interpreter (tests/worker_sharing_probe.py).
"""

import json
import os
import resource
import subprocess
import sys

from openfold.utils import worker_sharing

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROBE = os.path.join(REPO_ROOT, "tests", "worker_sharing_probe.py")
OLD_SOFT_LIMIT = 1024  # `ulimit -Sn` on the A6000, measured 2026-09-23 (hard 1048576)
# hang guard only (the loader probe takes ~1 s on the A6000), not a tuned value
PROBE_TIMEOUT_S = 300


def _probe(mode):
    # PYTHONPATH pins the imports to THIS checkout: the env also carries a site-packages openfold (~/openfold)
    env = dict(os.environ, PYTHONPATH=REPO_ROOT, CUDA_VISIBLE_DEVICES="")
    p = subprocess.run([sys.executable, PROBE, mode], cwd=REPO_ROOT, env=env, capture_output=True, text=True,
                       timeout=PROBE_TIMEOUT_S)
    print(f"--- probe {mode}: rc={p.returncode}\n{p.stdout}\n{p.stderr[-2000:]}")
    line = next(x for x in p.stdout.splitlines() if x.startswith("probe imports:"))
    imports = json.loads(line.split("probe imports:", 1)[1])
    assert imports["worker_sharing"].startswith(REPO_ROOT), imports
    assert imports["data_modules"].startswith(REPO_ROOT), imports
    return p


def test_imports_are_this_checkout():
    print("worker_sharing:", worker_sharing.__file__)
    assert worker_sharing.__file__.startswith(REPO_ROOT), (worker_sharing.__file__, REPO_ROOT)


def test_configure_selects_file_descriptor_and_raises_soft_to_hard():
    p = _probe("configure")
    assert p.returncode == 0
    out = json.loads(p.stdout.splitlines()[-1])
    soft, hard = out["rlimit_nofile"]
    assert out["strategy"] == "file_descriptor"
    assert soft == hard == out["configured"]["hard"] == out["configured"]["soft_after"]


def test_openfold_loader_completes_with_more_live_shared_tensors_than_the_old_soft_limit():
    p = _probe("new")
    assert p.returncode == 0, "file_descriptor after configure_worker_sharing() must serve the whole epoch"
    out = json.loads(p.stdout.splitlines()[-1])
    assert out["strategy"] == "file_descriptor"
    assert out["live_shared_tensors"] > OLD_SOFT_LIMIT, out


def test_known_bad_control_file_descriptor_at_the_old_soft_limit_runs_out_of_fds():
    p = _probe("old_fd1024")
    assert p.returncode != 0
    # torch DataLoader._try_get_data's own EMFILE message
    assert "Too many open files" in p.stderr, p.stderr[-2000:]


def test_infinite_hard_limit_uses_the_kernel_cap(monkeypatch):
    calls = []
    monkeypatch.setattr(worker_sharing.resource, "getrlimit",
                        lambda _: (OLD_SOFT_LIMIT, resource.RLIM_INFINITY))
    monkeypatch.setattr(worker_sharing.resource, "setrlimit", lambda _, lim: calls.append(lim))
    monkeypatch.setattr(worker_sharing.mp, "set_sharing_strategy", lambda s: calls.append(s))
    worker_sharing.configure_worker_sharing()
    with open("/proc/sys/fs/nr_open") as fh:
        nr_open = int(fh.read())
    assert calls == [(nr_open, resource.RLIM_INFINITY), "file_descriptor"]
