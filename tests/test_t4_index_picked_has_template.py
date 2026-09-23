"""The T4 index records which sample trained (`picked`) and whether a template was given (`has_template`).

Without them t4/tm_pred, margin and promote_rate can only be approximated offline (window audit A used a
mean over the K rungs as a proxy): the index held every rung's tm_pred but not which rung the step used,
and a no-template step was indistinguishable from a template scoring 0.
"""

import json
import os

import numpy as np
import pytest

from openfold.np import residue_constants as rc
from openfold.utils import t4_pool
from openfold.utils.t4_pool import PromotedTemplatePool, PromotedTemplateWriter

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
L = 6
QUERY = "ACDEFG"
K = 4  # Run C v2's --explore_k


def _submit(w, step, sample, picked, has_template):
    aat = np.array([rc.restype_order[c] for c in QUERY], np.int8)
    msk = np.zeros((L, 37), bool)
    msk[:, :3] = True
    w.submit(chain="1abc_A", epoch=0, step=step, tm_pred=0.5 + 0.1 * sample, tm_template=0.25,
             coords37=np.full((L, 37, 3), float(sample), np.float32), atom_mask37=msk, aatype=aat,
             residue_index=np.arange(L), sample=sample, picked=picked, has_template=has_template)


def test_imports_are_this_checkout():
    print("t4_pool:", t4_pool.__file__)
    assert t4_pool.__file__.startswith(REPO_ROOT), (t4_pool.__file__, REPO_ROOT)


def test_records_carry_picked_and_has_template(tmp_path):
    w = PromotedTemplateWriter(str(tmp_path), rank=0)
    for step, (pick, has) in enumerate([(2, True), (0, False)]):   # promote-all: K rungs per step
        for k in range(K):
            _submit(w, step, k, picked=(k == pick), has_template=has)
    w.close()
    recs = [json.loads(x) for x in (tmp_path / "rank0/index.jsonl").read_text().splitlines() if x]
    print(recs[0])
    assert len(recs) == 2 * K
    assert [r["sample"] for r in recs if r["picked"]] == [2, 0], "exactly one picked sample per step"
    assert [r["has_template"] for r in recs] == [True] * K + [False] * K
    # the rank-0 t4/tm_pred of each step is now exactly recoverable from the index
    assert [r["tm_pred"] for r in recs if r["picked"]] == pytest.approx([0.7, 0.5])
    # and the read side is unaffected by the extra keys
    assert PromotedTemplatePool(str(tmp_path)).refresh() == 2 * K


def test_submit_without_the_new_fields_fails_loudly(tmp_path):
    w = PromotedTemplateWriter(str(tmp_path), rank=0)
    msk = np.zeros((L, 37), bool)
    with pytest.raises(TypeError):
        w.submit("1abc_A", 0, 0, 0.5, 0.25, np.zeros((L, 37, 3), np.float32), msk,
                 np.zeros(L, np.int8), np.arange(L))
    w.close()
