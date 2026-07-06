"""Build real AF2 template features for a target chain, placed at a specific offset within a
longer concatenated query sequence (e.g. [binder_seq, target_seq]) -- the rest of the query
(the binder) is left as an unmapped/gap template position, exactly matching Bennett et al.
2023's "AF2 initial guess" convention (real target structure templated, binder blank).

Reuses OpenFold's own template pipeline (openfold/data/templates.py, data_transforms.py)
rather than re-deriving the feature construction -- this module only wires them together.
"""
import sys

import torch

BASE = "/home/jupyter-chenxi"
sys.path.insert(0, f"{BASE}/openfold")
from openfold.data import mmcif_parsing
from openfold.data.templates import _extract_template_features
from openfold.data import data_transforms as dt

KALIGN = f"{BASE}/miniconda3/envs/cue_openfold_gated/bin/kalign"


def build_template_features(mmcif_path, pdb_id, target_chain_id, query_sequence, target_offset, device):
    """query_sequence = the FULL concatenated query (e.g. binder_seq + target_seq).
    target_offset = the index within query_sequence where the target chain's sequence starts
    (so mapping covers query_sequence[target_offset : target_offset + len(target_seq)])."""
    with open(mmcif_path) as f:
        cif_string = f.read()
    parsed = mmcif_parsing.parse(file_id=pdb_id, mmcif_string=cif_string)
    mmcif_object = parsed.mmcif_object
    target_seq = mmcif_object.chain_to_seqres[target_chain_id]

    mapping = {target_offset + i: i for i in range(len(target_seq))}

    raw, warning = _extract_template_features(
        mmcif_object=mmcif_object,
        pdb_id=pdb_id,
        mapping=mapping,
        template_sequence=target_seq,
        query_sequence=query_sequence,
        template_chain_id=target_chain_id,
        kalign_binary_path=KALIGN,
        skip_alignment=False,
    )
    if warning:
        print(f"template_feat_builder warning: {warning}", flush=True)

    num_res = len(query_sequence)
    protein = {
        "template_aatype": torch.as_tensor(raw["template_aatype"], dtype=torch.float32)[None],
        "template_all_atom_positions": torch.as_tensor(raw["template_all_atom_positions"], dtype=torch.float32)[None],
        "template_all_atom_mask": torch.as_tensor(raw["template_all_atom_mask"], dtype=torch.float32)[None],
    }
    protein = dt.fix_templates_aatype(protein)          # one-hot -> OpenFold-ordered indices
    protein = dt.make_template_mask(protein)            # template_mask = ones[n_templ]
    protein = dt.make_pseudo_beta(prefix="template_")(protein)          # @curry1
    protein = dt.atom37_to_torsion_angles(prefix="template_")(protein)  # @curry1

    assert protein["template_aatype"].shape == (1, num_res)
    assert protein["template_all_atom_positions"].shape == (1, num_res, 37, 3)

    return {k: v.to(device) for k, v in protein.items()}
