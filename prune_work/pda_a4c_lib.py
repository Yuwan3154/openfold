"""A4c shared library: gap-concat multi-chain feature building (Minkyung Baek "AlphaFold-Gap" trick,
already used elsewhere in this repo as process_multiseq_fasta's ri_gap pattern) + GT-structure
template injection for a chosen subset of chains (openfold.data.templates.get_custom_template_features,
the same kalign-realign mechanism --single_seq_keep_templates uses everywhere else in this project).

Design chain is ALWAYS slot 0 (never templated) so scoring code can always read
out["final_atom_positions"][0, :design_len] as the design chain's own prediction, regardless of arm.
"""
import numpy as np
import torch

from openfold.data import data_pipeline, mmcif_parsing, templates


def kabsch_rmsd(a, b):
    """Verbatim from prune_work/eval_pda_self_consistency.py -- reimplemented (not imported) since
    that module transitively imports single_seq_infer.py, which hardcodes
    sys.path.insert(0, "/home/jupyter-chenxi/openfold/openfold/block_replacement_scripts") -- a
    DIFFERENT, unrelated worktree -- as a side effect of import, silently poisoning `openfold`
    resolution for everything imported afterward (caught 2026-07-15 via a real reshape crash inside
    embed_templates whose traceback pointed at that other worktree's template.py)."""
    a = a - a.mean(0)
    b = b - b.mean(0)
    u, s, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    a_aligned = a @ (vt.T @ np.diag([1, 1, d]) @ u.T).T
    return float(np.sqrt(((a_aligned - b) ** 2).sum(-1).mean()))
from openfold.np import residue_constants as rc

RI_GAP = 200


def build_slots_from_components(components):
    """components: list of {seq, copies, is_design, template(optional dict)}.
    Returns an ordered slot list, design chain first."""
    design = [c for c in components if c["is_design"]]
    assert len(design) == 1, f"expected exactly one design component, got {len(design)}"
    dc = design[0]
    others = [c for c in components if not c["is_design"]]
    slots = [{"seq": dc["seq"], "template": None}]
    for _ in range(dc["copies"] - 1):
        slots.append({"seq": dc["seq"], "template": None})
    for c in others:
        for _ in range(c["copies"]):
            slots.append({"seq": c["seq"], "template": c.get("template")})
    return slots


def _template_block_for_slot(seq, template_info, kalign_binary_path):
    res = templates.get_custom_template_features(
        mmcif_path=template_info["mmcif_path"],
        query_sequence=seq,
        pdb_id=template_info["pdb_id"],
        chain_id=template_info["chain_id"],
        kalign_binary_path=kalign_binary_path,
    )
    return res.features


def build_multichain_features(slots, kalign_binary_path, ri_gap=RI_GAP):
    input_sequence = "".join(s["seq"] for s in slots)
    num_res = len(input_sequence)

    seq_feats = data_pipeline.make_sequence_features(
        sequence=input_sequence, description="a4c_multichain", num_res=num_res)
    offset = 0
    for s in slots:
        offset += len(s["seq"])
        seq_feats["residue_index"][offset:] += ri_gap

    msa_feats = data_pipeline.make_dummy_msa_feats(input_sequence)

    # Ground-truth base fields the standard feature_pipeline schema always expects present
    # (common_cfg.feat), even in pure-inference eval mode -- downstream transforms derive
    # atom14_gt_*/rigidgroups_gt_*/chi_angles_* etc. from these. Zero-filled here: we score against
    # ground truth fetched SEPARATELY (get_native_design_coords), so these never feed the network
    # (GT coords are loss-only fields, not model inputs) and zero/masked is safe for every slot.
    gt_feats = {
        "all_atom_positions": np.zeros((num_res, rc.atom_type_num, 3), dtype=np.float32),
        "all_atom_mask": np.zeros((num_res, rc.atom_type_num), dtype=np.float32),
        "resolution": np.array([0.], dtype=np.float32),
        "is_distillation": np.array(0., dtype=np.float32),
    }

    # ALWAYS build exactly 1 template row (never n_templates=0): with cfg.data.eval.fixed_size=False
    # (chosen so the design chain, slot 0, is never cropped), the standard make_fixed_size transform
    # that normally pads the template dim to >=1 slot never runs -- a genuinely 0-sized template
    # tensor hits a real PyTorch ambiguous-reshape edge case inside TemplatePointwiseAttention's
    # chunked path (`reshape(-1, 0, ...)`, caught 2026-07-15 via full traceback). A single FULLY
    # MASKED template (all-gap aatype, zero positions/mask) is semantically identical to "no
    # template info" -- it contributes nothing to the pair/single embeddings either way -- so this
    # is not a behavior change, just avoiding the empty-tensor shape.
    n_aatype = len(rc.restypes_with_x_and_gap)
    template_aatype = np.zeros((1, num_res, n_aatype), dtype=np.float32)
    template_aatype[..., rc.HHBLITS_AA_TO_ID['-']] = 1
    template_all_atom_positions = np.zeros((1, num_res, rc.atom_type_num, 3), dtype=np.float32)
    template_all_atom_mask = np.zeros((1, num_res, rc.atom_type_num), dtype=np.float32)

    cur = 0
    for s in slots:
        L = len(s["seq"])
        if s.get("template") is not None:
            block = _template_block_for_slot(s["seq"], s["template"], kalign_binary_path)
            template_aatype[:, cur:cur + L] = block["template_aatype"]
            template_all_atom_positions[:, cur:cur + L] = block["template_all_atom_positions"]
            template_all_atom_mask[:, cur:cur + L] = block["template_all_atom_mask"]
        cur += L

    template_feats = {
        "template_aatype": template_aatype,
        "template_all_atom_positions": template_all_atom_positions,
        "template_all_atom_mask": template_all_atom_mask,
        "template_pseudo_beta_mask": np.zeros((1, num_res), dtype=np.float32),
        "template_pseudo_beta": np.zeros((1, num_res, 3), dtype=np.float32),
        "template_dgram_probs": np.zeros((1, num_res, num_res, 39), dtype=np.float32),
        "template_domain_names": np.array([b"a4c"], dtype=object),
        "template_sequence": np.array([input_sequence.encode()], dtype=object),
        "template_sum_probs": np.array([[1.0]], dtype=np.float32),
    }

    return {**seq_feats, **msa_feats, **gt_feats, **template_feats}


def build_cfg(model_config_fn, crop_size):
    """finetuning_ptm already has model.template.enabled=True / data.common.use_templates=True by
    default -- only single-seq MSA clamp needs explicit override.

    fixed_size=True (the DEFAULT, same as pda_baseline_full.py and the whole v4/v5/WS5-continued
    PDA harness) is used deliberately, NOT fixed_size=False: with fixed_size=False, several arrays
    (template, extra_msa) can legitimately end up genuinely 0-sized for a single-template/single-seq
    batch, which hits a real PyTorch "reshape(-1, 0, ...) is ambiguous" edge case inside chunked
    attention paths that the standard fixed_size=True padding (make_fixed_size) always avoids by
    guaranteeing a non-zero minimum size. Caller MUST pass crop_size >= this run's own max
    concatenated length (from the actual per-arm manifest survey, never invented) so no entry -- in
    particular the design chain, always slot 0 -- gets truncated; random_crop_to_size is a no-op
    when the real length already fits within crop_size."""
    cfg = model_config_fn("finetuning_ptm", train=False, low_prec=False)
    cfg.globals.chunk_size = None
    for g in ["use_deepspeed_evo_attention", "use_lma", "use_flash"]:
        setattr(cfg.globals, g, False)
    cfg.data.common.max_recycling_iters = 3
    cfg.data.common.max_extra_msa = 1
    cfg.data.common.max_msa_clusters = 1
    cfg.data.eval.crop_size = crop_size
    return cfg


def build_cfg_stock(model_config_fn, crop_size):
    """Stock AF2, mirroring pda_baseline_full.py's build_cfg_stock() exactly (model_1_ptm preset,
    jax weights, templates explicitly OFF, single-seq clamp) -- only crop_size is parameterized
    per-entry here instead of the fixed 256 that script uses for isolated single-chain folding."""
    cfg = model_config_fn("model_1_ptm", train=False, low_prec=False)
    cfg.globals.chunk_size = None
    for g in ["use_deepspeed_evo_attention", "use_lma", "use_flash"]:
        setattr(cfg.globals, g, False)
    cfg.data.common.max_recycling_iters = 3
    cfg.data.common.max_extra_msa = 1
    cfg.data.common.max_msa_clusters = 1
    cfg.data.eval.crop_size = crop_size
    cfg.model.template.enabled = False
    cfg.data.common.use_templates = False
    cfg.data.common.use_template_torsion_angles = False
    return cfg


def get_native_design_coords(cif_cache_dir, pdb, chain_id):
    """Ground-truth CA coordinates + per-residue validity mask for the design chain, aligned 1:1 by
    index to chain_to_seqres[chain_id] (same ordering used to build slot 0's input sequence) --
    caller must apply `valid` to BOTH predicted and native arrays positionally, never compact/
    truncate either array independently first, or internal (non-trailing) unresolved residues in
    the deposited structure silently desync the two arrays (same convention pda_baseline_full.py
    uses: mask in place, don't filter-then-compare)."""
    with open(f"{cif_cache_dir}/{pdb}.cif") as f:
        mmcif_string = f.read()
    parsed = mmcif_parsing.parse(file_id=pdb, mmcif_string=mmcif_string)
    mo = parsed.mmcif_object
    all_atom_positions, all_atom_mask = mmcif_parsing.get_atom_coords(mmcif_object=mo, chain_id=chain_id)
    ca_idx = rc.atom_order["CA"]
    ca = all_atom_positions[:, ca_idx, :]
    valid = all_atom_mask[:, ca_idx].astype(bool)
    return ca, valid
