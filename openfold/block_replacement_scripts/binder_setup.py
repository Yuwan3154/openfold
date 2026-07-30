"""BindCraft-faithful 2-chain binder setup for AFdistill end-to-end caching.

build_binder_complex_feats: target chain = its real structure as an AF2 template (seq+sidechains kept),
binder chain = hallucinated (no template). Reuses openfold's make_sequence_features_with_custom_template
+ multimer pair_and_merge (same pipeline that built cache_2chain_bf16). Chain order A=target, B=binder
=> residues [0:tL]=target, [tL:tL+bL]=binder.

bindcraft_loss: the BindCraft 4stage hallucination objective, ported exactly from ColabDesign af/loss.py
(verified bin definitions identical between openfold and ColabDesign): weighted sum
plddt 0.1 + pae 0.4 + i_pae 0.1 + con 1.0 + i_con 1.0 over the binder/interface.
"""
import os
import tempfile
import shutil
import torch
import torch.nn.functional as F
import openfold.np.residue_constants as rc
from openfold.config import model_config
from openfold.data import (
    data_pipeline,
    feature_pipeline,
    templates,
    feature_processing_multimer,
    msa_pairing,
    parsers,
)
import hallucination_straight_through as H

KALIGN = "/usr/bin/kalign"
_DROP = {
    "target_feat", "msa_feat", "aatype", "residue_index", "no_recycling_iters",
    "atom14_atom_exists", "residx_atom14_to_atom37", "residx_atom37_to_atom14", "atom37_atom_exists",
}
BINDCRAFT_WEIGHTS = dict(plddt=0.1, pae=0.4, i_pae=0.1, con=1.0, i_con=1.0)
CON_INTRA = dict(cutoff=14.0, num=2, seqsep=9)   # BindCraft 4stage intra contacts
CON_INTER = dict(cutoff=20.0, num=2)             # BindCraft 4stage interface contacts


def _add_all_seq(feats, seq, desc):
    msa = parsers.Msa(sequences=[seq], deletion_matrix=[[0] * len(seq)], descriptions=[desc])
    allf = data_pipeline.make_msa_features([msa])
    valid = set(msa_pairing.MSA_FEATURES) | {"msa_species_identifiers"}
    for k, v in allf.items():
        if k in valid:
            feats[f"{k}_all_seq"] = v
    return feats


def build_binder_complex_feats(complex_id, target_seq, binder_seq, target_cif, target_chain, device,
                               config_preset="model_1_multimer_v3"):
    """Sequence-INDEPENDENT 2-chain const feats (template, asym/entity/sym ids, masks) at msa-depth 1,
    recycle-dim 1. Returns (const, tL, bL, target_onehot[tL,20], residue_index[N])."""
    config = model_config(config_preset, train=False, low_prec=False)
    config.data.predict.masked_msa_replace_fraction = 0
    config.data.predict.max_extra_msa = 1
    tmpl_dir = tempfile.mkdtemp()
    with open(os.path.join(tmpl_dir, "dummy.cif"), "w") as f:
        f.write(H._DUMMY_CIF)
    tf = templates.HhsearchHitFeaturizer(
        mmcif_dir=tmpl_dir, max_template_date="2025-01-01", max_hits=0,
        kalign_binary_path=KALIGN, release_dates_path=None, obsolete_pdbs_path=None,
    )
    data_processor = data_pipeline.DataPipeline(template_featurizer=tf)
    feature_processor = feature_pipeline.FeaturePipeline(config.data)

    tcif = os.path.expanduser(target_cif)
    tgt = data_pipeline.make_sequence_features_with_custom_template(
        sequence=target_seq, mmcif_path=tcif, pdb_id=complex_id, chain_id=target_chain,
        kalign_binary_path=KALIGN, skip_alignment=True,
    )
    align = tempfile.mkdtemp()
    bdir = os.path.join(align, "binder")
    os.makedirs(bdir, exist_ok=True)
    bfasta = os.path.join(align, "binder.fasta")
    with open(bfasta, "w") as fp:
        fp.write(f">binder\n{binder_seq}\n")
    with open(os.path.join(bdir, "output.a3m"), "w") as fp:
        fp.write(f">binder\n{binder_seq}\n")
    bnd = data_processor.process_fasta(fasta_path=bfasta, alignment_dir=bdir, seqemb_mode=False)
    shutil.rmtree(align)

    tgt = _add_all_seq(tgt, target_seq, f"{complex_id}_t")
    bnd = _add_all_seq(bnd, binder_seq, f"{complex_id}_b")
    tgt = data_pipeline.convert_monomer_features(tgt, chain_id="A")
    bnd = data_pipeline.convert_monomer_features(bnd, chain_id="B")
    all_chain = data_pipeline.add_assembly_features({"A": tgt, "B": bnd})
    np_example = data_pipeline.pad_msa(
        feature_processing_multimer.pair_and_merge(all_chain_features=all_chain), 512,
    )
    proc = feature_processor.process_features(np_example, mode="predict", is_multimer=True)
    proc = {k: torch.as_tensor(v, device=device) for k, v in proc.items()}
    shutil.rmtree(tmpl_dir)

    n_clusters = proc["msa_feat"].shape[0]
    const = {}
    for k, v in proc.items():
        if k in _DROP:
            continue
        if v.dim() >= 1 and v.shape[0] == n_clusters:
            v = v[0:1]
        const[k] = v[..., 0:1].contiguous()

    tL, bL = len(target_seq), len(binder_seq)
    target_onehot = F.one_hot(
        torch.tensor([rc.restype_order.get(c, 0) for c in target_seq], device=device), 20
    ).float()
    residue_index = torch.cat(
        [torch.arange(tL, device=device), torch.arange(bL, device=device)]
    ).long()
    return const, tL, bL, target_onehot, residue_index


def make_binder_batch(binder_probs20, target_onehot, residue_index, const_feats, recycle_dim=1):
    """Differentiable full-complex feature batch; gradient flows ONLY to binder_probs20."""
    seq_probs20 = torch.cat([target_onehot, binder_probs20], dim=0)
    return H._make_feature_batch_multimer(seq_probs20, residue_index, recycle_dim, const_feats)


# ---------- BindCraft loss (exact ColabDesign math; bins verified identical to openfold) ----------

def _dgram_bins(device):
    return torch.cat([torch.zeros(1, device=device), torch.linspace(2.3125, 21.6875, 63, device=device)])


def _con_term(dgram, bins, cutoff, binary=False):
    contact = (bins < cutoff).to(dgram.dtype)
    px = F.softmax(dgram, dim=-1)
    px_ = F.softmax(dgram - 1e7 * (1 - contact), dim=-1)
    cat_ent = -(px_ * F.log_softmax(dgram, dim=-1)).sum(-1)
    bin_ent = -torch.log((contact * px).sum(-1) + 1e-8)
    return bin_ent if binary else cat_ent


def _min_k(x, k, mask):
    """Mean of the k smallest masked entries along the last axis (ColabDesign min_k)."""
    y = torch.where(mask.bool(), x, torch.full_like(x, float("nan")))
    y, _ = torch.sort(y, dim=-1)  # NaNs sort to the end
    idx = torch.arange(y.shape[-1], device=x.device).expand_as(y)
    km = (idx < k) & ~torch.isnan(y)
    return torch.where(km, y, torch.zeros_like(y)).sum(-1) / (km.sum(-1).clamp_min(1) + 1e-8)


def _con_loss(dgram, residue_index, opt, mask_1d, mask_1b):
    bins = _dgram_bins(dgram.device)
    offset = residue_index[:, None] - residue_index[None, :]
    p = _con_term(dgram, bins, opt["cutoff"], binary=False)
    m = (offset.abs() >= opt["seqsep"]) if "seqsep" in opt else torch.ones_like(offset, dtype=torch.bool)
    m = m & mask_1b[None, :].bool()
    p = _min_k(p, opt["num"], m)
    return _min_k(p, p.shape[-1], mask_1d)  # num_pos = inf -> mean over mask_1d


def _plddt(lddt_logits):
    nb = lddt_logits.shape[-1]
    bw = 1.0 / nb
    centers = torch.arange(0.5 * bw, 1.0, bw, device=lddt_logits.device)
    return (F.softmax(lddt_logits, dim=-1) * centers).sum(-1)


def _pae(tm_logits):
    nb = tm_logits.shape[-1]
    breaks = torch.linspace(0, 31, nb - 1, device=tm_logits.device)
    step = breaks[1] - breaks[0]
    centers = breaks + step / 2
    centers = torch.cat([centers, centers[-1:] + step])
    return (F.softmax(tm_logits, dim=-1) * centers).sum(-1)


def _mask_loss_2d(x, mask2d):
    return (x * mask2d).sum() / (mask2d.sum() + 1e-8)


def bindcraft_loss(out, tL, bL, residue_index, model=None, weights=None):
    """ColabDesign binder-hallucination Φ over the binder/interface. Returns (total, terms_dict)."""
    weights = weights or BINDCRAFT_WEIGHTS
    dgram = out["distogram_logits"] if "distogram_logits" in out else model.aux_heads.distogram(out["pair"])
    tm = out["tm_logits"] if "tm_logits" in out else model.aux_heads.tm(out["pair"])
    lddt = out["lddt_logits"] if "lddt_logits" in out else model.aux_heads.plddt(out["single"])
    assert dgram.dim() == 3 and lddt.dim() == 2, f"unexpected shapes {dgram.shape} {lddt.shape}"
    device = dgram.device
    N = tL + bL
    binder_id = torch.zeros(N, device=device)
    binder_id[tL:] = 1.0
    target_id = torch.zeros(N, device=device)
    target_id[:tL] = 1.0

    plddt = (1.0 - _plddt(lddt))
    plddt = (plddt * binder_id).sum() / (binder_id.sum() + 1e-8)

    pae = _pae(tm) / 31.0
    pae = (pae + pae.transpose(-1, -2)) / 2
    pae_intra = _mask_loss_2d(pae, binder_id[:, None] * binder_id[None, :])
    i_pae = _mask_loss_2d(pae, binder_id[:, None] * target_id[None, :])

    con = _con_loss(dgram, residue_index, CON_INTRA, mask_1d=binder_id, mask_1b=binder_id)
    i_con = _con_loss(dgram, residue_index, CON_INTER, mask_1d=binder_id, mask_1b=target_id)

    terms = {"plddt": plddt, "pae": pae_intra, "i_pae": i_pae, "con": con, "i_con": i_con}
    total = sum(weights[k] * v for k, v in terms.items())
    return total, {k: float(v) for k, v in terms.items()}
