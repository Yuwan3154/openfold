"""Faithful PyTorch port of BindCraft/ColabDesign monomer DESIGN losses (cd_loss.py) onto OpenFold
outputs. NO approximation. Terms: con, pae, plddt, rg, helix. Monomer => interface terms dropped.
References (cached ColabDesign source): get_plddt/get_pae/get_con_loss/_get_con_loss/get_helix_loss
in af/loss.py; add_rg_loss in BindCraft colabdesign_utils.py."""
import torch
import torch.nn.functional as F

WEIGHTS = {"con": 1.0, "pae": 0.4, "plddt": 0.1, "rg": 0.3, "helix": -0.3}
CON_OPT = {"num": 2, "cutoff": 14.0, "binary": False, "seqsep": 9}  # BindCraft default_4stage con


def dgram_bins_64(device, dtype=torch.float32):
    # cd_loss.get_dgram_bins, 64-bin branch: append(0, linspace(2.3125,21.6875,63))
    return torch.cat([torch.zeros(1, device=device, dtype=dtype),
                      torch.linspace(2.3125, 21.6875, 63, device=device, dtype=dtype)])


def _get_con_loss(dgram, dbins, cutoff, binary):
    # dgram [L,L,nb]; dbins [nb]. cd_loss._get_con_loss
    bins = (dbins < cutoff).to(dgram.dtype)                  # [nb]
    px = F.softmax(dgram, -1)
    px_ = F.softmax(dgram - 1e7 * (1 - bins), -1)
    cat_ent = -(px_ * F.log_softmax(dgram, -1)).sum(-1)      # [L,L]
    bin_ent = -torch.log((bins * px).sum(-1) + 1e-8)         # [L,L]
    return bin_ent if binary else cat_ent


def _min_k(x, k, mask):
    # cd_loss.min_k: per-row k-smallest masked mean. x[...,N], mask[...,N] bool. k may be inf.
    y = torch.where(mask, x, torch.full_like(x, float("nan")))
    ys, _ = torch.sort(y, dim=-1)                            # NaN sorts to the end (ascending)
    N = ys.shape[-1]
    ar = torch.arange(N, device=x.device)
    kmask = (ar < k) & (~torch.isnan(ys))
    return torch.where(kmask, ys, torch.zeros_like(ys)).sum(-1) / (kmask.sum(-1).to(x.dtype) + 1e-8)


def get_pae(tm_logits, max_bin=31, no_bins=64):
    # cd_loss.get_pae: breaks + step/2, append last+step  (== OpenFold _calculate_bin_centers)
    breaks = torch.linspace(0, max_bin, no_bins - 1, device=tm_logits.device, dtype=tm_logits.dtype)
    step = breaks[1] - breaks[0]
    centers = breaks + step / 2
    centers = torch.cat([centers, (centers[-1] + step).unsqueeze(0)])
    return (F.softmax(tm_logits, -1) * centers).sum(-1)      # [L,L]


def get_plddt(lddt_logits):
    # cd_loss.get_plddt: bin_centers = arange(0.5*bw, 1.0, bw), bw=1/nb
    nb = lddt_logits.shape[-1]
    bw = 1.0 / nb
    centers = torch.arange(0.5 * bw, 1.0, bw, device=lddt_logits.device, dtype=lddt_logits.dtype)
    return (F.softmax(lddt_logits, -1) * centers).sum(-1)    # [L]


def bc_losses(out, residue_index, ca_xyz, con_opt=CON_OPT):
    """out: dict with distogram_logits [L,L,64], tm_logits [L,L,64], lddt_logits [L,50].
    ca_xyz: [L,3] predicted CA (structure module). residue_index: [L]."""
    dgram = out["distogram_logits"]
    dbins = dgram_bins_64(dgram.device, dgram.dtype)
    Ln = dgram.shape[0]
    idx = residue_index.flatten().to(dgram.device)
    offset = idx[:, None] - idx[None, :]

    # con: min over `num` nearest partners per row (|offset|>=seqsep), then mean over positions (num_pos=inf)
    con_mtx = _get_con_loss(dgram, dbins, con_opt["cutoff"], binary=con_opt["binary"])   # [L,L]
    m = (offset.abs() >= con_opt["seqsep"])
    p = _min_k(con_mtx, con_opt["num"], m)                                               # [L]
    allmask = torch.ones(1, Ln, dtype=torch.bool, device=dgram.device)
    con = _min_k(p.unsqueeze(0), float("inf"), allmask)[0]

    # pae (intra): /31, symmetrized, mean
    pae = get_pae(out["tm_logits"]) / 31.0
    pae = (pae + pae.t()) / 2
    pae_loss = pae.mean()

    # plddt: 1 - plddt, mean
    plddt_loss = (1.0 - get_plddt(out["lddt_logits"])).mean()

    # rg: elu(rg - 2.38*N^0.365)
    rg = torch.sqrt(((ca_xyz - ca_xyz.mean(0)) ** 2).sum(-1).mean() + 1e-8)
    rg_th = 2.38 * (ca_xyz.shape[0] ** 0.365)
    rg_loss = F.elu(rg - rg_th)

    # helix: binary con at cutoff 6.0 on the (i, i+3) diagonal (offset==3)
    hx = _get_con_loss(dgram, dbins, 6.0, binary=True)                                   # [L,L]
    hmask = (offset == 3)
    helix = torch.where(hmask, hx, torch.zeros_like(hx)).sum() / (hmask.sum().to(hx.dtype) + 1e-8)

    return {"con": con, "pae": pae_loss, "plddt": plddt_loss, "rg": rg_loss, "helix": helix}


def total_loss(losses, weights=WEIGHTS):
    return sum(weights[k] * losses[k] for k in weights)
