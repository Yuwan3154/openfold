"""WS6 follow-up: test the user's threshold hypothesis -- does WS5's actual FOLDING output
quality (RMSD-to-GT of the model's own prediction) collapse to no-template-like failure whenever
the real hhsearch-selected template is mediocre (TM<0.7), and only look good when the template is
genuinely close (TM>=0.7)? Earlier tests only measured the two extremes: a literal self-template
(TM=1.0 by construction) and no template at all. This combines inspect_real_ws5_templates.py's
real template SELECTION (actual hhsearch hits, actual OpenFold filtering) with
single_seq_infer.py's actual WS5 FOLDING forward pass using that real (imperfect) template, then
buckets folding-output RMSD-to-GT by template quality.
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from single_seq_infer import load_ws5, load_ws5_templated, score_sequence, score_sequence_templated, resolve_ws5_ckpt
from template_feat_builder import finish_template_features
from openfold.data import mmcif_parsing, parsers
from openfold.data.templates import HhsearchHitFeaturizer, _get_atom_positions
from openfold.np import residue_constants as rc

BASE = "/home/jupyter-chenxi"
CKPT = os.environ.get("CKPT") or None
DEVICE = os.environ.get("DEVICE", "cuda:0")
N_SAMPLE = int(os.environ.get("N_SAMPLE", "30"))
RMSD_THRESHOLD = 2.0
TM_GOOD_THRESHOLD = 0.7  # user-specified split: "good template" vs "poor template"

VAL_LIST = f"{BASE}/prune_work/lists_pdb/slim_struct_val.list"
ALN_DIR = f"{BASE}/data/openproteinset_aln"
MMCIF_DIR = f"{BASE}/data/pdb_mmcif/mmcif_files"
OBS = f"{BASE}/data/pdb_mmcif/obsolete.dat"
CACHE = f"{BASE}/data/pdb_mmcif/mmcif_cache.json"
KALIGN = f"{BASE}/miniconda3/envs/cue_openfold_gated/bin/kalign"
MAX_TEMPLATE_DATE = "2018-04-30"
OUT_CSV = os.environ.get(
    "OUT_CSV", "/home/jupyter-chenxi/prune_work/eval_out/real_template_folding.csv")

CA_IDX = rc.atom_order["CA"]


def kabsch_rmsd(a, b):
    a = a - a.mean(0)
    b = b - b.mean(0)
    u, s, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    a_aligned = a @ (vt.T @ np.diag([1, 1, d]) @ u.T).T
    return float(np.sqrt(((a_aligned - b) ** 2).sum(-1).mean()))


def tm_score(a, b):
    L = len(a)
    a_c, b_c = a - a.mean(0), b - b.mean(0)
    u, s, vt = np.linalg.svd(a_c.T @ b_c)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    a_aligned = a_c @ (vt.T @ np.diag([1, 1, d]) @ u.T).T
    di = np.sqrt(((a_aligned - b_c) ** 2).sum(-1))
    d0 = max(1.24 * (max(L, 19) - 15) ** (1 / 3) - 1.8, 0.5)
    return float(np.mean(1.0 / (1.0 + (di / d0) ** 2)))


def score_one_chain(pdb_chain, featurizer, model_templ, model_notempl):
    pdbid, chain = pdb_chain.split("_")
    hhr_path = os.path.join(ALN_DIR, pdb_chain, "pdb70_hits.hhr")
    mmcif_path = os.path.join(MMCIF_DIR, f"{pdbid}.cif")
    if not os.path.exists(hhr_path) or not os.path.exists(mmcif_path):
        return None

    with open(mmcif_path) as f:
        cif_string = f.read()
    parsed = mmcif_parsing.parse(file_id=pdbid, mmcif_string=cif_string)
    obj = parsed.mmcif_object
    if obj is None or chain not in obj.chain_to_seqres:
        return None
    seq = obj.chain_to_seqres[chain]
    real_pos, real_mask = _get_atom_positions(
        obj, chain, max_ca_ca_distance=150.0, _zero_center_positions=False)
    real_ca = real_pos[:, CA_IDX, :]
    real_valid = real_mask[:, CA_IDX].astype(bool)

    with open(hhr_path) as f:
        hits = parsers.parse_hhr(f.read())
    feats = featurizer.get_templates(query_sequence=seq, hits=hits).features
    n_templ = feats["template_aatype"].shape[0]

    if n_templ == 0:
        # No real template passed filtering -- fold with the NO-template model instead,
        # matching what WS5 would actually see for this chain (0 usable hits).
        bucket, domain, template_tm, template_rmsd = "no_template", "", None, None
        pred_ca = score_sequence(model_notempl, seq, device=DEVICE)["ca_coords"]
        plddt = None
    else:
        top1_mask = feats["template_all_atom_mask"][0, :, CA_IDX].astype(bool)
        matched = top1_mask & real_valid
        if matched.sum() < 5:
            return None
        templ_ca = feats["template_all_atom_positions"][0, :, CA_IDX, :]
        template_rmsd = kabsch_rmsd(templ_ca[matched], real_ca[matched])
        template_tm = tm_score(templ_ca[matched], real_ca[matched])
        domain = feats["template_domain_names"][0]
        if isinstance(domain, bytes):
            domain = domain.decode()
        bucket = "good_template" if template_tm >= TM_GOOD_THRESHOLD else "poor_template"

        template_feats = finish_template_features(
            feats["template_aatype"][0:1], feats["template_all_atom_positions"][0:1],
            feats["template_all_atom_mask"][0:1], DEVICE)
        out = score_sequence_templated(model_templ, seq, template_feats, device=DEVICE)
        pred_ca = out["ca_coords"]
        plddt = out["plddt_mean"]

    fold_rmsd = kabsch_rmsd(pred_ca[real_valid], real_ca[real_valid])
    return {
        "pdb_chain": pdb_chain, "length": len(seq), "n_templates": int(n_templ),
        "bucket": bucket, "top1_domain": domain,
        "template_tm": template_tm, "template_rmsd": template_rmsd,
        "fold_rmsd": fold_rmsd, "fold_success": fold_rmsd < RMSD_THRESHOLD,
        "fold_plddt": plddt,
    }


def main():
    with open(VAL_LIST) as f:
        val_chains = [l.strip() for l in f if l.strip()]

    featurizer = HhsearchHitFeaturizer(
        mmcif_dir=MMCIF_DIR, max_template_date=MAX_TEMPLATE_DATE, max_hits=4,
        kalign_binary_path=KALIGN, release_dates_path=CACHE, obsolete_pdbs_path=OBS,
        _shuffle_top_k_prefiltered=None,
    )
    model_templ = load_ws5_templated(CKPT or resolve_ws5_ckpt(), device=DEVICE)
    model_notempl = load_ws5(CKPT or resolve_ws5_ckpt(), device=DEVICE)
    print("both WS5 configs loaded (templates on + templates off)", flush=True)

    rows = []
    for pdb_chain in val_chains:
        if len(rows) >= N_SAMPLE:
            break
        try:
            row = score_one_chain(pdb_chain, featurizer, model_templ, model_notempl)
        except Exception as e:
            print(f"{pdb_chain}: failed: {e}", flush=True)
            continue
        if row is None:
            continue
        rows.append(row)
        tm_str = f"{row['template_tm']:.3f}" if row["template_tm"] is not None else "n/a"
        print(f"{pdb_chain}: bucket={row['bucket']} template_TM={tm_str} "
              f"fold_RMSD={row['fold_rmsd']:.2f}A success={row['fold_success']}", flush=True)

    if not rows:
        print("no rows scored -- aborting", flush=True)
        return

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows -> {OUT_CSV}", flush=True)

    for bucket in ["no_template", "poor_template", "good_template"]:
        b = [r for r in rows if r["bucket"] == bucket]
        if not b:
            print(f"{bucket}: n=0")
            continue
        n_success = sum(r["fold_success"] for r in b)
        rmsds = [r["fold_rmsd"] for r in b]
        print(f"{bucket}: n={len(b)}  recall@2A={n_success}/{len(b)} ({n_success/len(b):.3f})  "
              f"mean_fold_RMSD={np.mean(rmsds):.2f}A  median={np.median(rmsds):.2f}A")


if __name__ == "__main__":
    main()
