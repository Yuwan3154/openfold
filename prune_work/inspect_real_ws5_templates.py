"""WS6 follow-up: what do WS5's REAL training-time templates actually look like? Runs the exact
same template-selection code path WS5's dataloader uses (openfold.data.templates.
HhsearchHitFeaturizer, same mmcif_dir/max_template_date/kalign/release_dates/obsolete_pdbs as
run_prune_singleseq.sh) against the REAL pdb70_hits.hhr files cached for WS5's own validation
chains, then measures TM-score + Kabsch RMSD of the selected template(s) vs. the chain's own true
(native) structure. This is the honest answer to "how close are eval templates to GT" -- unlike
the templates-ablation self-template (which is trivially TM=1.0/RMSD=0A by construction), these
are genuine hhsearch hits, subject to the SAME date-cutoff + duplicate-exclusion filtering AF2's
real training protocol applies (verified in openfold/data/templates.py::_assess_hhsearch_hit).

NOTE: uses _shuffle_top_k_prefiltered=None (deterministic top-sum_probs order) rather than WS5's
train-time random shuffle, since we want a reproducible read of "which real templates would be
available/top-ranked", not a specific random draw. Real training also randomly subsamples the
template COUNT down to 0-4 per example (data_transforms.random_crop_to_size); this script always
reports what's available up to max_hits=4 (matching EVAL-mode config: subsample_templates=False,
"we want top templates" -- i.e., what WS5's OWN validation actually saw).
"""
import csv
import datetime
import os
import sys

import numpy as np

BASE = "/home/jupyter-chenxi"
sys.path.insert(0, f"{BASE}/openfold")
from openfold.data import mmcif_parsing, parsers
from openfold.data.templates import HhsearchHitFeaturizer, _get_atom_positions
from openfold.np import residue_constants as rc

VAL_LIST = f"{BASE}/prune_work/lists_pdb/slim_struct_val.list"
ALN_DIR = f"{BASE}/data/openproteinset_aln"
MMCIF_DIR = f"{BASE}/data/pdb_mmcif/mmcif_files"
OBS = f"{BASE}/data/pdb_mmcif/obsolete.dat"
CACHE = f"{BASE}/data/pdb_mmcif/mmcif_cache.json"
KALIGN = f"{BASE}/miniconda3/envs/cue_openfold_gated/bin/kalign"
MAX_TEMPLATE_DATE = "2018-04-30"  # matches run_prune_singleseq.sh exactly
N_SAMPLE = int(os.environ.get("N_SAMPLE", "20"))
OUT_CSV = os.environ.get(
    "OUT_CSV", "/home/jupyter-chenxi/prune_work/eval_out/real_template_quality.csv")

CA_IDX = rc.atom_order["CA"]


def kabsch_rmsd(a, b):
    a = a - a.mean(0)
    b = b - b.mean(0)
    u, s, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    a_aligned = a @ (vt.T @ np.diag([1, 1, d]) @ u.T).T
    return float(np.sqrt(((a_aligned - b) ** 2).sum(-1).mean()))


def tm_score(a, b):
    """Standard Zhang & Skolnick TM-score, computed over an ALREADY-matched (by index)
    correspondence -- Kabsch-align a onto b, then apply the standard d0(L) normalization."""
    L = len(a)
    a_c, b_c = a - a.mean(0), b - b.mean(0)
    u, s, vt = np.linalg.svd(a_c.T @ b_c)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    a_aligned = a_c @ (vt.T @ np.diag([1, 1, d]) @ u.T).T
    di = np.sqrt(((a_aligned - b_c) ** 2).sum(-1))
    d0 = 1.24 * (max(L, 19) - 15) ** (1 / 3) - 1.8
    d0 = max(d0, 0.5)
    return float(np.mean(1.0 / (1.0 + (di / d0) ** 2)))


def main():
    with open(VAL_LIST) as f:
        val_chains = [l.strip() for l in f if l.strip()]

    featurizer = HhsearchHitFeaturizer(
        mmcif_dir=MMCIF_DIR,
        max_template_date=MAX_TEMPLATE_DATE,
        max_hits=4,
        kalign_binary_path=KALIGN,
        release_dates_path=CACHE,
        obsolete_pdbs_path=OBS,
        _shuffle_top_k_prefiltered=None,
    )

    rows = []
    for pdb_chain in val_chains:
        if len(rows) >= N_SAMPLE:
            break
        pdbid, chain = pdb_chain.split("_")
        hhr_path = os.path.join(ALN_DIR, pdb_chain, "pdb70_hits.hhr")
        if not os.path.exists(hhr_path):
            print(f"{pdb_chain}: no pdb70_hits.hhr, skipping", flush=True)
            continue
        mmcif_path = os.path.join(MMCIF_DIR, f"{pdbid}.cif")
        if not os.path.exists(mmcif_path):
            print(f"{pdb_chain}: no local mmcif at {mmcif_path}, skipping", flush=True)
            continue
        try:
            with open(mmcif_path) as f:
                cif_string = f.read()
            parsed = mmcif_parsing.parse(file_id=pdbid, mmcif_string=cif_string)
            obj = parsed.mmcif_object
            if obj is None or chain not in obj.chain_to_seqres:
                print(f"{pdb_chain}: mmcif parse/chain-lookup failed", flush=True)
                continue
            seq = obj.chain_to_seqres[chain]
            real_pos, real_mask = _get_atom_positions(
                obj, chain, max_ca_ca_distance=150.0, _zero_center_positions=False)
            real_ca = real_pos[:, CA_IDX, :]
            real_valid = real_mask[:, CA_IDX].astype(bool)

            with open(hhr_path) as f:
                hits = parsers.parse_hhr(f.read())
            result = featurizer.get_templates(query_sequence=seq, hits=hits)
            feats = result.features
            n_templ = feats["template_aatype"].shape[0]
            if n_templ == 0:
                rows.append({"pdb_chain": pdb_chain, "length": len(seq), "n_templates": 0,
                             "top1_domain": "", "top1_tm": "", "top1_rmsd": "", "top1_coverage": ""})
                print(f"{pdb_chain}: 0 real templates passed filtering (date/duplicate/length)", flush=True)
                continue

            top1_mask = feats["template_all_atom_mask"][0, :, CA_IDX].astype(bool)
            matched = top1_mask & real_valid
            if matched.sum() < 5:
                print(f"{pdb_chain}: top1 template has <5 overlapping resolved CA positions, skipping metric", flush=True)
                continue
            templ_ca = feats["template_all_atom_positions"][0, :, CA_IDX, :]
            rmsd = kabsch_rmsd(templ_ca[matched], real_ca[matched])
            tm = tm_score(templ_ca[matched], real_ca[matched])
            domain = feats["template_domain_names"][0]
            if isinstance(domain, bytes):
                domain = domain.decode()
            rows.append({
                "pdb_chain": pdb_chain, "length": len(seq), "n_templates": int(n_templ),
                "top1_domain": domain, "top1_tm": tm, "top1_rmsd": rmsd,
                "top1_coverage": float(matched.sum()) / len(seq),
            })
            print(f"{pdb_chain}: n_templ={n_templ} top1={domain} TM={tm:.3f} RMSD={rmsd:.2f}A "
                  f"cov={matched.sum()}/{len(seq)}", flush=True)
        except Exception as e:
            print(f"{pdb_chain}: failed: {e}", flush=True)
            continue

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows -> {OUT_CSV}", flush=True)

    with_templ = [r for r in rows if r["n_templates"] and r["top1_tm"] != ""]
    zero_templ = [r for r in rows if not r["n_templates"]]
    print(f"chains with >=1 usable real template: {len(with_templ)}/{len(rows)}")
    print(f"chains with 0 real templates (post-filter): {len(zero_templ)}/{len(rows)}")
    if with_templ:
        tms = [r["top1_tm"] for r in with_templ]
        rmsds = [r["top1_rmsd"] for r in with_templ]
        print(f"top-1 real template TM-score: mean={np.mean(tms):.3f} median={np.median(tms):.3f} "
              f"min={min(tms):.3f} max={max(tms):.3f}")
        print(f"top-1 real template RMSD:     mean={np.mean(rmsds):.2f}A median={np.median(rmsds):.2f}A "
              f"min={min(rmsds):.2f}A max={max(rmsds):.2f}A")


if __name__ == "__main__":
    main()
