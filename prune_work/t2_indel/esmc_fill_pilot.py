"""ESMC one-shot fill pilot (user 10-07): mask the inserted positions of the Gly-arm edit plans, predict all of them in ONE
forward pass, sample each position independently over the 20 standard aa. Logits do not depend on T/top-p, so each
(chain, draw) costs one forward pass. Order of ops follows the ESM sampler: top-p on the raw logits (keep cumsum <= p, at least the top-1, as esm/utils/sampling.py), then softmax(logits/T).
QC alignment is LOCAL BLOSUM62 open -10 extend -0.5 (gap values from raygun_edit.py, local mode is new, provisional); aln_ident is divided by len(fill);
nll_T1 is renormalised over the 20 aa; null_p has resolution 1/N_NULL and 0.0 means fewer than 1/N_NULL.
QC is reported as DISTRIBUTIONS (no pass/fail cut-offs); the null for the copy check is composition-matched shuffles of the fill.
Env: esmfold2. Run: python esmc_fill_pilot.py --inputs-dir ~/t2_indel_data/inputs --out-dir OUT --chains .. --draws ..
"""
import argparse
import csv
import json
import math
import os
import zlib
from collections import Counter

import numpy as np
import torch
from Bio.Align import PairwiseAligner, substitution_matrices
from esm.models.esmc import ESMC

AA = "ACDEFGHIKLMNPQRSTVWY"
THREE2ONE = {"ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F", "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K",
             "LEU": "L", "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R", "SER": "S", "THR": "T", "VAL": "V",
             "TRP": "W", "TYR": "Y"}
TEMPS = [0.1, 0.3, 0.5, 0.7, 1.0, 1.3]
TOPPS = [0.9, 1.0]
N_NULL = 20


def sample_positions(logits, temp, top_p, rng):
    """logits: (n, 20) raw; returns sampled aa indices."""
    out = []
    for row in logits:
        probs = torch.softmax(row, -1).numpy()
        order = np.argsort(-probs)
        keep = order[: max(1, int((np.cumsum(probs[order]) <= top_p).sum()))]
        masked = np.full_like(row.numpy(), -np.inf)
        masked[keep] = row.numpy()[keep]
        p = torch.softmax(torch.tensor(masked) / temp, -1).numpy().astype(np.float64)
        out.append(rng.choice(20, p=p / p.sum()))
    return np.array(out)


def longest_run(s):
    best = cur = 1
    for a, b in zip(s, s[1:]):
        cur = cur + 1 if a == b else 1
        best = max(best, cur)
    return best


def entropy_bits(s):
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in Counter(s).values())


def best_local(aligner, seg, native):
    aln = aligner.align(native, seg)[0]
    ident = sum(a == b for ra, rb in zip(*aln.aligned) for a, b in zip(native[ra[0]:ra[1]], seg[rb[0]:rb[1]]))
    cov = sum(rb[1] - rb[0] for rb in aln.aligned[1]) / len(seg)
    return aln.score, ident / len(seg), cov


def kmer_overlap(seg, native, k=3):
    ks = {seg[i:i + k] for i in range(len(seg) - k + 1)}
    nk = {native[i:i + k] for i in range(len(native) - k + 1)}
    return len(ks & nk) / len(ks) if ks else float("nan")


def qc(seg, native, aligner, rng, nll):
    score, ident, cov = best_local(aligner, seg, native)
    null = [best_local(aligner, "".join(rng.permutation(list(seg))), native)[0] for _ in range(N_NULL)]
    return dict(len=len(seg), longest_run=longest_run(seg), entropy_bits=entropy_bits(seg),
                frac_GSA=sum(c in "GSA" for c in seg) / len(seg), nll_T1=nll, aln_score=score, aln_ident=ident, aln_cov=cov,
                null_p=float(np.mean([n >= score for n in null])), kmer3_overlap=kmer_overlap(seg, native))


def segments(orig):
    segs, cur = [], []
    for j, o in enumerate(orig):
        if o < 0:
            cur.append(j)
        elif cur:
            segs.append(cur)
            cur = []
    if cur:
        segs.append(cur)
    return segs


def sampler_test():
    row = torch.log(torch.tensor([[.6, .3, .1] + [1e-9] * 17]))
    rng = np.random.default_rng(0)
    assert (sample_positions(row, 0.01, 1.0, rng) == 0).all()
    drawn = {int(sample_positions(row, 1.0, 0.95, rng)[0]) for _ in range(300)}
    assert drawn <= {0, 1} and drawn == {0, 1}, drawn
    print("sampler_test ok", flush=True)


def selftest(model, tok, ids_of, native_seqs, rng):
    seq = native_seqs[0]
    pos = rng.choice(len(seq), size=max(10, len(seq) // 10), replace=False)
    ids = [tok.cls_token_id] + [tok.mask_token_id if i in set(pos) else ids_of[c] for i, c in enumerate(seq)] + [tok.eos_token_id]
    with torch.no_grad():
        lg = model(sequence_tokens=torch.tensor([ids]).to(model.device)).sequence_logits[0, 1:-1].float().cpu()
    pred = lg[:, [ids_of[a] for a in AA]].argmax(-1).numpy()
    rec = np.mean([AA[pred[i]] == seq[i] for i in pos])
    assert rec > 0.2, f"masked-recovery selftest {rec:.3f} not above 4x chance (0.05)"
    print(f"selftest: ESMC-300M masked recovery on {len(pos)} native positions = {rec:.3f} (chance 0.05)", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--chains", nargs="+", required=True)
    p.add_argument("--draws", type=int, nargs="+", required=True)
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = ESMC.from_pretrained("esmc_300m", device=torch.device(dev)).eval()
    tok = model.tokenizer
    ids_of = {c: tok.convert_tokens_to_ids(c) for c in AA}
    aligner = PairwiseAligner(mode="local", substitution_matrix=substitution_matrices.load("BLOSUM62"),
                              open_gap_score=-10, extend_gap_score=-0.5)
    plans = {c: json.load(open(os.path.join(a.inputs_dir, c, "plans.json"))) for c in a.chains}
    nat = {c: "".join(THREE2ONE[r] for r in plans[c]["native_resnames"]) for c in a.chains}
    selftest(model, tok, ids_of, [nat[c] for c in a.chains], np.random.default_rng(0))
    sampler_test()
    rows, fills = [], []
    for c in a.chains:
        for k in a.draws:
            orig = np.array(plans[c]["plans"][k]["orig_idx"])
            ins = np.where(orig < 0)[0]
            ids = [tok.cls_token_id] + [tok.mask_token_id if o < 0 else ids_of[nat[c][o]] for o in orig] + [tok.eos_token_id]
            with torch.no_grad():
                lg = model(sequence_tokens=torch.tensor([ids]).to(dev)).sequence_logits[0, 1:-1].float().cpu()
            lg = lg[ins][:, [ids_of[x] for x in AA]]
            lp = torch.log_softmax(lg, -1).numpy()
            assert len(ins) > 0, (c, k)
            pos = {j: r for r, j in enumerate(ins)}
            rng = np.random.default_rng([zlib.crc32(c.encode()), k, 91])
            arms = {"gly": np.full(len(ins), AA.index("G")),
                    "comp": np.array([AA.index(x) for x in np.random.default_rng([zlib.crc32(c.encode()), k, 77]).choice(list(nat[c]), size=len(ins))])}
            for T in TEMPS:
                for tp in TOPPS:
                    arms[f"esmc_T{T}_p{tp}"] = sample_positions(lg, T, tp, rng)
            if "comp_seq" in plans[c]["plans"][k]:
                assert "".join(AA[i] for i in arms["comp"]) == "".join(plans[c]["plans"][k]["comp_seq"][j] for j in ins)
            for arm, idx in arms.items():
                nll = -lp[np.arange(len(ins)), idx]
                full = {j: AA[i] for j, i in zip(ins, idx)}
                fills.append(dict(chain=c, draw=k, arm=arm, fill="".join(full[j] for j in ins)))
                for si, seg in enumerate(segments(orig)):
                    s = "".join(full[j] for j in seg)
                    rows.append(dict(chain=c, draw=k, arm=arm, seg=si, seq=s,
                                     **qc(s, nat[c], aligner, rng, float(np.mean([-lp[pos[j], AA.index(full[j])] for j in seg])))))
            print(c, k, "done", flush=True)
    with open(os.path.join(a.out_dir, "qc_segments.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    json.dump(fills, open(os.path.join(a.out_dir, "fills.json"), "w"))
    print(f"wrote {len(rows)} segment rows, {len(fills)} fills", flush=True)


if __name__ == "__main__":
    main()
