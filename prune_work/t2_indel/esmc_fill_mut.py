"""ESMC fill-in + MUTATION arm (user 10-08): mask the INSERTED positions and a random set of surviving positions (plans.json rec['mut'], drawn by
make_mild_inputs.py --mut-frac-*) and predict all masked positions in ONE ESMC forward pass; sample each position over the 20 standard aa
(order of ops as the ESM sampler: top-p on the raw logits, then softmax(logits/T); esmc_fill_pilot.sample_positions). At MUTATED positions the
native residue is EXCLUDED from the distribution (logit -inf) so every mutation is a real substitution (my choice, logged). The alignment to the native is
known exactly (orig_idx), so no alignment is needed. Writes JSON [{chain, draw, arm, seq (full new-frame sequence), n_ins, n_mut}].
Env: esmfold2 (esm + Bio). Run: python esmc_fill_mut.py --inputs-dir D --out f.json --esmc esmc_600m --chains .. --draws .. [--temp 0.7 --top-p 1.0]
"""
import argparse
import json
import os
import zlib

import numpy as np
import torch
from esm.models.esmc import ESMC

from esmc_fill_pilot import AA, THREE2ONE, sample_positions, sampler_test, selftest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--esmc", required=True, choices=["esmc_300m", "esmc_600m"])
    p.add_argument("--chains", nargs="+", required=True)
    p.add_argument("--draws", type=int, nargs="+", required=True)
    p.add_argument("--temp", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=1.0)
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = ESMC.from_pretrained(a.esmc, device=torch.device(dev)).eval()
    tok = model.tokenizer
    ids_of = {c: tok.convert_tokens_to_ids(c) for c in AA}
    plans = {c: json.load(open(os.path.join(a.inputs_dir, c, "plans.json"))) for c in a.chains}
    nat = {c: "".join(THREE2ONE[r] for r in plans[c]["native_resnames"]) for c in a.chains}
    selftest(model, tok, ids_of, [nat[c] for c in a.chains], np.random.default_rng(0))
    sampler_test()
    out = []
    for c in a.chains:
        for k in a.draws:
            rec = plans[c]["plans"][k]
            orig = np.array(rec["orig_idx"])
            mut = np.array(rec["mut"]["new_idx"], dtype=int)
            assert (orig[mut] >= 0).all(), (c, k, "mutation on an inserted residue")
            masked = np.union1d(np.flatnonzero(orig < 0), mut)
            ids = [tok.cls_token_id] + [tok.mask_token_id if j in set(masked.tolist()) else ids_of[nat[c][o]] for j, o in enumerate(orig)] + [tok.eos_token_id]
            with torch.no_grad():
                lg = model(sequence_tokens=torch.tensor([ids]).to(dev)).sequence_logits[0, 1:-1].float().cpu()
            lg = lg[:, [ids_of[x] for x in AA]]
            for j in mut:
                lg[j, AA.index(nat[c][orig[j]])] = -float("inf")
            rng = np.random.default_rng([zlib.crc32(c.encode()), k, 93])
            pick = sample_positions(lg[masked], a.temp, a.top_p, rng)
            seq = [nat[c][o] if o >= 0 else None for o in orig]
            for j, i in zip(masked, pick):
                seq[j] = AA[i]
            assert all(s is not None for s in seq)
            assert all(seq[j] != nat[c][orig[j]] for j in mut), (c, k, "a mutation kept the native residue")
            out.append(dict(chain=c, draw=k, arm=a.esmc, seq="".join(seq), n_ins=int((orig < 0).sum()), n_mut=int(len(mut))))
            print(c, k, "ins", out[-1]["n_ins"], "mut", out[-1]["n_mut"], "L'", len(seq), flush=True)
    json.dump(out, open(a.out, "w"))
    print(f"wrote {len(out)} sequences -> {a.out}")


if __name__ == "__main__":
    main()
