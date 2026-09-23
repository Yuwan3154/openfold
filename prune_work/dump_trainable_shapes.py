"""Dump a Lightning checkpoint's parameter shapes (registration order) + trainable flags, on CPU.

Input for prune_work/ddp_desync_lightning_harness.py, which rebuilds a model with the SAME parameter
tensors so DDP's bucket layout matches the real run. Trainable = exactly what
pruned_evoformer.freeze_all_except_evoformer leaves trainable (--freeze_non_evoformer): model.evoformer.*
and model.recycling_embedder.contractive_pair_update.*. Cross-checked against the checkpoint's own Adam
state: param_groups index i is the i-th model.parameters() tensor, and only trainable tensors carry exp_avg.

  CUDA_VISIBLE_DEVICES= python prune_work/dump_trainable_shapes.py --ckpt <last.ckpt> --out shapes.json
"""

import argparse
import json

import torch

TRAINABLE_PREFIXES = ("model.evoformer.", "model.recycling_embedder.contractive_pair_update.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", mmap=True, weights_only=False)
    sd = ck["state_dict"]
    names = list(sd.keys())
    entries = [{"name": n, "shape": list(sd[n].shape), "dtype": str(sd[n].dtype),
                "trainable": n.startswith(TRAINABLE_PREFIXES)} for n in names]
    train = [e for e in entries if e["trainable"]]
    n_train, numel_train = len(train), sum(int(torch.Size(e["shape"]).numel()) for e in train)
    print("state_dict tensors", len(entries), "numel", sum(int(torch.Size(e["shape"]).numel()) for e in entries))
    print("dtypes", sorted({e["dtype"] for e in entries}))
    print("trainable tensors", n_train, "numel", numel_train)

    # Adam state is keyed by the index into param_groups' flat list, which is model.parameters() order
    opt = ck["optimizer_states"][0]
    flat = [i for g in opt["param_groups"] for i in g["params"]]
    has_state = {i for i, s in opt["state"].items() if "exp_avg" in s}
    print("optimizer param_groups params", len(flat), "with exp_avg", len(has_state))
    assert len(flat) == len(entries), (len(flat), len(entries))
    mism = 0
    for pos, idx in enumerate(flat):
        e = entries[pos]
        if (idx in has_state) != e["trainable"]:
            mism += 1
        elif idx in has_state and list(opt["state"][idx]["exp_avg"].shape) != e["shape"]:
            mism += 1
    print("trainable-flag or exp_avg-shape mismatches vs optimizer state:", mism, "of", len(flat))

    with open(args.out, "w") as fh:
        json.dump({"ckpt": args.ckpt, "epoch": ck["epoch"], "global_step": ck["global_step"],
                   "n_tensors": len(entries), "n_trainable": n_train, "numel_trainable": numel_train,
                   "params": entries}, fh)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
