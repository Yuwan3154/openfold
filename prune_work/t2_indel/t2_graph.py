"""CUDA-graph replay of the coordinate denoiser for the T2 partial-diffusion sampler (user 10-10: speed). The sampler calls the denoiser ~250 times per batch with identical shapes; eager PyTorch spends
~20 us of CPU per launch (350-430 launches per step), which becomes the bottleneck once the kernels are fast (fp16) or the batch is small. GraphedDenoiser captures the module once per
(argument shapes/dtypes/None-pattern) key with static input buffers, then each call copies the inputs in, replays the graph and returns a clone (the sampler keeps outputs across steps).
The wrapped module runs a few eager warm-up calls first, which also fills the module's memoised sync-needing checks (all-ones seq_mask test, relpos cache) so the capture itself has no host sync.
Every tensor argument is a graph input (None patterns are part of the key). The denoiser memoises the relative-position embedding and the rotary frequencies in single-entry caches keyed on tensor
identity+version; a graph reads those cached tensors as constants, so a later capture (new static buffers -> cache miss -> entry replaced -> old tensors freed) would leave an earlier graph
pointing at freed memory (segfault on its next replay). `keepalive` holds every cache entry that existed at each capture. Those cached tensors belong to the chain whose residue_index was captured, so a call with a different residue_index /
chain_index (identity+version signature of the incoming tensors, kept alive in the record) refreshes them in place (_refresh) before replaying.
"""
import torch
import torch.nn as nn


class GraphedDenoiser(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.m = module
        self.graphs = {}
        self.keepalive = []

    def clear(self):
        self.graphs.clear()
        self.keepalive.clear()

    def forward(self, noisy_coords, noise_level, seq_mask, residue_index=None, chain_index=None, hotspot_mask=None, struct_self_cond=None, struct_crop_cond=None, sse_cond=None,
                adj_cond=None, tol=1e-6):
        args = dict(noisy_coords=noisy_coords, noise_level=noise_level, seq_mask=seq_mask, residue_index=residue_index, chain_index=chain_index, hotspot_mask=hotspot_mask,
                    struct_self_cond=struct_self_cond, struct_crop_cond=struct_crop_cond, sse_cond=sse_cond, adj_cond=adj_cond)
        key = tuple((k, None if v is None else (tuple(v.shape), v.dtype)) for k, v in args.items())
        sig = tuple(None if v is None else (id(v), v._version) for v in (residue_index, chain_index, seq_mask))
        if key in self.graphs and self.graphs[key]["sig"] != sig:
            self._refresh(key, args, sig, tol)
        if key not in self.graphs:
            static = {k: None if v is None else v.clone() for k, v in args.items()}
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side), torch.no_grad():
                for _ in range(3):
                    self.m(**static, tol=tol)
            torch.cuda.current_stream().wait_stream(side)
            allones = bool((static["seq_mask"] == 1).all())
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g), torch.no_grad():
                out = self.m(**static, tol=tol)
            consts = []   # (module, attribute, cache entry the graph reads, index of the memoised tensor in the entry)
            for mod in self.m.modules():
                for attr, pos in (("_relpos_cache", 2), ("_freqs_cache", 1)):
                    if getattr(mod, attr, None) is not None:
                        self.keepalive.append(getattr(mod, attr))
                        consts.append((mod, attr, getattr(mod, attr), pos))
            self.graphs[key] = dict(g=g, static=static, out=out, consts=consts, sig=sig, hold=(residue_index, chain_index, seq_mask), allones=allones)
        rec = self.graphs[key]
        for k, v in args.items():
            if v is not None:
                rec["static"][k].copy_(v)
        rec["g"].replay()
        return rec["out"].clone()

    def _refresh(self, key, args, sig, tol):
        """A new residue_index / chain_index / seq_mask (the next chain): the graph's memoised position tensors are constants of the graph and still hold the previous chain's values.
        Recompute them eagerly from the new static inputs and copy them in place. A seq_mask whose all-ones status differs from the capture's (a branch baked into the graph) drops the graph."""
        rec = self.graphs[key]
        static = rec["static"]
        for k, v in args.items():
            if v is not None:
                static[k].copy_(v)
        if bool((static["seq_mask"] == 1).all()) != rec["allones"]:
            del self.graphs[key]
            return
        with torch.no_grad():
            getattr(self.m, "_orig_mod", self.m)(**static, tol=tol)
        for mod, attr, entry, pos in rec["consts"]:
            entry[pos].copy_(getattr(mod, attr)[pos])
        rec["sig"] = sig
        rec["hold"] = (args["residue_index"], args["chain_index"], args["seq_mask"])
