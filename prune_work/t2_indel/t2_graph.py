"""CUDA-graph replay of the coordinate denoiser for the T2 partial-diffusion sampler (user 10-10: speed). The sampler calls the denoiser ~250 times per batch with identical shapes; eager PyTorch spends
~20 us of CPU per launch (350-430 launches per step), which becomes the bottleneck once the kernels are fast (fp16) or the batch is small. GraphedDenoiser captures the module once per
(argument shapes/dtypes/None-pattern) key with static input buffers, then each call copies the inputs in, replays the graph and returns a clone (the sampler keeps outputs across steps).
The wrapped module runs a few eager warm-up calls first, which also fills the module's memoised sync-needing checks (all-ones seq_mask test, relpos cache) so the capture itself has no host sync.
Only the arguments the partial-diffusion sampler uses are supported (hotspot / crop / sse / adjacency conditioning must be None).
"""
import torch
import torch.nn as nn


class GraphedDenoiser(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.m = module
        self.graphs = {}

    def forward(self, noisy_coords, noise_level, seq_mask, residue_index=None, chain_index=None, hotspot_mask=None, struct_self_cond=None, struct_crop_cond=None, sse_cond=None,
                adj_cond=None, tol=1e-6):
        assert hotspot_mask is None and struct_crop_cond is None and sse_cond is None and adj_cond is None
        args = dict(noisy_coords=noisy_coords, noise_level=noise_level, seq_mask=seq_mask, residue_index=residue_index, chain_index=chain_index, struct_self_cond=struct_self_cond)
        key = tuple((k, None if v is None else (tuple(v.shape), v.dtype)) for k, v in args.items())
        if key not in self.graphs:
            static = {k: None if v is None else v.clone() for k, v in args.items()}
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side), torch.no_grad():
                for _ in range(3):
                    self.m(**static, tol=tol)
            torch.cuda.current_stream().wait_stream(side)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g), torch.no_grad():
                out = self.m(**static, tol=tol)
            self.graphs[key] = (g, static, out)
        g, static, out = self.graphs[key]
        for k, v in args.items():
            if v is not None:
                static[k].copy_(v)
        g.replay()
        return out.clone()
