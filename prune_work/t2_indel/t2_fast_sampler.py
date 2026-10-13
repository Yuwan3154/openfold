"""Whole-sampler-step CUDA graph for the partial-diffusion path of the T2 pipeline (RAW 215-217, T2_KERNEL_OPT_PLAN.md direction (a)); env-gated by the callers, default OFF.
protpardelle's Protpardelle.sample() runs a Python loop around the denoiser: per step it builds sigma / sigma_next with torch.full, draws a randn that is multiplied by zero (ODE, s_churn 0), masks, clones,
runs the denoiser, does the Euler update and the logging scale (~25 small kernels + host work), and only the denoiser is inside t2_graph.GraphedDenoiser's CUDA graph. For the exact path of the T2 pipeline
(cc89 all-atom, partial diffusion, ODE, step_scale 1, no guidance, no MPNN (sidechain_mode false), uniform steps, struct self-conditioning on) the step is
    xt_hat = xt * seq_mask * atom37_mask(aatype);  x0 = denoiser(xt_hat, sigma, self_cond);  xt = xt_hat + (xt_hat - x0) / sigma * (sigma_next - sigma);  self_cond = x0 / sigma_data
(the randn x 0 term is dropped: exact; self_cond None == zeros in the denoiser, so step 0 uses the same graph). FastPD.run() captures THIS WHOLE STEP once per (B, L) with static buffers and a device-side
step counter (sigma / sigma_next come from device tables), then replays it once per step: ~1 graph launch per step instead of ~30 kernel launches + Python. The denoiser's memoised relative-position / rotary
tensors are constants of the graph (see t2_graph.py): they are refreshed in place when the next chain brings another residue_index, as in GraphedDenoiser._refresh.
Initial state = the sampler's: dummy_fill_noise_coords(positions, known atoms, sigma_0) (or xt_start), masked by seq_mask. Returns the dict stage B reads: xt_traj[-1] (CPU), atom_mask, s.
"""
import torch

import protpardelle.core.models as pm
import t2_stage_b as sb
from protpardelle.data.atom import atom37_mask_from_aatype, dummy_fill_noise_coords


class FastPD:
    def __init__(self, model):
        self.model = model
        st = model.struct_model
        self.bare = st.m if hasattr(st, "m") else st            # the compiled denoiser, NOT the per-call graph wrapper (graphs do not nest)
        self.orig = getattr(self.bare, "_orig_mod", self.bare)
        self.graphs = {}
        self.keepalive = []

    def clear(self):
        self.graphs.clear()
        self.keepalive.clear()

    def _capture(self, B, L, dev, pb):
        st = dict(xt=torch.zeros(B, L, 37, 3, device=dev), xsc=torch.zeros(B, L, 37, 3, device=dev),
                  mask=pb["mask"].clone(), ridx=pb["ridx"].clone(), cidx=torch.zeros_like(pb["ridx"]), hot=torch.zeros_like(pb["ridx"]),
                  aimask=atom37_mask_from_aatype(pb["aat"], pb["mask"]).clone(), idx=torch.zeros(1, dtype=torch.long, device=dev),
                  sig=torch.ones(512, device=dev), sign=torch.ones(512, device=dev))
        sigma_data = float(self.model.sigma_data)

        def step():
            sigma = st["sig"].index_select(0, st["idx"]).expand(B).contiguous()
            sigma_next = st["sign"].index_select(0, st["idx"]).expand(B).contiguous()
            m4 = st["mask"][:, :, None, None]
            xt_hat = (st["xt"] * m4) * m4 * st["aimask"].unsqueeze(-1)
            with torch.autocast("cuda", dtype=pm._T2_AUTOCAST, enabled=pm._T2_AUTOCAST is not None):
                den = self.bare(xt_hat.clone(), sigma, st["mask"], residue_index=st["ridx"], chain_index=st["cidx"], hotspot_mask=st["hot"], struct_self_cond=st["xsc"],
                                struct_crop_cond=None, sse_cond=None, adj_cond=None)
            den = den.float()
            score = (xt_hat - den) / sigma.clamp(min=1e-6)[:, None, None, None]
            st["xt"].copy_(xt_hat + score * 1.0 * (sigma_next - sigma)[:, None, None, None])
            st["xsc"].copy_(den.detach() / sigma_data)
            st["idx"].add_(1)

        saved = {k: v.clone() for k, v in st.items() if k in ("xt", "xsc", "idx")}   # state before the warm-up steps (enqueued BEFORE the side stream waits)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), torch.no_grad():
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(side)
        for k, v in saved.items():
            st[k].copy_(v)
        allones = bool((st["mask"] == 1).all())
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g), torch.no_grad():
            step()
        for k, v in saved.items():
            st[k].copy_(v)   # the warm-up steps changed the state; the capture itself executes nothing
        consts = []
        for mod in self.orig.modules():
            for attr, pos in (("_relpos_cache", 2), ("_freqs_cache", 1)):
                if getattr(mod, attr, None) is not None:
                    self.keepalive.append(getattr(mod, attr))
                    consts.append((mod, attr, getattr(mod, attr), pos))
        return dict(g=g, st=st, consts=consts, allones=allones)

    def _refresh(self, rec, pb):
        """new chain: copy its static inputs and recompute the memoised position tensors the graph holds as constants (eager, uncompiled module), in place"""
        st = rec["st"]
        st["mask"].copy_(pb["mask"]); st["ridx"].copy_(pb["ridx"]); st["aimask"].copy_(atom37_mask_from_aatype(pb["aat"], pb["mask"]))
        if bool((st["mask"] == 1).all()) != rec["allones"]:
            return False
        B = st["xt"].shape[0]
        with torch.no_grad(), torch.autocast("cuda", dtype=pm._T2_AUTOCAST, enabled=pm._T2_AUTOCAST is not None):
            self.orig(st["xt"], torch.ones(B, device=st["xt"].device), st["mask"], residue_index=st["ridx"], chain_index=st["cidx"], hotspot_mask=st["hot"], struct_self_cond=st["xsc"],
                      struct_crop_cond=None, sse_cond=None, adj_cond=None)
        for mod, attr, entry, pos in rec["consts"]:
            entry[pos].copy_(getattr(mod, attr)[pos])
        return True

    @torch.no_grad()
    def run(self, pb, rewind, num_steps=None, xt_start=None):
        B, L = pb["aat"].shape
        dev = pb["pos"].device
        kw = sb.sampling_kwargs([rewind] * B, num_steps)
        assert kw["step_scale"] == 1.0 and kw["s_churn"] == 0 and not kw["sidechain_mode"] and kw["uniform_steps"], "FastPD covers the T2 path only (ODE, step scale 1, no MPNN, uniform steps)"
        ns, sched = int(kw["num_steps"]), kw["noise_schedule"]
        ts = torch.linspace(1, 0, ns + 1)
        assert rewind < ns, "rewind == num_steps zeroes the last sigma_next in the sampler (not covered)"
        pd_step = ns - rewind
        sig, sign, s = [], [], float(sched(ts[pd_step]))
        sigma0 = s
        for t in ts[pd_step:][1:]:
            sn = float(sched(t))
            sig.append(s); sign.append(sn); s = sn
        assert min(sig) > 0 and len(sig) <= 512
        mask = pb["mask"]
        if xt_start is None:
            am = atom37_mask_from_aatype(pb["aat"], mask) * pb["known"]
            xt = dummy_fill_noise_coords(pb["pos"], am, noise_level=torch.full((B,), sigma0, device=dev, dtype=pb["pos"].dtype), dummy_fill_mode=self.model.config.data.dummy_fill_mode)
        else:
            xt = xt_start
        xt = xt * mask[:, :, None, None]
        key = (B, L)
        rec = self.graphs.get(key)
        if rec is not None and not self._refresh(rec, pb):
            del self.graphs[key]
            rec = None
        if rec is None:
            rec = self.graphs[key] = self._capture(B, L, dev, pb)
        st = rec["st"]
        st["xt"].copy_(xt); st["xsc"].zero_(); st["idx"].zero_()
        st["sig"][:len(sig)] = torch.tensor(sig, device=dev); st["sign"][:len(sign)] = torch.tensor(sign, device=dev)
        for _ in range(len(sig)):
            rec["g"].replay()
        scale = float(self.model.sigma_data) / (sign[-1] ** 2 + float(self.model.sigma_data) ** 2) ** 0.5
        out = (st["xt"] * scale).cpu()
        return dict(xt_traj=[out], atom_mask=atom37_mask_from_aatype(pb["aat"], mask), s=pb["aat"])
