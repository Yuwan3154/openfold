"""FP8 (e4m3) linear layers for the cc89 denoiser (T2_KERNEL_OPT_PLAN.md direction (d); OPTION ONLY, parity risk, tested last). convert(module, min_features=64) replaces every nn.Linear whose in/out features are multiples of 16 by FP8Linear:
the weight is quantised once (per-tensor scale), the activation per call (dynamic per-tensor amax scale), the product is torch._scaled_mm (fp8 tensor cores of L40S / H200, fp16 output, fp32 accumulate); biases stay fp16.
Usage in a bench: from t2_fp8 import convert; convert(model.struct_model)  (before torch.compile). Parity must be judged with the panel gates (TM / pass shares vs the seed spread), not per atom: fp8 e4m3 has 3 mantissa bits.
"""
import torch
import torch.nn as nn

E4M3_MAX = 448.0


class FP8Linear(nn.Module):
    def __init__(self, lin: nn.Linear):
        super().__init__()
        w = lin.weight.detach().float()
        self.w_scale = (w.abs().max() / E4M3_MAX).clamp(min=1e-12).reshape(())
        self.register_buffer("w8", (w / self.w_scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).contiguous())
        self.register_buffer("w_scale_t", self.w_scale.reshape(()).float())
        self.bias = None if lin.bias is None else nn.Parameter(lin.bias.detach().half(), requires_grad=False)
        self.in_features, self.out_features = lin.in_features, lin.out_features

    def forward(self, x):
        shp = x.shape
        x2 = x.reshape(-1, self.in_features).to(torch.float16)
        a_scale = (x2.abs().amax().float() / E4M3_MAX).clamp(min=1e-12)
        x8 = (x2.float() / a_scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
        n = x8.shape[0]
        pad = (-n) % 16
        if pad:
            x8 = torch.cat([x8, x8.new_zeros(pad, self.in_features)])
        out = torch._scaled_mm(x8, self.w8.t(), scale_a=a_scale, scale_b=self.w_scale_t, bias=self.bias, out_dtype=torch.float16)
        if isinstance(out, tuple):
            out = out[0]
        return out[:n].reshape(*shp[:-1], self.out_features)


def convert(module, min_features=64):
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and child.in_features % 16 == 0 and child.out_features % 16 == 0 and min(child.in_features, child.out_features) >= min_features:
            setattr(module, name, FP8Linear(child))
            n += 1
        else:
            n += convert(child, min_features)
    return n
