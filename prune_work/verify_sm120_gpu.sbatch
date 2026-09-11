#!/bin/bash
#SBATCH -p pi_so3
#SBATCH -J of3_sm120_verify
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2-00:00:00
#SBATCH -o /orcd/pool/006/chenxiou/logs/of3_sm120_verify-%j.out
#SBATCH -e /orcd/pool/006/chenxiou/logs/of3_sm120_verify-%j.out
#
# The on-GPU half of the sm_120 rebuild. The build job proves the BINARIES carry sm_120 cubins;
# only this proves kernels RUN. It exits in a couple of minutes -- the 2-day limit is the partition
# max per the always-request-max-wallclock rule, not a hold on the GPU.
#
# ENV is a parameter so the same script can re-verify the H200 path later:
#   sbatch --gres=gpu:h200:1 prune_work/verify_sm120_gpu.sbatch
set -eo pipefail

S=/orcd/scratch/orcd/011/chenxiou
ENV=${OF3_ENV:-$S/envs/of3_env_sm120}
REPO=$S/openfold-esmfold2-recycling

echo "START $(date -Is) host=$(hostname) job=$SLURM_JOB_ID"
echo "ENV=$ENV"
nvidia-smi --query-gpu=index,name,compute_cap,memory.total --format=csv,noheader
module load cuda/12.9.1

cd "$REPO"
PYTHONPATH="$REPO/openfold" "$ENV/bin/python" - <<'PYGPU' || { echo "VERIFY FAILED"; exit 1; }
import torch
from openfold.utils.kernel.attention_core import attention_core

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("arch_list", torch.cuda.get_arch_list())
name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
print(f"device: {name}  capability sm_{cap[0]}{cap[1]}")

# 1. plain kernels
a = torch.randn(1024, 1024, device="cuda", requires_grad=True)
(a @ a.T).sum().backward()
torch.cuda.synchronize()
print("plain matmul fwd+bwd OK")

# 2. cuDNN and cuBLAS, at the precision Run D actually trains in
b = torch.randn(8, 16, 64, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
w = torch.randn(16, 16, 3, 3, device="cuda", dtype=torch.bfloat16, requires_grad=True)
torch.nn.functional.conv2d(b, w, padding=1).float().sum().backward()
c = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
_ = c @ c.T
torch.cuda.synchronize()
print("conv2d (cuDNN) + bf16 matmul (cuBLAS) fwd+bwd OK")

# 3. THE openfold extension -- forward_ and backward_, checked for CORRECTNESS, not just liveness.
#    A kernel that runs but returns garbage is the failure mode a crash-only test cannot see.
#    Tolerances come from torch.testing's own per-dtype defaults; nothing is hand-picked here.
torch.manual_seed(0)
B, H, Q, K, D = 2, 4, 37, 53, 32
for dtype in (torch.float32, torch.bfloat16):
    q = torch.randn(B, H, Q, D, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn(B, H, K, D, device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn(B, H, K, D, device="cuda", dtype=dtype, requires_grad=True)
    qr, kr, vr = (t.detach().clone().requires_grad_(True) for t in (q, k, v))

    out = attention_core(q, k, v)
    ref = torch.matmul(torch.softmax(torch.matmul(qr, kr.transpose(-1, -2)), dim=-1), vr)
    torch.testing.assert_close(out, ref)

    out.sum().backward()
    ref.sum().backward()
    for nm, x, y in (("q", q, qr), ("k", k, kr), ("v", v, vr)):
        torch.testing.assert_close(x.grad, y.grad, msg=lambda m, nm=nm: f"grad_{nm} mismatch\n{m}")

    fwd_err = (out.float() - ref.float()).abs().max().item()
    print(f"  {str(dtype):>18}  matches autograd reference  (fwd_max_err={fwd_err:.3e})")

print("attn_core_inplace_cuda forward_ + backward_ NUMERICALLY CORRECT on", name)
print("VERIFY OK")
PYGPU

echo "DONE $(date -Is)"
