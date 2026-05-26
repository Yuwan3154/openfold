# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partialmethod, partial
import math
from typing import Optional, List

import torch
import torch.nn as nn

from openfold.model.primitives import Linear, LayerNorm, Attention
from openfold.utils.chunk_utils import chunk_layer
from openfold.utils.tensor_utils import (
    permute_final_dims,
    flatten_final_dims,
)


class TriangleAttention(nn.Module):
    def __init__(
        self, c_in, c_hidden, no_heads, starting=True, inf=1e9
    ):
        """
        Args:
            c_in:
                Input channel dimension
            c_hidden:
                Overall hidden channel dimension (not per-head)
            no_heads:
                Number of attention heads
        """
        super(TriangleAttention, self).__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.starting = starting
        self.inf = inf

        self.layer_norm = LayerNorm(self.c_in)

        self.linear = Linear(c_in, self.no_heads, bias=False, init="normal")

        self.mha = Attention(
            self.c_in, self.c_in, self.c_in, self.c_hidden, self.no_heads
        )

    @torch.jit.ignore
    def _chunk(self,
        x: torch.Tensor,
        biases: List[torch.Tensor],
        chunk_size: int,
        use_memory_efficient_kernel: bool = False,
        use_deepspeed_evo_attention: bool = False,
        use_cuequivariance_attention: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        use_torch_sdpa: bool = False,
        use_torch_vanilla: bool = False,
        use_torch_cueq: bool = False,
    ) -> torch.Tensor:
        "triangle! triangle!"
        mha_inputs = {
            "q_x": x,
            "kv_x": x,
            "biases": biases,
        }

        return chunk_layer(
            partial(
                self.mha,
                use_memory_efficient_kernel=use_memory_efficient_kernel,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_cuequivariance_attention=use_cuequivariance_attention,
                use_lma=use_lma,
                use_torch_sdpa=use_torch_sdpa,
                use_torch_vanilla=use_torch_vanilla,
                use_torch_cueq=use_torch_cueq,
            ),
            mha_inputs,
            chunk_size=chunk_size,
            no_batch_dims=len(x.shape[:-2]),
            _out=x if inplace_safe else None,
        )

    def forward(self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
        use_memory_efficient_kernel: bool = False,
        use_deepspeed_evo_attention: bool = False,
        use_cuequivariance_attention: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        use_torch_sdpa: bool = False,
        use_torch_vanilla: bool = False,
        use_torch_cueq: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x:
                [*, I, J, C_in] input tensor (e.g. the pair representation)
        Returns:
            [*, I, J, C_in] output tensor
        """
        if mask is None:
            # [*, I, J]
            mask = x.new_ones(
                x.shape[:-1],
            )

        if(not self.starting):
            x = x.transpose(-2, -3)
            mask = mask.transpose(-1, -2)

        # [*, I, J, C_in]
        x = self.layer_norm(x)

        # [*, I, 1, 1, J]
        # cuEq path (use_torch_cueq) wants a 0/1 mask, same as the legacy
        # use_cuequivariance_attention. SDPA/vanilla use an additive bias.
        if (use_cuequivariance_attention or use_torch_cueq) and not (use_torch_sdpa or use_torch_vanilla):
            mask_bias = mask[..., :, None, None, :]
        else:
            mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]

        # [*, H, I, J]
        triangle_bias = permute_final_dims(self.linear(x), (2, 0, 1))

        # [*, 1, H, I, J]
        triangle_bias = triangle_bias.unsqueeze(-4)

        biases = [mask_bias, triangle_bias]

        # In compile-friendly mode, disable kernels that wrap incompatible
        # CUDA extensions (deepspeed / cuequivariance / memory_efficient).
        # When use_torch_cueq is on, we DO use the cueq triangle attention
        # kernel — but via the compile-friendly branch in Attention.forward
        # (which skips the shape-dependent cueq_would_fall_back check).
        if use_torch_sdpa or use_torch_vanilla:
            use_memory_efficient_kernel = False
            use_deepspeed_evo_attention = False
            use_cuequivariance_attention = False
        elif use_torch_cueq:
            use_memory_efficient_kernel = False
            use_deepspeed_evo_attention = False
            # Don't take the legacy cuequivariance dispatch; the new
            # compile-friendly path is triggered by use_torch_cueq.
            use_cuequivariance_attention = False

        if chunk_size is not None:
            x = self._chunk(
                x,
                biases,
                chunk_size,
                use_memory_efficient_kernel=use_memory_efficient_kernel,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_cuequivariance_attention=use_cuequivariance_attention,
                use_lma=use_lma,
                inplace_safe=inplace_safe,
                use_torch_sdpa=use_torch_sdpa,
                use_torch_vanilla=use_torch_vanilla,
                use_torch_cueq=use_torch_cueq,
            )
        else:
            x = self.mha(
                q_x=x,
                kv_x=x,
                biases=biases,
                use_memory_efficient_kernel=use_memory_efficient_kernel,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_cuequivariance_attention=use_cuequivariance_attention,
                use_lma=use_lma,
                use_torch_sdpa=use_torch_sdpa,
                use_torch_vanilla=use_torch_vanilla,
                use_torch_cueq=use_torch_cueq,
            )

        if(not self.starting):
            x = x.transpose(-2, -3)

        return x


# Implements Algorithm 13
TriangleAttentionStartingNode = TriangleAttention


class TriangleAttentionEndingNode(TriangleAttention):
    """
    Implements Algorithm 14.
    """
    __init__ = partialmethod(TriangleAttention.__init__, starting=False)
