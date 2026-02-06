# Copyright 2021 AlQuraishi Laboratory
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
import functools
import importlib
from typing import Any, Tuple, List, Callable, Optional, Dict

deepspeed_is_installed = importlib.util.find_spec("deepspeed") is not None
if(deepspeed_is_installed):
    import deepspeed

import torch
import torch.utils.checkpoint


BLOCK_ARG = Any
BLOCK_ARGS = List[BLOCK_ARG]


def _unwrap_block(block: Callable) -> Tuple[Callable, Tuple[Any, ...], Dict[str, Any]]:
    if isinstance(block, functools.partial):
        return block.func, block.args, block.keywords or {}
    return block, (), {}


def _is_replacement_only_block(block: Callable) -> bool:
    func, _, _ = _unwrap_block(block)
    return (
        hasattr(func, "replacement_block")
        and hasattr(func, "original_block")
        and getattr(func, "ckpt_replacement_only", False)
        and getattr(func, "use_straight_through", False)
    )


def _run_replacement_only_checkpoint(
    block: Callable,
    args: BLOCK_ARGS,
    checkpoint: Callable,
) -> BLOCK_ARGS:
    func, bound_args, bound_kwargs = _unwrap_block(block)
    full_args = (*bound_args, *args)
    with torch.no_grad():
        orig_m, orig_z = func.original_block(*full_args, **bound_kwargs)

    def repl_fn(*inner_args):
        return func.replacement_block(*inner_args, **bound_kwargs)

    rep_m, rep_z = checkpoint(repl_fn, *full_args, use_reentrant=False)
    m_out = rep_m + orig_m.detach() - rep_m.detach()
    z_out = rep_z + orig_z.detach() - rep_z.detach()
    return (m_out, z_out)


def get_checkpoint_fn():
    deepspeed_is_configured = (
        deepspeed_is_installed and
        deepspeed.checkpointing.is_configured()
    )
    if(deepspeed_is_configured):
        checkpoint = deepspeed.checkpointing.checkpoint
    else:
        checkpoint = torch.utils.checkpoint.checkpoint

    return checkpoint


@torch.jit.ignore
def checkpoint_blocks(
    blocks: List[Callable],
    args: BLOCK_ARGS,
    blocks_per_ckpt: Optional[int],
    outputs: Optional[Dict[str, Any]] = None,
    cycle_no: Optional[int] = None,
) -> BLOCK_ARGS:
    """
    Chunk a list of blocks and run each chunk with activation
    checkpointing. We define a "block" as a callable whose only inputs are
    the outputs of the previous block.

    Implements Subsection 1.11.8

    Args:
        blocks:
            List of blocks
        args:
            Tuple of arguments for the first block.
        blocks_per_ckpt:
            Size of each chunk. A higher value corresponds to fewer 
            checkpoints, and trades memory for speed. If None, no checkpointing 
            is performed.
    Returns:
        The output of the final block
    """
    def wrap(a):
        return (a,) if type(a) is not tuple else a

    def exec(b, a):
        for i, block in enumerate(b):
            a = wrap(block(*a))
            if outputs is not None:
                outputs[f"recycle_{cycle_no}_block_{i}"] = a
        return a

    def chunker(s, e):
        def exec_sliced(*a):
            return exec(blocks[s:e], a)

        return exec_sliced

    # Avoids mishaps when the blocks take just one argument
    args = wrap(args)

    if blocks_per_ckpt is None or not torch.is_grad_enabled():
        return exec(blocks, args)
    elif blocks_per_ckpt < 1 or blocks_per_ckpt > len(blocks):
        raise ValueError("blocks_per_ckpt must be between 1 and len(blocks)")

    checkpoint = get_checkpoint_fn()
    if any(_is_replacement_only_block(block) for block in blocks):
        for i, block in enumerate(blocks):
            if _is_replacement_only_block(block):
                args = wrap(_run_replacement_only_checkpoint(block, args, checkpoint))
            else:
                args = wrap(checkpoint(block, *args, use_reentrant=False))
            if outputs is not None:
                outputs[f"recycle_{cycle_no}_block_{i}"] = args
        return args

    if any(_has_triangle_st(block) for block in blocks):
        for i, block in enumerate(blocks):
            if _has_triangle_st(block):
                args = wrap(_run_triangle_st_checkpoint(block, args, checkpoint))
            else:
                args = wrap(checkpoint(block, *args, use_reentrant=False))
            if outputs is not None:
                outputs[f"recycle_{cycle_no}_block_{i}"] = args
        return args

    for s in range(0, len(blocks), blocks_per_ckpt):
        e = s + blocks_per_ckpt
        args = checkpoint(chunker(s, e), *args, use_reentrant=False)
        args = wrap(args)

    return args


def _has_triangle_st(block: Callable) -> bool:
    func, _, _ = _unwrap_block(block)
    return getattr(func, '_has_triangle_st', False)


def _run_triangle_st_checkpoint(
    block: Callable,
    args: BLOCK_ARGS,
    checkpoint: Callable,
) -> BLOCK_ARGS:
    """
    Handle a block with triangle-only ST via 4-phase execution:
      Phase 1: MSA + OPM (checkpointed)
      Phase 2: all 4 triangle ops in no_grad (NOT checkpointed, runs once)
      Phase 3: single replacement call (checkpointed) + ST combine
      Phase 4: pair_transition (checkpointed)
    """
    func, bound_args, bound_kwargs = _unwrap_block(block)
    m, z = args
    kw = bound_kwargs

    pair_mask = kw.get('pair_mask')
    msa_mask = kw.get('msa_mask')
    chunk_size = kw.get('chunk_size')
    use_deepspeed = kw.get('use_deepspeed_evo_attention', False)
    use_lma = kw.get('use_lma', False)
    use_flash = kw.get('use_flash', False)
    inplace_safe = kw.get('inplace_safe', False)
    _mask_trans = kw.get('_mask_trans', True)
    _attn_chunk_size = kw.get('_attn_chunk_size')

    # Phase 1: MSA + OPM (checkpointed)
    def msa_opm_fn(m, z):
        return func.forward_msa_opm(
            m, z, msa_mask=msa_mask, pair_mask=pair_mask,
            chunk_size=chunk_size, use_deepspeed_evo_attention=use_deepspeed,
            use_lma=use_lma, use_flash=use_flash, inplace_safe=inplace_safe,
            _mask_trans=_mask_trans, _attn_chunk_size=_attn_chunk_size,
        )
    m, z_after_opm = checkpoint(msa_opm_fn, m, z, use_reentrant=False)

    # Phase 2: all 4 triangle ops (no_grad, NOT checkpointed — runs once)
    with torch.no_grad():
        z_orig = func.pair_stack.forward_triangle_ops(
            z_after_opm,
            pair_mask=pair_mask, chunk_size=chunk_size,
            use_deepspeed_evo_attention=use_deepspeed, use_lma=use_lma,
            inplace_safe=inplace_safe, _attn_chunk_size=_attn_chunk_size,
        )

    # Phase 3: single replacement (checkpointed) + ST combine
    replacement = func._triangle_replacement
    def replacement_fn(z_in):
        return replacement(z_in, pair_mask=pair_mask)
    repl_out = checkpoint(replacement_fn, z_after_opm, use_reentrant=False)

    # ST combine: forward value from original, gradient from replacement
    z_combined = repl_out + (z_orig - repl_out).detach()
    del z_orig

    # Phase 4: pair_transition (checkpointed)
    def pair_transition_fn(z):
        return func.pair_stack.forward_pair_transition(
            z, pair_mask=pair_mask, _mask_trans=_mask_trans,
            chunk_size=chunk_size, inplace_safe=inplace_safe,
        )
    z_final = checkpoint(pair_transition_fn, z_combined, use_reentrant=False)

    return (m, z_final)
