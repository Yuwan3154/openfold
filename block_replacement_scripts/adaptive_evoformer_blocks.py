#!/usr/bin/env python3
"""
Adaptive Evoformer blocks that combine original and replacement blocks with learned weights.

This module provides wrapper classes that:
1. Run both original Evoformer blocks and trained replacement blocks
2. Predict adaptive weights based on MSA representation
3. Output weighted combination: w * original_output + (1-w) * replacement_output
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Any
from pathlib import Path

# Import from parent directory
import sys
sys.path.append(str(Path(__file__).parent.parent))

from openfold.model.primitives import Linear
from .custom_evoformer_replacement import SimpleEvoformerReplacement


class AdaptiveWeightPredictor(nn.Module):
    """Predicts weights for adaptive Evoformer-replacement mixing"""
    
    def __init__(self, c_m: int):
        super().__init__()
        self.linear = nn.Linear(c_m, 1)
        
    def forward(self, msa_representation):
        """
        Args:
            msa_representation: MSA representation [batch, n_seq, n_res, c_m]
        Returns:
            weight: Scalar weight for this block [batch, 1]
        """
        # Extract first sequence: m[..., 0, :, :] -> [batch, n_res, c_m]
        single_rep = msa_representation[..., 0, :, :]
        
        # Mean pool across residues: [batch, n_res, c_m] -> [batch, c_m]
        pooled = torch.mean(single_rep, dim=-2)
        
        # Apply linear transformation: [batch, c_m] -> [batch, 1]
        weight = self.linear(pooled)
        
        # Apply sigmoid to get weight in [0, 1]
        weight = torch.sigmoid(weight)
        
        return weight


class AdaptiveEvoformerBlock(nn.Module):
    """
    Wrapper for Evoformer block that combines original and replacement outputs.
    
    This wrapper:
    1. Runs the original Evoformer block
    2. Runs the pre-trained replacement block
    3. Predicts adaptive weight from MSA representation
    4. Returns weighted combination of outputs
    """
    
    def __init__(
        self,
        original_block: nn.Module,
        replacement_block: nn.Module,
        weight_predictor: AdaptiveWeightPredictor,
        block_idx: int,
    ):
        super().__init__()
        self.original_block = original_block
        self.replacement_block = replacement_block
        self.weight_predictor = weight_predictor
        self.block_idx = block_idx
        
        # Freeze original block parameters
        for param in self.original_block.parameters():
            param.requires_grad = False
            
        # Keep replacement block parameters trainable (we want to fine-tune them)
        for param in self.replacement_block.parameters():
            param.requires_grad = True
    
    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: Optional[int] = None,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        use_flash: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
        _attn_chunk_size: Optional[int] = None,
        _offload_inference: bool = False,
        _offloadable_inputs: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass combining original and replacement blocks.
        
        Args:
            m: MSA representation [*, N_seq, N_res, C_m]
            z: Pair representation [*, N_res, N_res, C_z]
            msa_mask: MSA mask [*, N_seq, N_res]
            pair_mask: Pair mask [*, N_res, N_res]
            Other args: Passed through to original block
            
        Returns:
            m_out: Weighted MSA output
            z_out: Weighted pair output
        """
        # Store input for replacement block (it expects different signature)
        m_input = m.clone() if self.training else m
        z_input = z.clone() if self.training else z
        
        # 1. Run original Evoformer block
        m_orig, z_orig = self.original_block(
            m=m,
            z=z,
            msa_mask=msa_mask,
            pair_mask=pair_mask,
            chunk_size=chunk_size,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            use_lma=use_lma,
            use_flash=use_flash,
            inplace_safe=inplace_safe,
            _mask_trans=_mask_trans,
            _attn_chunk_size=_attn_chunk_size,
            _offload_inference=_offload_inference,
            _offloadable_inputs=_offloadable_inputs,
        )
        
        # 2. Run replacement block (with simpler signature)
        with torch.cuda.amp.autocast(enabled=False):
            # Cast to float32 for replacement block if needed
            m_replace_in = m_input.float()
            z_replace_in = z_input.float()
            msa_mask_replace = msa_mask.float() if msa_mask is not None else None
            pair_mask_replace = pair_mask.float() if pair_mask is not None else None
            
            m_replace, z_replace = self.replacement_block(
                m_replace_in,
                z_replace_in,
                msa_mask_replace,
                pair_mask_replace
            )
            
            # Cast back to original dtype
            m_replace = m_replace.to(m_orig.dtype)
            z_replace = z_replace.to(z_orig.dtype)
        
        # 3. Predict adaptive weight from MSA representation
        weight = self.weight_predictor(m_input)  # [batch, 1]
        
        # Reshape weight for broadcasting
        # m: [batch, N_seq, N_res, C_m] -> weight: [batch, 1, 1, 1]
        # z: [batch, N_res, N_res, C_z] -> weight: [batch, 1, 1, 1]
        weight_m = weight.view(-1, 1, 1, 1)
        weight_z = weight.view(-1, 1, 1, 1)
        
        # 4. Compute weighted combination
        # Note: we use (1-w) for replacement to encourage using original by default
        m_out = weight_m * m_orig + (1 - weight_m) * m_replace
        z_out = weight_z * z_orig + (1 - weight_z) * z_replace
        
        return m_out, z_out


def load_pretrained_replacement_block(
    checkpoint_path: Path,
    c_m: int,
    c_z: int,
    linear_type: str = "full",
    hidden_dim: Optional[int] = None,
) -> SimpleEvoformerReplacement:
    """
    Load a pre-trained replacement block from checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        c_m: MSA channel dimension
        c_z: Pair channel dimension
        linear_type: Type of linear layer ("full", "diagonal", "affine")
        hidden_dim: Hidden dimension (if None, uses default)
        
    Returns:
        Loaded replacement block model
    """
    # Create model
    if hidden_dim is None:
        if linear_type == "full":
            hidden_dim = max(c_m, c_z)
        else:
            hidden_dim = None
    
    model = SimpleEvoformerReplacement(
        c_m=c_m,
        c_z=c_z,
        c_hidden=hidden_dim,
        linear_type=linear_type
    )
    
    # Load checkpoint
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        # Remove 'replacement_block.' prefix if present
        state_dict = {k.replace('replacement_block.', ''): v 
                     for k, v in state_dict.items() 
                     if k.startswith('replacement_block.') or 'replacement_block' not in k}
        
        model.load_state_dict(state_dict)
        print(f"Loaded replacement block from {checkpoint_path}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    return model


def replace_evoformer_blocks_with_adaptive(
    model: nn.Module,
    trained_models_dir: Path,
    linear_type: str = "full",
    c_m: int = 256,
    c_z: int = 128,
    hidden_dim: Optional[int] = None,
    replace_blocks: Optional[list] = None,
) -> Tuple[nn.Module, Dict[int, AdaptiveWeightPredictor]]:
    """
    Replace specified Evoformer blocks with adaptive versions.
    
    Args:
        model: The OpenFold model
        trained_models_dir: Directory containing trained replacement blocks
        linear_type: Type of linear layer used in replacement blocks
        c_m: MSA channel dimension
        c_z: Pair channel dimension
        hidden_dim: Hidden dimension for replacement blocks
        replace_blocks: List of block indices to replace (if None, replaces all available)
        
    Returns:
        model: Modified model with adaptive blocks
        weight_predictors: Dictionary mapping block index to weight predictor
    """
    # Find available trained blocks
    available_blocks = []
    for block_dir in trained_models_dir.glob("block_*"):
        if block_dir.is_dir():
            checkpoint_path = block_dir / linear_type / "best_model.ckpt"
            if checkpoint_path.exists():
                block_idx = int(block_dir.name.split("_")[1])
                available_blocks.append(block_idx)
    
    available_blocks.sort()
    print(f"Found {len(available_blocks)} trained replacement blocks: {available_blocks}")
    
    # Determine which blocks to replace
    if replace_blocks is None:
        replace_blocks = available_blocks
    else:
        # Filter to only available blocks
        replace_blocks = [b for b in replace_blocks if b in available_blocks]
    
    print(f"Replacing {len(replace_blocks)} blocks: {replace_blocks}")
    
    # Create weight predictors for each block
    weight_predictors = {}
    
    # Replace blocks in the evoformer stack
    evoformer_stack = model.evoformer_stack
    
    for block_idx in replace_blocks:
        if block_idx >= len(evoformer_stack.blocks):
            print(f"Warning: Block {block_idx} out of range, skipping")
            continue
            
        # Load pre-trained replacement block
        checkpoint_path = trained_models_dir / f"block_{block_idx:02d}" / linear_type / "best_model.ckpt"
        replacement_block = load_pretrained_replacement_block(
            checkpoint_path=checkpoint_path,
            c_m=c_m,
            c_z=c_z,
            linear_type=linear_type,
            hidden_dim=hidden_dim,
        )
        
        # Create weight predictor for this block
        weight_predictor = AdaptiveWeightPredictor(c_m=c_m)
        weight_predictors[block_idx] = weight_predictor
        
        # Get original block
        original_block = evoformer_stack.blocks[block_idx]
        
        # Create adaptive block
        adaptive_block = AdaptiveEvoformerBlock(
            original_block=original_block,
            replacement_block=replacement_block,
            weight_predictor=weight_predictor,
            block_idx=block_idx,
        )
        
        # Replace in model
        evoformer_stack.blocks[block_idx] = adaptive_block
        
    print(f"Successfully replaced {len(replace_blocks)} blocks with adaptive versions")
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    return model, weight_predictors
