"""
Custom Evoformer Block Replacement for Training Experiments
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional, Sequence


class SimpleEvoformerReplacement(nn.Module):
    """
    Simple replacement for EvoformerBlock with LayerNorm -> Linear -> ReLU -> Linear architecture
    Maintains the same input/output signature as the original EvoformerBlock
    """
    
    def __init__(self, c_m: int, c_z: int, hidden_dim: Optional[int] = None):
        """
        Args:
            c_m: MSA representation dimension
            c_z: Pair representation dimension  
            hidden_dim: Hidden dimension for the linear layers (defaults to max(c_m, c_z))
        """
        super(SimpleEvoformerReplacement, self).__init__()
        
        self.c_m = c_m
        self.c_z = c_z
        
        if hidden_dim is None:
            hidden_dim = max(c_m, c_z)
        
        # MSA processing layers
        self.msa_layer_norm = nn.LayerNorm(c_m)
        self.msa_linear1 = nn.Linear(c_m, hidden_dim)
        self.msa_relu = nn.ReLU()
        self.msa_linear2 = nn.Linear(hidden_dim, c_m)
        
        # Pair processing layers
        self.pair_layer_norm = nn.LayerNorm(c_z)
        self.pair_linear1 = nn.Linear(c_z, hidden_dim)
        self.pair_relu = nn.ReLU()
        self.pair_linear2 = nn.Linear(hidden_dim, c_z)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self,
        m: Optional[torch.Tensor],
        z: Optional[torch.Tensor],
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
        _offloadable_inputs: Optional[Sequence[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass that mimics EvoformerBlock signature but uses simple linear transformations
        
        Args:
            m: MSA representation [batch, n_seq, n_res, c_m]
            z: Pair representation [batch, n_res, n_res, c_z]
            msa_mask: MSA mask
            pair_mask: Pair mask
            (other args are ignored but kept for compatibility)
            
        Returns:
            Tuple of (updated_m, updated_z)
        """
        
        if m is not None:
            # Apply residual connection: m = m + f(LayerNorm(m))
            m_normalized = self.msa_layer_norm(m)
            m_transformed = self.msa_linear2(self.msa_relu(self.msa_linear1(m_normalized)))
            
            # Apply mask if provided
            if msa_mask is not None and _mask_trans:
                # Expand mask to match tensor dimensions
                mask_expanded = msa_mask.unsqueeze(-1)  # [batch, n_seq, n_res, 1]
                m_transformed = m_transformed * mask_expanded
            
            m = m + m_transformed
        
        if z is not None:
            # Apply residual connection: z = z + f(LayerNorm(z))
            z_normalized = self.pair_layer_norm(z)
            z_transformed = self.pair_linear2(self.pair_relu(self.pair_linear1(z_normalized)))
            
            # Apply mask if provided
            if pair_mask is not None and _mask_trans:
                # Expand mask to match tensor dimensions
                mask_expanded = pair_mask.unsqueeze(-1)  # [batch, n_res, n_res, 1]
                z_transformed = z_transformed * mask_expanded
            
            z = z + z_transformed
        
        return m, z


def replace_evoformer_block(model, block_index: int, c_m: int, c_z: int, hidden_dim: Optional[int] = None):
    """
    Replace a specific EvoformerBlock with SimpleEvoformerReplacement
    
    Args:
        model: OpenFold model
        block_index: Index of the block to replace (not first or last)
        c_m: MSA representation dimension
        c_z: Pair representation dimension
        hidden_dim: Hidden dimension for replacement block
    
    Returns:
        The model with the replaced block
    """
    
    total_blocks = len(model.evoformer.blocks)
    
    if block_index <= 0 or block_index >= total_blocks - 1:
        raise ValueError(f"Block index {block_index} must be between 1 and {total_blocks-2} (not first or last)")
    
    # Create the replacement block
    replacement_block = SimpleEvoformerReplacement(c_m, c_z, hidden_dim)
    
    # Replace the block
    model.evoformer.blocks[block_index] = replacement_block
    
    print(f"Replaced EvoformerBlock {block_index} with SimpleEvoformerReplacement")
    
    return model


def freeze_all_except_replaced_block(model, replaced_block_index: int):
    """
    Freeze all parameters except those in the replaced block
    
    Args:
        model: OpenFold model with replaced block
        replaced_block_index: Index of the replaced block to keep trainable
    
    Returns:
        Number of trainable parameters
    """
    
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        
        # Check if this parameter belongs to the replaced block
        if f"evoformer.blocks.{replaced_block_index}" in name:
            param.requires_grad = True
            trainable_params += param.numel()
            print(f"Keeping trainable: {name} ({param.numel()} params)")
        else:
            param.requires_grad = False
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    
    return trainable_params

