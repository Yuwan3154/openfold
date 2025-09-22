"""
Custom Evoformer Block Replacement for Training Experiments
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Sequence


class SimpleEvoformerReplacement(nn.Module):
    """
    Simple replacement for EvoformerBlock with LayerNorm -> Linear -> ReLU -> Linear architecture
    Maintains the same input/output signature as the original EvoformerBlock
    """
    
    def __init__(self, c_m: int, c_z: int, m_hidden_dim: Optional[int] = None, z_hidden_dim: Optional[int] = None, linear_type: str = "full", gating: bool = True, residual: bool = True):
        """
        Args:
            c_m: MSA representation dimension
            c_z: Pair representation dimension  
            m_hidden_dim: Hidden dimension for the MSA linear layers (defaults to max(c_m, c_z))
            z_hidden_dim: Hidden dimension for the pair linear layers (defaults to max(c_m, c_z))
            linear_type: Type of linear layer to use (defaults to "full")
            gating: Whether to use gating mechanism (defaults to True)
            residual: Whether to use residual connections (defaults to True)
        """
        super(SimpleEvoformerReplacement, self).__init__()
        
        self.c_m = c_m
        self.c_z = c_z
        self.residual = residual
        self.gating = gating
        self.linear_type = linear_type

        if m_hidden_dim is None:
            m_hidden_dim = c_m
        if z_hidden_dim is None:
            z_hidden_dim = c_z

        if linear_type == "full":
            self.linear_class = nn.Linear
        elif linear_type == "diagonal":
            assert m_hidden_dim == c_m and z_hidden_dim == c_z, "Diagonal linear requires input_dim and hidden_dim to be the same"
            self.linear_class = DiagonalLinear
        elif linear_type == "affine":
            assert m_hidden_dim == c_m and z_hidden_dim == c_z, "Affine linear requires input_dim and hidden_dim to be the same"
            self.linear_class = AffineLinear
        else:
            raise ValueError(f"Invalid linear type: {linear_type}")

        if gating:
            # Gating layers will always be full linear layers
            self.m_gating_linear = nn.Linear(c_m, c_m)
            self.z_gating_linear = nn.Linear(c_z, c_z)

        # MSA representation processing layers
        self.msa_layer_norm = nn.LayerNorm(c_m)
        self.msa_linear1 = self.linear_class(c_m, m_hidden_dim)
        self.msa_relu = nn.ReLU()
        self.msa_linear2 = self.linear_class(m_hidden_dim, c_m)
        
        # Pair representation processing layers
        self.pair_layer_norm = nn.LayerNorm(c_z)
        self.pair_linear1 = self.linear_class(c_z, z_hidden_dim)
        self.pair_relu = nn.ReLU()
        self.pair_linear2 = self.linear_class(z_hidden_dim, c_z)

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
            if self.gating:
                gating_output = F.sigmoid(self.m_gating_linear(m))
                m_transformed = m_transformed * gating_output
            if self.residual:
                m = m + m_transformed
            else:
                m = m_transformed
            
            # Apply mask if provided
            if msa_mask is not None and _mask_trans:
                # Expand mask to match tensor dimensions
                mask_expanded = msa_mask.unsqueeze(-1)  # [batch, n_seq, n_res, 1]
                m_transformed = m_transformed * mask_expanded
        
        if z is not None:
            # Apply residual connection: z = z + f(LayerNorm(z))
            z_normalized = self.pair_layer_norm(z)
            z_transformed = self.pair_linear2(self.pair_relu(self.pair_linear1(z_normalized)))
            if self.gating:
                gating_output = F.sigmoid(self.z_gating_linear(z))
                z_transformed = z_transformed * gating_output
            if self.residual:
                z = z + z_transformed
            else:
                z = z_transformed
            
            # Apply mask if provided
            if pair_mask is not None and _mask_trans:
                # Expand mask to match tensor dimensions
                mask_expanded = pair_mask.unsqueeze(-1)  # [batch, n_res, n_res, 1]
                z_transformed = z_transformed * mask_expanded
        
        return m, z


class DiagonalLinear(nn.Module):
    def __init__(self, num_features):
        super(DiagonalLinear, self).__init__()
        # Only need a vector for the diagonal elements
        self.diag_weight = nn.Parameter(torch.randn(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        return F.linear(x, self.diag_weight, self.bias)


class AffineLinear(nn.Module):
    def __init__(self, num_features):
        super(AffineLinear, self).__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.weight * x + self.bias


def replace_evoformer_block(model, block_index: int, c_m: int, c_z: int, m_hidden_dim: Optional[int] = None, z_hidden_dim: Optional[int] = None, linear_type: str = "full", gating: bool = True, residual: bool = True):
    """
    Replace a specific EvoformerBlock with SimpleEvoformerReplacement
    
    Args:
        model: OpenFold model
        block_index: Index of the block to replace (not first or last)
        c_m: MSA representation dimension
        c_z: Pair representation dimension
        m_hidden_dim: Hidden dimension for the MSA linear layers (defaults to max(c_m, c_z))
        z_hidden_dim: Hidden dimension for the pair linear layers (defaults to max(c_m, c_z))
        linear_type: Type of linear layer to use (defaults to "full")
        gating: Whether to use gating mechanism (defaults to True)
        residual: Whether to use residual connections (defaults to True)
    
    Returns:
        The model with the replaced block
    """
    
    total_blocks = len(model.evoformer.blocks)
    
    if block_index <= 0 or block_index >= total_blocks - 1:
        raise ValueError(f"Block index {block_index} must be between 1 and {total_blocks-2} (not first or last)")
    
    # Create the replacement block
    replacement_block = SimpleEvoformerReplacement(c_m, c_z, m_hidden_dim, z_hidden_dim, linear_type, gating, residual)
    
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

