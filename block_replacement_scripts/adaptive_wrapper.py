#!/usr/bin/env python3
"""
Adaptive training wrapper that modifies OpenFold models for adaptive weighting.

This module provides functions to modify an existing OpenFold model to support
adaptive weighting between original and replacement Evoformer blocks.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Tuple
import json

# Import adaptive evoformer blocks
from .adaptive_evoformer_blocks import (
    replace_evoformer_blocks_with_adaptive,
    AdaptiveWeightPredictor
)


def setup_adaptive_training_model(
    model: nn.Module,
    config_path: Path,
    model_config: dict,
) -> Tuple[nn.Module, Dict[str, any]]:
    """
    Modify an OpenFold model for adaptive training.
    
    Args:
        model: The OpenFold model to modify
        config_path: Path to adaptive training configuration
        model_config: Model configuration dictionary
        
    Returns:
        model: Modified model with adaptive blocks
        training_info: Dictionary with training information
    """
    # Load adaptive config
    with open(config_path, 'r') as f:
        adaptive_config = json.load(f)
    
    # Note: We can't access trainer.is_global_zero here, so these prints will happen on all ranks
    # They will be filtered in the wrapper's _setup_adaptive_training method
    
    # Get model dimensions
    c_m = model_config.model.evoformer_stack.c_m
    c_z = model_config.model.evoformer_stack.c_z
    
    # Replace Evoformer blocks with adaptive versions
    model, weight_predictors = replace_evoformer_blocks_with_adaptive(
        model=model,
        trained_models_dir=Path(adaptive_config['trained_models_dir']),
        linear_type=adaptive_config['linear_type'],
        c_m=c_m,
        c_z=c_z,
        hidden_dim=None,  # Use default based on linear type
        replace_blocks=None,  # Replace all available blocks
    )
    
    # Create training info
    training_info = {
        'weight_predictors': weight_predictors,
        'replace_loss_scaler': adaptive_config['replace_loss_scaler'],
        'linear_type': adaptive_config['linear_type'],
        'num_adaptive_blocks': len(weight_predictors),
        'log_structure_every_k_epoch': adaptive_config.get('log_structure_every_k_epoch', 1),
    }
    
    # Print is handled in wrapper
    return model, training_info


def compute_adaptive_replace_loss(
    model: torch.nn.Module,
    replace_loss_scaler: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute replace loss that penalizes mean of adaptive weights.
    
    This function collects the actual predicted weights from the adaptive blocks
    during the forward pass, rather than using dummy inputs.
    
    Args:
        model: The model containing adaptive blocks with predicted weights
        replace_loss_scaler: Scaling factor for replace loss
        device: Device to create tensors on
        
    Returns:
        Replace loss value
    """
    # Collect all predicted weights from adaptive blocks
    all_weights = []
    
    # Find all AdaptiveEvoformerBlock instances and collect their predicted weights
    for name, module in model.named_modules():
        # Check if this is an AdaptiveEvoformerBlock with stored weights
        if hasattr(module, '_predicted_weights') and hasattr(module, 'block_idx'):
            if module.block_idx in module._predicted_weights:
                weight = module._predicted_weights[module.block_idx]
                # Average over batch dimension
                all_weights.append(weight.mean())
    
    if not all_weights:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # Compute mean weight across all blocks
    mean_weight = torch.stack(all_weights).mean()
    
    # Penalize deviation from 0.0 (encouraging use of replacement Evoformer)
    replace_loss = mean_weight
    
    return replace_loss * replace_loss_scaler


def freeze_model_except_adaptive_components(model: nn.Module) -> int:
    """
    Freeze all model parameters except adaptive weight predictors and replacement blocks.
    
    Args:
        model: The model to freeze
        
    Returns:
        Number of trainable parameters
    """
    # First, freeze everything
    for param in model.parameters():
        param.requires_grad = False
    
    # Then unfreeze adaptive components (weight predictors and replacement blocks)
    trainable_params = 0
    for name, module in model.named_modules():
        if isinstance(module, AdaptiveWeightPredictor):
            # Unfreeze weight predictors
            for param in module.parameters():
                param.requires_grad = True
                trainable_params += param.numel()
        elif hasattr(module, 'replacement_block'):
            # Unfreeze replacement blocks (already handled in AdaptiveEvoformerBlock.__init__)
            for param in module.replacement_block.parameters():
                param.requires_grad = True
                trainable_params += param.numel()
    
    total_params = sum(p.numel() for p in model.parameters())
    # Print is handled in wrapper
    
    return trainable_params
