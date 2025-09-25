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

# Add block_replacement_scripts to path
import sys
sys.path.append(str(Path(__file__).parent / "block_replacement_scripts"))

from adaptive_evoformer_blocks import (
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
    
    print("\n=== Setting up Adaptive Training ===")
    print(f"Trained models directory: {adaptive_config['trained_models_dir']}")
    print(f"Linear type: {adaptive_config['linear_type']}")
    print(f"Replace loss scaler: {adaptive_config['replace_loss_scaler']}")
    
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
    }
    
    print(f"Successfully set up adaptive training with {len(weight_predictors)} blocks")
    
    return model, training_info


def compute_adaptive_replace_loss(
    weight_predictors: Dict[int, AdaptiveWeightPredictor],
    replace_loss_scaler: float,
    c_m: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute replace loss that penalizes mean of adaptive weights.
    
    Args:
        weight_predictors: Dictionary of weight predictors
        replace_loss_scaler: Scaling factor for replace loss
        c_m: MSA channel dimension
        device: Device to create tensors on
        
    Returns:
        Replace loss value
    """
    if not weight_predictors:
        return torch.tensor(0.0, device=device)
    
    # Collect all predicted weights
    all_weights = []
    
    # Get a dummy input for weight prediction (batch size 1)
    dummy_msa = torch.zeros(1, 1, 1, c_m, device=device)
    
    for block_idx, predictor in weight_predictors.items():
        weight = predictor(dummy_msa)  # [1, 1]
        all_weights.append(weight.squeeze())
    
    # Compute mean weight across all blocks
    mean_weight = torch.stack(all_weights).mean()
    
    # Penalize deviation from 1.0 (encouraging use of original Evoformer)
    replace_loss = (1.0 - mean_weight) ** 2
    
    return replace_loss * replace_loss_scaler


def freeze_model_except_adaptive_weights(model: nn.Module) -> int:
    """
    Freeze all model parameters except adaptive weight predictors.
    
    Args:
        model: The model to freeze
        
    Returns:
        Number of trainable parameters
    """
    # First, freeze everything
    for param in model.parameters():
        param.requires_grad = False
    
    # Then unfreeze weight predictors
    trainable_params = 0
    for name, module in model.named_modules():
        if isinstance(module, AdaptiveWeightPredictor):
            for param in module.parameters():
                param.requires_grad = True
                trainable_params += param.numel()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Frozen all parameters except adaptive weights")
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    return trainable_params
