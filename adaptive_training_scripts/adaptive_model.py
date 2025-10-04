#!/usr/bin/env python3
"""
Adaptive Model - Handles model creation, weight loading, and block replacement.

This module provides clean separation between:
1. Model creation with correct config
2. Weight loading (both PyTorch .pt and JAX .npz)
3. Evoformer block replacement with adaptive blocks
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Tuple
import sys

# Add openfold to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from openfold.config import model_config
from openfold.model.model import AlphaFold
from openfold.utils.import_weights import import_jax_weights_

# Import replacement block components from block_replacement_scripts
from block_replacement_scripts.adaptive_evoformer_blocks import (
    AdaptiveEvoformerBlock,
    AdaptiveWeightPredictor,
    load_pretrained_replacement_block
)
from block_replacement_scripts.custom_evoformer_replacement import SimpleEvoformerReplacement


class AdaptiveModelBuilder:
    """Builds and configures adaptive models with proper weight loading."""
    
    def __init__(
        self,
        weights_path: str,
        trained_models_dir: str,
        linear_type: str = "full",
        replace_loss_scaler: float = 1.0,
        c_m: int = 256,
        c_z: int = 128,
    ):
        """
        Initialize the model builder.
        
        Args:
            weights_path: Path to pre-trained weights (.pt or .npz)
            trained_models_dir: Directory containing pre-trained replacement blocks
            linear_type: Type of linear layer in replacement blocks
            replace_loss_scaler: Scaling factor for replace loss
            c_m: MSA channel dimension
            c_z: Pair channel dimension
        """
        self.weights_path = Path(weights_path)
        self.trained_models_dir = Path(trained_models_dir)
        self.linear_type = linear_type
        self.replace_loss_scaler = replace_loss_scaler
        self.c_m = c_m
        self.c_z = c_z
        
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {self.weights_path}")
        if not self.trained_models_dir.exists():
            raise FileNotFoundError(f"Trained models directory not found: {self.trained_models_dir}")
    
    def build_model(self, config_preset: str = "finetuning_no_templ_ptm") -> Tuple[AlphaFold, Dict]:
        """
        Build the complete adaptive model.
        
        Steps:
        1. Create base model with appropriate config
        2. Load pre-trained weights (PTM checkpoint)
        3. Replace Evoformer blocks with adaptive blocks
        4. Return model and training info
        
        Args:
            config_preset: Config preset to use
            
        Returns:
            model: Modified AlphaFold model with adaptive blocks
            training_info: Dictionary with training configuration
        """
        print("\n" + "=" * 80)
        print("Building Adaptive Model")
        print("=" * 80)
        
        # Step 1: Create base model
        print("\n1. Creating base model...")
        config = model_config(config_preset, train=True, low_prec=True)
        model = AlphaFold(config)
        print(f"   Config preset: {config_preset}")
        print(f"   Template enabled in model: {config.model.template.enabled}")
        print(f"   Template data usage: {config.data.common.use_templates}")
        
        # Step 2: Load pre-trained weights
        print("\n2. Loading pre-trained weights...")
        self._load_weights(model, config_preset)
        
        # Step 3: Replace Evoformer blocks with adaptive blocks
        print("\n3. Replacing Evoformer blocks with adaptive versions...")
        weight_predictors = self._replace_with_adaptive_blocks(model)
        
        # Step 4: Prepare training info
        training_info = {
            'weight_predictors': weight_predictors,
            'replace_loss_scaler': self.replace_loss_scaler,
            'linear_type': self.linear_type,
            'num_adaptive_blocks': len(weight_predictors),
            'config': config,
        }
        
        print("\n" + "=" * 80)
        print(f"Successfully built adaptive model with {len(weight_predictors)} adaptive blocks")
        print("=" * 80 + "\n")
        
        return model, training_info
    
    def _load_weights(self, model: AlphaFold, config_preset: str):
        """
        Load pre-trained weights into the model.
        
        Supports both PyTorch (.pt) and JAX (.npz) weights.
        Uses strict=False to allow template-related weight mismatches.
        """
        if self.weights_path.suffix == ".npz":
            # JAX weights
            print(f"   Loading JAX weights from: {self.weights_path}")
            model_basename = self.weights_path.stem
            # Extract version (e.g., "params_model_2_ptm" -> "model_2_ptm")
            model_version = "_".join(model_basename.split("_")[1:])
            
            # Import JAX weights
            import_jax_weights_(model, str(self.weights_path), version=model_version)
            print(f"   ✓ Loaded JAX weights (version: {model_version})")
            
        else:
            # PyTorch weights (.pt)
            print(f"   Loading PyTorch weights from: {self.weights_path}")
            checkpoint = torch.load(self.weights_path, map_location="cpu")
            
            # Extract state dict from checkpoint
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            
            # Handle 'model.' prefix in keys
            if any(k.startswith('model.') for k in state_dict.keys()):
                state_dict = {k.replace('model.', '', 1): v for k, v in state_dict.items()}
            
            # Load with strict=False to allow missing template keys
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            
            # Filter out template-related missing keys (expected)
            template_missing = [k for k in missing_keys if 'template' in k]
            other_missing = [k for k in missing_keys if 'template' not in k]
            
            print(f"   ✓ Loaded PyTorch weights")
            if template_missing:
                print(f"   - Skipped {len(template_missing)} template-related keys (expected)")
            if other_missing:
                print(f"   ⚠ Warning: {len(other_missing)} non-template keys missing")
                if len(other_missing) <= 10:
                    for key in other_missing:
                        print(f"      - {key}")
            if unexpected_keys:
                print(f"   ⚠ Warning: {len(unexpected_keys)} unexpected keys")
    
    def _replace_with_adaptive_blocks(self, model: AlphaFold) -> Dict[int, AdaptiveWeightPredictor]:
        """
        Replace Evoformer blocks with adaptive versions.
        
        Args:
            model: The AlphaFold model to modify
            
        Returns:
            Dictionary mapping block indices to weight predictors
        """
        # Find available trained replacement blocks
        available_blocks = []
        for block_dir in self.trained_models_dir.glob("block_*"):
            if block_dir.is_dir():
                checkpoint_path = block_dir / self.linear_type / "best_model.ckpt"
                if checkpoint_path.exists():
                    block_idx = int(block_dir.name.split("_")[1])
                    available_blocks.append(block_idx)
        
        available_blocks.sort()
        print(f"   Found {len(available_blocks)} trained replacement blocks")
        if available_blocks:
            print(f"   Blocks: {available_blocks[:10]}{'...' if len(available_blocks) > 10 else ''}")
        
        if not available_blocks:
            raise ValueError(f"No trained replacement blocks found in {self.trained_models_dir}")
        
        # Replace blocks
        weight_predictors = {}
        evoformer_stack = model.evoformer
        
        for block_idx in available_blocks:
            if block_idx >= len(evoformer_stack.blocks):
                print(f"   ⚠ Warning: Block {block_idx} out of range, skipping")
                continue
            
            # Load pre-trained replacement block
            checkpoint_path = self.trained_models_dir / f"block_{block_idx:02d}" / self.linear_type / "best_model.ckpt"
            replacement_block = load_pretrained_replacement_block(
                checkpoint_path=checkpoint_path,
                c_m=self.c_m,
                c_z=self.c_z,
                linear_type=self.linear_type,
                hidden_dim=None,  # Use default
            )
            
            # Create weight predictor
            weight_predictor = AdaptiveWeightPredictor(c_m=self.c_m)
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
        
        print(f"   ✓ Replaced {len(weight_predictors)} Evoformer blocks with adaptive versions")
        
        return weight_predictors


def freeze_model_except_adaptive_components(model: nn.Module) -> Tuple[int, int]:
    """
    Freeze all model parameters except adaptive weight predictors and replacement blocks.
    
    Args:
        model: The model to freeze
        
    Returns:
        Tuple of (trainable_params, total_params)
    """
    # First, freeze everything
    for param in model.parameters():
        param.requires_grad = False
    
    # Then unfreeze adaptive components
    trainable_params = 0
    for name, module in model.named_modules():
        if isinstance(module, AdaptiveWeightPredictor):
            # Unfreeze weight predictors
            for param in module.parameters():
                param.requires_grad = True
                trainable_params += param.numel()
        elif isinstance(module, AdaptiveEvoformerBlock):
            # Unfreeze replacement blocks (already handled in AdaptiveEvoformerBlock.__init__)
            for param in module.replacement_block.parameters():
                param.requires_grad = True
                trainable_params += param.numel()
    
    total_params = sum(p.numel() for p in model.parameters())
    
    return trainable_params, total_params


def compute_adaptive_replace_loss(
    model: torch.nn.Module,
    replace_loss_scaler: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute replace loss that encourages use of replacement blocks.
    
    This penalizes high adaptive weights (weight=1 means full original Evoformer,
    weight=0 means full replacement block).
    
    Args:
        model: The model containing adaptive blocks
        replace_loss_scaler: Scaling factor for the loss
        device: Device to create tensors on
        
    Returns:
        Replace loss value
    """
    # Collect all predicted weights from adaptive blocks
    all_weights = []
    
    for name, module in model.named_modules():
        if hasattr(module, '_predicted_weights') and hasattr(module, 'block_idx'):
            if module.block_idx in module._predicted_weights:
                weight = module._predicted_weights[module.block_idx]
                all_weights.append(weight.mean())
    
    if not all_weights:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # Compute mean weight across all blocks
    mean_weight = torch.stack(all_weights).mean()
    
    # Penalize high weights (encourage use of replacement blocks)
    replace_loss = mean_weight * replace_loss_scaler
    
    return replace_loss

