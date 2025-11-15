#!/usr/bin/env python3
"""
Per-Block Parallel Pretraining Script

This script trains all replacement blocks (blocks 1-46) simultaneously by:
1. Running teacher model inference on-the-fly to capture intermediate block outputs
2. Training individual replacement blocks sequentially within each batch
3. Avoiding disk storage by computing targets dynamically

Key features:
- On-the-fly teacher inference with block output capture
- Sequential block training within each batch to minimize memory
- Step-based checkpointing for large datasets
- Loads pretrained replacement blocks from disk
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
import yaml

# Add openfold to path
sys.path.append(str(Path(__file__).parent.parent))

from openfold.config import model_config
from openfold.model.model import AlphaFold
from openfold.utils.import_weights import import_jax_weights_
from openfold.data import feature_pipeline
from openfold.utils.tensor_utils import tensor_tree_map

# Import custom replacement blocks
from openfold.block_replacement_scripts.custom_evoformer_replacement import SimpleEvoformerReplacement


class TeacherModelWithBlockCapture:
    """
    Wrapper for teacher AlphaFold model that captures intermediate block outputs.
    
    Uses forward hooks to capture inputs and outputs of each Evoformer block during
    the forward pass. All captured tensors are detached to prevent gradient accumulation.
    """
    
    def __init__(self, model: AlphaFold, device: torch.device):
        """
        Args:
            model: AlphaFold model to wrap
            device: Device to run the model on (must be GPU)
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.block_outputs = {}
        self.hooks = []
        self._register_hooks()
        
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
    
    def _register_hooks(self):
        """Register forward hooks on all Evoformer blocks"""
        for idx, block in enumerate(self.model.evoformer.blocks):
            hook = block.register_forward_hook(self._make_hook(idx))
            self.hooks.append(hook)
    
    def _make_hook(self, block_idx: int):
        """Create a forward hook for a specific block"""
        def hook(module, inputs, outputs):
            # Inputs: (m, z, msa_mask, pair_mask, ...)
            # Outputs: (m, z)
            m_in, z_in = inputs[0], inputs[1]
            m_out, z_out = outputs[0], outputs[1]
            
            # Extract ONLY the single sequence representation (first row of MSA)
            # and clone to break connection to original tensor
            m_in_single = m_in[..., 0, :, :].detach().clone()
            m_out_single = m_out[..., 0, :, :].detach().clone()
            
            # Store detached clones to prevent gradient accumulation and memory leaks
            self.block_outputs[block_idx] = {
                'input': (m_in_single, z_in.detach().clone()),
                'output': (m_out_single, z_out.detach().clone())
            }
        return hook
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[int, Dict[str, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Run forward pass and capture all block outputs.
        
        Args:
            batch: Input batch dictionary
            
        Returns:
            Dictionary mapping block_idx to {'input': (m, z), 'output': (m, z)}
        """
        # Clear previous outputs
        self.block_outputs = {}
        
        # Run forward pass (no gradients needed)
        with torch.no_grad():
            _ = self.model(batch)
        
        return self.block_outputs
    
    def cleanup(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


class SequenceDataset(Dataset):
    """Dataset for loading protein sequences from TSV files"""
    
    def __init__(self, dataset_file: str, min_length: Optional[int] = None, max_length: Optional[int] = None):
        self.dataset_file = dataset_file
        self.min_length = min_length
        self.max_length = max_length
        
        # Load sequences
        self.sequences = self._load_sequences(dataset_file)
        
        # Filter by length
        if min_length:
            original_count = len(self.sequences)
            self.sequences = [s for s in self.sequences if len(s['sequence']) >= min_length]
            if original_count > len(self.sequences):
                print(f"Filtered {original_count - len(self.sequences)} sequences shorter than {min_length}")
        if max_length:
            original_count = len(self.sequences)
            self.sequences = [s for s in self.sequences if len(s['sequence']) <= max_length]
            if original_count > len(self.sequences):
                print(f"Filtered {original_count - len(self.sequences)} sequences longer than {max_length}")
        
        print(f"Loaded {len(self.sequences)} sequences")
    
    def _load_sequences(self, dataset_file: str) -> List[Dict[str, str]]:
        """Load sequences from TSV file"""
        sequences = []
        with open(dataset_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) >= 2:
                    seq_id = parts[0]
                    sequence = parts[1]
                    sequences.append({'id': seq_id, 'sequence': sequence})
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx]


class PerBlockDataModule(pl.LightningDataModule):
    """Data module for per-block pretraining"""
    
    def __init__(
        self,
        dataset_path: str,
        config_preset: str,
        batch_size: int = 1,
        num_workers: int = 0,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        validation_fraction: float = 0.1,
        split_seed: int = 42,
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.config_preset = config_preset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.min_length = min_length
        self.max_length = max_length
        self.validation_fraction = validation_fraction
        self.split_seed = split_seed
        
        # Setup data and feature processors
        self.data_processor, self.feature_processor = self._setup_data_pipeline()
    
    def _setup_data_pipeline(self):
        """Setup the data and feature processing pipelines"""
        import tempfile
        from openfold.data import data_pipeline, templates
        
        config = model_config(self.config_preset, train=False, low_prec=False)
        
        # Create dummy template directory with a dummy CIF file
        temp_template_dir = tempfile.mkdtemp()
        self.temp_template_dir = temp_template_dir
        dummy_cif_path = os.path.join(temp_template_dir, "dummy.cif")
        
        with open(dummy_cif_path, 'w') as f:
            f.write("""data_dummy
_entry.id dummy
_atom_site.group_PDB ATOM
_atom_site.id 1
_atom_site.type_symbol C
_atom_site.label_atom_id CA
_atom_site.label_alt_id .
_atom_site.label_comp_id ALA
_atom_site.label_asym_id A
_atom_site.label_entity_id 1
_atom_site.label_seq_id 1
_atom_site.pdbx_PDB_ins_code ?
_atom_site.Cartn_x 0.000
_atom_site.Cartn_y 0.000
_atom_site.Cartn_z 0.000
_atom_site.occupancy 1.00
_atom_site.B_iso_or_equiv 50.00
_atom_site.auth_seq_id 1
_atom_site.auth_comp_id ALA
_atom_site.auth_asym_id A
_atom_site.auth_atom_id CA
_atom_site.pdbx_PDB_model_num 1
""")
        
        template_featurizer = templates.HhsearchHitFeaturizer(
            mmcif_dir=temp_template_dir,
            max_template_date="2025-01-01",
            max_hits=0,
            kalign_binary_path="/usr/bin/kalign",
            release_dates_path=None,
            obsolete_pdbs_path=None
        )
        
        data_processor = data_pipeline.DataPipeline(template_featurizer=template_featurizer)
        feature_processor = feature_pipeline.FeaturePipeline(config.data)
        
        return data_processor, feature_processor
    
    def setup(self, stage: Optional[str] = None):
        """Setup train and validation datasets"""
        # Load all sequences
        full_dataset = SequenceDataset(
            self.dataset_path,
            min_length=self.min_length,
            max_length=self.max_length
        )
        
        # Split into train and validation
        total_size = len(full_dataset)
        val_size = int(total_size * self.validation_fraction)
        train_size = total_size - val_size
        
        # Use fixed seed for reproducible splits
        generator = torch.Generator().manual_seed(self.split_seed)
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size], generator=generator
        )
        
        print(f"Dataset split: {train_size} train, {val_size} validation")
    
    def collate_fn(self, batch):
        """Process a batch of sequences into model inputs"""
        import tempfile
        import shutil
        
        # For batch_size=1, just process single sequence
        seq_data = batch[0]
        seq_id = seq_data['id']
        sequence = seq_data['sequence']
        
        # Create temporary FASTA file
        tmp_fasta_path = os.path.join(tempfile.gettempdir(), f"tmp_{os.getpid()}_{seq_id.replace('/', '_')}.fasta")
        with open(tmp_fasta_path, 'w') as f:
            f.write(f">{seq_id}\n{sequence}")
        
        # Create temporary alignment directory
        temp_alignment_dir = tempfile.mkdtemp()
        local_alignment_dir = os.path.join(temp_alignment_dir, seq_id.replace('/', '_'))
        os.makedirs(local_alignment_dir, exist_ok=True)
        
        # Create minimal MSA file (single sequence)
        msa_path = os.path.join(local_alignment_dir, "output.a3m")
        with open(msa_path, 'w') as f:
            f.write(f">{seq_id}\n{sequence}\n")
        
        # Process features
        feature_dict = self.data_processor.process_fasta(
            fasta_path=tmp_fasta_path,
            alignment_dir=local_alignment_dir,
            seqemb_mode=False
        )
        
        # Add placeholder ground truth features
        num_res = len(sequence)
        feature_dict['all_atom_positions'] = np.zeros((num_res, 37, 3), dtype=np.float32)
        feature_dict['all_atom_mask'] = np.ones((num_res, 37), dtype=np.float32)
        feature_dict['resolution'] = np.array([0.0], dtype=np.float32)
        feature_dict['is_distillation'] = np.array(0.0, dtype=np.float32)
        feature_dict['release_date'] = np.array(['2025-01-01'.encode('utf-8')], dtype=object)
        
        # Cleanup
        os.remove(tmp_fasta_path)
        shutil.rmtree(temp_alignment_dir)
        
        # Process features
        processed_features = self.feature_processor.process_features(
            feature_dict, mode='train', is_multimer=False
        )
        
        # Convert to tensors
        processed_batch = {
            k: torch.as_tensor(v)
            for k, v in processed_features.items()
        }
        
        # Add sequence metadata
        processed_batch['seq_id'] = seq_id
        processed_batch['seq_length'] = len(sequence)
        
        return processed_batch
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            persistent_workers=False
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            persistent_workers=False
        )
    
    def __del__(self):
        """Cleanup temporary directory"""
        import shutil
        if hasattr(self, 'temp_template_dir'):
            shutil.rmtree(self.temp_template_dir, ignore_errors=True)



class ParallelBlockPretrainer(pl.LightningModule):
    """
    Lightning module for training all replacement blocks in parallel.
    
    Loads pretrained replacement blocks and trains them sequentially within each batch
    using targets generated on-the-fly from the teacher model.
    """
    
    def __init__(
        self,
        config,
        config_preset: str,
        weights_path: str,
        trained_models_dir: str,
        linear_type: str = 'full',
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.config = config
        self.config_preset = config_preset
        self.weights_path = weights_path
        self.trained_models_dir = trained_models_dir
        self.linear_type = linear_type
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Get dimensions from config
        self.c_m = config.model.evoformer_stack.c_m
        self.c_z = config.model.evoformer_stack.c_z
        
        # Create teacher model with block capture
        print("Loading teacher model...")
        teacher_model = self._load_teacher_model()
        self.teacher = None  # Will be initialized in setup
        self._teacher_model = teacher_model  # Store for later initialization
        
        # Load all replacement blocks
        print("Loading replacement blocks...")
        self.replacement_blocks = nn.ModuleDict()
        self._load_replacement_blocks()
        
        print(f"Initialized ParallelBlockPretrainer with {len(self.replacement_blocks)} blocks")
    
    def _load_teacher_model(self) -> AlphaFold:
        """Load the pretrained teacher model"""
        config = model_config(self.config_preset, train=False, low_prec=False)
        model = AlphaFold(config)
        
        # Load weights
        if self.weights_path.endswith('.npz'):
            model_basename = Path(self.weights_path).stem
            model_version = "_".join(model_basename.split("_")[1:])
            import_jax_weights_(model, self.weights_path, version=model_version)
        else:
            checkpoint = torch.load(self.weights_path, map_location='cpu')
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            
            if any(k.startswith('model.') for k in state_dict.keys()):
                state_dict = {k[6:]: v for k, v in state_dict.items() if k.startswith('model.')}
            
            model.load_state_dict(state_dict, strict=False)
        
        return model
    
    def _load_replacement_blocks(self):
        """Load all pretrained replacement blocks from disk"""
        blocks_loaded = 0
        blocks_failed = 0
        
        for block_idx in range(1, 47):
            block_path = Path(self.trained_models_dir) / f"block_{block_idx:02d}" / self.linear_type / "best_model.ckpt"
            
            if not block_path.exists():
                print(f"Warning: Block {block_idx} checkpoint not found at {block_path}")
                blocks_failed += 1
                continue
            
            # Create replacement block
            replacement_block = SimpleEvoformerReplacement(
                c_m=self.c_m,
                c_z=self.c_z,
                linear_type=self.linear_type
            )
            
            # Load checkpoint
            checkpoint = torch.load(block_path, map_location='cpu')
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Strip 'replacement_block.' prefix if present
            if any(k.startswith('replacement_block.') for k in state_dict.keys()):
                state_dict = {k.replace('replacement_block.', ''): v for k, v in state_dict.items()}
            
            # Load weights
            replacement_block.load_state_dict(state_dict, strict=True)
            
            # Store in ModuleDict
            self.replacement_blocks[str(block_idx)] = replacement_block
            blocks_loaded += 1
        
        print(f"Loaded {blocks_loaded} replacement blocks, {blocks_failed} failed")
        
        if blocks_loaded == 0:
            raise ValueError(f"No replacement blocks loaded from {self.trained_models_dir}")
    
    def setup(self, stage: Optional[str] = None):
        """Setup teacher model on correct device"""
        if self.teacher is None:
            device = self.device if hasattr(self, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.teacher = TeacherModelWithBlockCapture(self._teacher_model, device)
            print(f"Teacher model initialized on {device}")
    
    def forward(self, batch):
        """Not used - training is done in training_step"""
        pass
    
    def training_step(self, batch, batch_idx):
        """
        Training step that:
        1. Runs teacher model to capture all block outputs
        2. Trains each replacement block sequentially
        3. Returns average loss across all blocks
        """
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Extract metadata
        seq_length = batch.pop('seq_length')
        seq_id = batch.pop('seq_id')
        
        # Get masks from batch
        # msa_mask should be [batch, n_seq, n_res]
        msa_mask = batch.get('msa_mask', None)
        # seq_mask is [batch, n_res], need to expand to pair_mask [batch, n_res, n_res]
        seq_mask = batch.get('seq_mask', None)
        if seq_mask is not None:
            pair_mask = seq_mask.unsqueeze(-1) * seq_mask.unsqueeze(-2)
        else:
            pair_mask = None
        
        # Run teacher model to capture all block outputs
        block_data = self.teacher.forward(batch)
        
        # Train each replacement block sequentially
        total_loss = 0.0
        num_blocks = 0
        
        for block_idx in range(1, 47):
            if str(block_idx) not in self.replacement_blocks:
                continue
            
            if block_idx not in block_data:
                continue
            
            # Get block input/output from teacher (already single representation)
            m_in_single, z_in = block_data[block_idx]['input']
            m_target_single, z_target = block_data[block_idx]['output']
            
            # Expand single representation back to MSA format for replacement block
            # Shape: [batch, n_res, c_m] -> [batch, 1, n_res, c_m]
            m_in_msa = m_in_single.unsqueeze(1)
            
            # Run replacement block
            replacement_block = self.replacement_blocks[str(block_idx)]
            m_pred_msa, z_pred = replacement_block(
                m_in_msa, z_in, msa_mask, pair_mask,
                chunk_size=None,
                use_deepspeed_evo_attention=False,
                use_lma=False,
                use_flash=False,
                inplace_safe=False,
                _mask_trans=False  # Disable mask application to avoid dimension issues
            )
            
            # Extract single representation from output
            # m_pred_msa shape: [batch, n_seq, n_res, c_m] where n_seq=1
            # We want: [batch, n_res, c_m]
            m_pred_single = m_pred_msa.squeeze(1)  # Remove the n_seq dimension
            
            # Compute MSE loss on single representation and pair representation
            loss_single = F.mse_loss(m_pred_single, m_target_single)
            loss_pair = F.mse_loss(z_pred, z_target)
            
            # Combine losses
            block_loss = loss_single + loss_pair
            total_loss += block_loss
            num_blocks += 1
            
            # Log per-block loss for all 46 blocks
            self.log(f'train/block_{block_idx:02d}_loss', block_loss, on_step=True, on_epoch=True, batch_size=1)
            
            # Clear intermediate tensors to free memory
            del m_in_msa, m_pred_msa, m_pred_single, block_loss
        
        # Average loss across all blocks
        if num_blocks > 0:
            avg_loss = total_loss / num_blocks
        else:
            avg_loss = torch.tensor(0.0, device=self.device)
        
        # Log overall metrics
        self.log('train/loss', avg_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=1)
        self.log('train/num_blocks_trained', float(num_blocks), on_step=False, on_epoch=True, batch_size=1)
        self.log('train/seq_length', float(seq_length), on_step=False, on_epoch=True, batch_size=1)
        
        # Clear block_data to free GPU memory
        del block_data
        torch.cuda.empty_cache()
        
        return avg_loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step - same as training but without gradient updates"""
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Extract metadata
        seq_length = batch.pop('seq_length')
        seq_id = batch.pop('seq_id')
        
        # Get masks from batch
        # msa_mask should be [batch, n_seq, n_res]
        msa_mask = batch.get('msa_mask', None)
        # seq_mask is [batch, n_res], need to expand to pair_mask [batch, n_res, n_res]
        seq_mask = batch.get('seq_mask', None)
        if seq_mask is not None:
            pair_mask = seq_mask.unsqueeze(-1) * seq_mask.unsqueeze(-2)
        else:
            pair_mask = None
        
        # Run teacher model to capture all block outputs
        block_data = self.teacher.forward(batch)
        
        # Evaluate each replacement block
        total_loss = 0.0
        num_blocks = 0
        
        for block_idx in range(1, 47):
            if str(block_idx) not in self.replacement_blocks:
                continue
            
            if block_idx not in block_data:
                continue
            
            # Get block input/output from teacher (already single representation)
            m_in_single, z_in = block_data[block_idx]['input']
            m_target_single, z_target = block_data[block_idx]['output']
            
            # Expand single representation back to MSA format for replacement block
            m_in_msa = m_in_single.unsqueeze(1)
            
            # Run replacement block
            replacement_block = self.replacement_blocks[str(block_idx)]
            m_pred_msa, z_pred = replacement_block(
                m_in_msa, z_in, msa_mask, pair_mask,
                chunk_size=None,
                use_deepspeed_evo_attention=False,
                use_lma=False,
                use_flash=False,
                inplace_safe=False,
                _mask_trans=False  # Disable mask application to avoid dimension issues
            )
            
            # Extract single representation from output
            # m_pred_msa shape: [batch, n_seq, n_res, c_m] where n_seq=1
            # We want: [batch, n_res, c_m]
            m_pred_single = m_pred_msa.squeeze(1)  # Remove the n_seq dimension
            
            # Compute MSE loss
            loss_single = F.mse_loss(m_pred_single, m_target_single)
            loss_pair = F.mse_loss(z_pred, z_target)
            
            block_loss = loss_single + loss_pair
            total_loss += block_loss
            num_blocks += 1
            
            # Log per-block loss for all 46 blocks
            self.log(f'val/block_{block_idx:02d}_loss', block_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=1)
            
            # Clear intermediate tensors
            del m_in_msa, m_pred_msa, m_pred_single, block_loss
        
        # Average loss
        if num_blocks > 0:
            avg_loss = total_loss / num_blocks
        else:
            avg_loss = torch.tensor(0.0, device=self.device)
        
        # Log validation metrics
        self.log('val/loss', avg_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=1)
        self.log('val/num_blocks_evaluated', float(num_blocks), on_step=False, on_epoch=True, sync_dist=True, batch_size=1)
        
        # Clear block_data to free GPU memory
        del block_data
        torch.cuda.empty_cache()
        
        return avg_loss
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        # Only optimize replacement block parameters
        optimizer = torch.optim.AdamW(
            self.replacement_blocks.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        return optimizer


def main(args, parser):
    # Set float32 matmul precision for Tensor Cores
    torch.set_float32_matmul_precision("medium")
    
    # Set random seeds
    if args.seed is not None:
        pl.seed_everything(args.seed, workers=True)
    
    # Create model config
    config = model_config(
        args.config_preset,
        train=True,
        low_prec=(args.precision in ["bf16-mixed", "16", "bf16", "16-true", "16-mixed"])
    )
    
    # Create data module
    print("Setting up data module...")
    data_module = PerBlockDataModule(
        dataset_path=args.dataset_path,
        config_preset=args.config_preset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        min_length=args.min_length,
        max_length=args.max_length,
        validation_fraction=args.validation_fraction,
        split_seed=args.split_seed,
    )
    
    # Create model
    print("Setting up model...")
    model = ParallelBlockPretrainer(
        config=config,
        config_preset=args.config_preset,
        weights_path=args.weights_path,
        trained_models_dir=args.trained_models_dir,
        linear_type=args.linear_type,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    
    # Setup callbacks
    callbacks = []
    
    # Checkpoint callback with step-based saving
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, 'checkpoints'),
        filename='per_block-{epoch:02d}-{step:06d}',
        every_n_train_steps=args.checkpoint_every_n_steps,
        save_top_k=args.save_top_k,
        monitor='train/loss',
        mode='min',
        save_last=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Learning rate monitor
    if args.log_lr:
        lr_monitor = LearningRateMonitor(logging_interval='step')
        callbacks.append(lr_monitor)
    
    # Setup loggers
    loggers = []
    if args.wandb:
        wandb_logger = WandbLogger(
            name=args.experiment_name,
            project=args.wandb_project,
            entity=args.wandb_entity,
            save_dir=args.output_dir,
        )
        loggers.append(wandb_logger)
    
    # Setup strategy
    if args.gpus > 1:
        strategy = DDPStrategy(
            find_unused_parameters=False,
            process_group_backend=args.distributed_backend
        )
    else:
        strategy = 'auto'
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu' if args.gpus > 0 else 'cpu',
        devices=args.gpus if args.gpus > 0 else None,
        strategy=strategy,
        precision=args.precision,
        gradient_clip_val=args.gradient_clip_val,
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=args.log_every_n_steps,
        val_check_interval=args.val_check_interval,
        limit_val_batches=args.limit_val_batches,  # Limit validation to N randomly sampled batches
        deterministic=True,
    )
    
    # Train
    print("Starting training...")
    trainer.fit(model, datamodule=data_module)
    
    print("Training completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Per-Block Parallel Pretraining')
    
    # Config file (optional, but if provided, makes other args optional)
    parser.add_argument('--config', type=str, default=None,
                       help='Path to YAML config file')
    
    # Data arguments
    parser.add_argument('--dataset_path', type=str, required=False,
                       help='Path to dataset TSV file')
    parser.add_argument('--weights_path', type=str, required=False,
                       help='Path to pretrained teacher model weights')
    parser.add_argument('--trained_models_dir', type=str, required=False,
                       help='Directory containing pretrained replacement blocks')
    parser.add_argument('--output_dir', type=str, required=False,
                       help='Output directory for checkpoints and logs')
    
    # Model arguments
    parser.add_argument('--config_preset', type=str, default='model_1_ptm',
                       help='Model config preset')
    parser.add_argument('--linear_type', type=str, default='full',
                       choices=['full', 'diagonal', 'affine'],
                       help='Type of linear layers in replacement blocks')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size (must be 1 for now)')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of data loader workers')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--max_epochs', type=int, default=100,
                       help='Maximum number of epochs')
    parser.add_argument('--gradient_clip_val', type=float, default=1.0,
                       help='Gradient clipping value')
    
    # Data filtering arguments
    parser.add_argument('--min_length', type=int, default=50,
                       help='Minimum sequence length')
    parser.add_argument('--max_length', type=int, default=384,
                       help='Maximum sequence length')
    parser.add_argument('--validation_fraction', type=float, default=0.1,
                       help='Fraction of data to use for validation')
    parser.add_argument('--split_seed', type=int, default=42,
                       help='Random seed for train/val split')
    
    # Checkpoint arguments
    parser.add_argument('--checkpoint_every_n_steps', type=int, default=1000,
                       help='Save checkpoint every N steps')
    parser.add_argument('--save_top_k', type=int, default=3,
                       help='Number of best checkpoints to keep')
    parser.add_argument('--val_check_interval', type=int, default=100,
                       help='Run validation every N training steps')
    parser.add_argument('--limit_val_batches', type=int, default=10,
                       help='Number of validation batches to run per validation check (randomly sampled)')
    
    # Logging arguments
    parser.add_argument('--log_every_n_steps', type=int, default=10,
                       help='Log metrics every N steps')
    parser.add_argument('--log_lr', action='store_true', default=False,
                       help='Log learning rate')
    parser.add_argument('--wandb', action='store_true', default=False,
                       help='Use Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='af2distill',
                       help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                       help='W&B entity name')
    parser.add_argument('--experiment_name', type=str, default='per_block_parallel',
                       help='Experiment name for logging')
    
    # Hardware arguments
    parser.add_argument('--gpus', type=int, default=1,
                       help='Number of GPUs to use')
    parser.add_argument('--precision', type=str, default='bf16-mixed',
                       help='Training precision')
    parser.add_argument('--distributed_backend', type=str, default='gloo',
                       choices=['nccl', 'gloo', 'mpi'],
                       help='Distributed backend')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Load config from YAML if provided (before validation)
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Get list of arguments provided on command line
        provided_args = set()
        for action in vars(args):
            if getattr(args, action) != parser.get_default(action):
                provided_args.add(action)
        
        # Update args with config values (command line takes precedence)
        for key, value in config_dict.items():
            if key not in provided_args:
                if hasattr(args, key):
                    setattr(args, key, value)
    
    # Validate required arguments
    required_args = ['dataset_path', 'weights_path', 'trained_models_dir', 'output_dir']
    missing_args = [arg for arg in required_args if getattr(args, arg) is None]
    if missing_args:
        parser.error(f"The following arguments are required: {', '.join('--' + arg for arg in missing_args)}")
    
    # Validate arguments
    if args.batch_size != 1:
        raise ValueError("batch_size must be 1 for per-block training")
    
    main(args, parser)

