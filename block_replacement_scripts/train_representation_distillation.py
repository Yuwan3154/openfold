#!/usr/bin/env python3
"""
Representation Distillation Training Script

This script trains adaptive Evoformer replacement blocks to match the internal
representations (single and pair) produced by the original AlphaFold2 model.

Key features:
- Sequence-only training (no structure files required)
- On-the-fly teacher inference with DDP support
- Follows OpenFold's uniform_recycling convention
- MSE loss on single (s) and pair (z) representations

DDP Support:
- Each worker process creates its own teacher model instance
- Teacher models are NOT shared across processes (no pickling issues)
- Supports multi-GPU training with PyTorch Lightning DDP
"""

import argparse
import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import yaml
import random
import json

# Add openfold to path
sys.path.append(str(Path(__file__).parent.parent))

from openfold.config import model_config
from openfold.model.model import AlphaFold
from openfold.utils.import_weights import import_jax_weights_
from openfold.data import feature_pipeline, data_pipeline, templates
from openfold.utils.tensor_utils import tensor_tree_map

# Import adaptive training utilities
sys.path.append(str(Path(__file__).parent))
from adaptive_wrapper import (
    setup_adaptive_training_model,
    compute_adaptive_replace_loss,
    freeze_model_except_adaptive_components
)


class SequenceDataset(Dataset):
    """Dataset for loading protein sequences from FASTA or TSV files
    
    Supported formats:
    - FASTA: Standard FASTA format with '>' headers
    - TSV: Tab-separated format with no header, each line is 'id<tab>sequence'
    
    Supports two loading modes:
    - preload (default): Load all sequences into memory (faster, higher memory)
    - on_demand: Load sequences on-the-fly from file (slower, lower memory)
    """
    
    def __init__(self, dataset_file: str, max_length: Optional[int] = None, preload: bool = True):
        """
        Args:
            dataset_file: Path to FASTA or TSV file
            max_length: Maximum sequence length (sequences longer than this are filtered)
            preload: If True, load all sequences into memory. If False, load on-demand.
        """
        self.dataset_file = dataset_file
        self.max_length = max_length
        self.preload = preload
        
        # Detect file format
        self.file_format = self._detect_format(dataset_file)
        print(f"Detected file format: {self.file_format}")
        
        if preload:
            # Preload all sequences into memory
            self.sequences = self._load_sequences(dataset_file)
            # Filter by length if specified
            if max_length:
                original_count = len(self.sequences)
                self.sequences = [s for s in self.sequences if len(s['sequence']) <= max_length]
                if original_count > len(self.sequences):
                    print(f"Filtered {original_count - len(self.sequences)} sequences longer than {max_length}")
        else:
            # On-demand mode: just index the file
            self.sequence_offsets = self._index_file(dataset_file)
            # Filter by length if specified (requires reading sequences)
            if max_length:
                original_count = len(self.sequence_offsets)
                self.sequence_offsets = [
                    offset for offset in self.sequence_offsets
                    if len(self._read_sequence_at_offset(offset)['sequence']) <= max_length
                ]
                if original_count > len(self.sequence_offsets):
                    print(f"Filtered {original_count - len(self.sequence_offsets)} sequences longer than {max_length}")
            self.sequences = None  # Not preloaded
    
    def _detect_format(self, dataset_file: str) -> str:
        """Detect if file is FASTA or TSV format"""
        with open(dataset_file, 'r') as f:
            first_line = f.readline().strip()
            if first_line.startswith('>'):
                return 'fasta'
            elif '\t' in first_line:
                return 'tsv'
            else:
                # Default to FASTA if unclear
                return 'fasta'
    
    def _load_sequences(self, dataset_file: str):
        """Load sequences from FASTA or TSV file"""
        if self.file_format == 'tsv':
            return self._load_sequences_tsv(dataset_file)
        else:
            return self._load_sequences_fasta(dataset_file)
    
    def _load_sequences_fasta(self, fasta_file: str):
        """Load sequences from FASTA file"""
        sequences = []
        current_id = None
        current_seq = []
        
        with open(fasta_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    # Save previous sequence
                    if current_id is not None:
                        sequences.append({
                            'id': current_id,
                            'sequence': ''.join(current_seq)
                        })
                    # Start new sequence
                    current_id = line[1:]  # Remove '>'
                    current_seq = []
                elif line:
                    current_seq.append(line)
            
            # Save last sequence
            if current_id is not None:
                sequences.append({
                    'id': current_id,
                    'sequence': ''.join(current_seq)
                })
        
        print(f"Loaded {len(sequences)} sequences from {fasta_file}")
        return sequences
    
    def _load_sequences_tsv(self, tsv_file: str):
        """Load sequences from TSV file (format: id<tab>sequence, no header)"""
        sequences = []
        
        with open(tsv_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                
                parts = line.split('\t')
                if len(parts) != 2:
                    print(f"Warning: Line {line_num} does not have exactly 2 tab-separated fields, skipping")
                    continue
                
                seq_id, sequence = parts
                seq_id = seq_id.strip()
                sequence = sequence.strip()
                
                if not seq_id or not sequence:
                    print(f"Warning: Line {line_num} has empty id or sequence, skipping")
                    continue
                
                sequences.append({
                    'id': seq_id,
                    'sequence': sequence
                })
        
        print(f"Loaded {len(sequences)} sequences from {tsv_file}")
        return sequences
    
    def _index_file(self, dataset_file: str):
        """Index FASTA or TSV file for on-demand loading"""
        if self.file_format == 'tsv':
            return self._index_tsv(dataset_file)
        else:
            return self._index_fasta(dataset_file)
    
    def _index_fasta(self, fasta_file: str):
        """Index FASTA file for on-demand loading (stores file offsets)"""
        offsets = []
        with open(fasta_file, 'r') as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                if line.startswith('>'):
                    offsets.append(offset)
        print(f"Indexed {len(offsets)} sequences from {fasta_file}")
        return offsets
    
    def _index_tsv(self, tsv_file: str):
        """Index TSV file for on-demand loading (stores file offsets)"""
        offsets = []
        with open(tsv_file, 'r') as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                # Each non-empty line is a sequence entry
                if line.strip():
                    offsets.append(offset)
        print(f"Indexed {len(offsets)} sequences from {tsv_file}")
        return offsets
    
    def _read_sequence_at_offset(self, offset: int):
        """Read a single sequence from file at given offset"""
        if self.file_format == 'tsv':
            return self._read_sequence_at_offset_tsv(offset)
        else:
            return self._read_sequence_at_offset_fasta(offset)
    
    def _read_sequence_at_offset_fasta(self, offset: int):
        """Read a single sequence from FASTA file at given offset"""
        with open(self.dataset_file, 'r') as f:
            f.seek(offset)
            header = f.readline().strip()
            seq_id = header[1:]  # Remove '>'
            seq_lines = []
            while True:
                line = f.readline()
                if not line or line.startswith('>'):
                    break
                seq_lines.append(line.strip())
            return {'id': seq_id, 'sequence': ''.join(seq_lines)}
    
    def _read_sequence_at_offset_tsv(self, offset: int):
        """Read a single sequence from TSV file at given offset"""
        with open(self.dataset_file, 'r') as f:
            f.seek(offset)
            line = f.readline().strip()
            if not line:
                return {'id': '', 'sequence': ''}
            
            parts = line.split('\t')
            if len(parts) != 2:
                return {'id': '', 'sequence': ''}
            
            seq_id, sequence = parts
            return {'id': seq_id.strip(), 'sequence': sequence.strip()}
    
    def __len__(self):
        if self.preload:
            return len(self.sequences)
        else:
            return len(self.sequence_offsets)
    
    def __getitem__(self, idx):
        if self.preload:
            return self.sequences[idx]
        else:
            return self._read_sequence_at_offset(self.sequence_offsets[idx])


def worker_init_fn(worker_id):
    """
    Initialize each worker with its own teacher model.
    This is called once per worker process and allows each worker to have
    its own teacher model instance (solving the pickling problem).
    """
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset
        # Signal that this worker needs to initialize its teacher
        # (actual initialization happens lazily in collate_fn)
        if not hasattr(dataset, '_worker_teacher_model'):
            dataset._worker_teacher_model = None
        if not hasattr(dataset, '_worker_feature_processor'):
            dataset._worker_feature_processor = None
        if not hasattr(dataset, '_worker_data_processor'):
            dataset._worker_data_processor = None


class RepresentationDistillationDataModule(pl.LightningDataModule):
    """
    DataModule for representation distillation with DDP support.
    
    Key design:
    - Teacher model is created lazily in each worker process
    - This allows multi-GPU DDP training without pickling issues
    - Each worker/process has its own teacher model instance
    """
    
    def __init__(
        self,
        dataset_path: str,
        batch_size: int = 1,
        num_workers: int = 0,
        max_length: Optional[int] = 256,
        config_preset: str = "model_2_ptm",
        weights_path: str = None,
        use_recycling: bool = True,
        uniform_recycling: bool = True,  # Following OpenFold convention
        max_recycling_iters: int = 3,
        validation_fraction: float = 0.2,
        split_seed: int = 42,
        data_loading_strategy: str = 'on_demand',
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length
        self.config_preset = config_preset
        self.weights_path = weights_path
        self.use_recycling = use_recycling
        self.uniform_recycling = uniform_recycling
        self.max_recycling_iters = max_recycling_iters
        self.validation_fraction = validation_fraction
        self.split_seed = split_seed
        self.data_loading_strategy = data_loading_strategy
        # Convert data_loading_strategy to preload flag
        self.preload_sequences = data_loading_strategy in ['preload_cpu', 'preload_gpu']
        
        # Store config for lazy teacher creation (NOT the model itself)
        self.teacher_config = {
            'config_preset': config_preset,
            'weights_path': weights_path,
        }
        
        # Main process teacher (for single-process training)
        self._main_teacher_model = None
        self._main_feature_processor = None
        self._main_data_processor = None
        self.temp_template_dir = None
    
    def setup(self, stage=None):
        """Setup datasets with train/val split"""
        # Load full dataset from single FASTA file
        full_dataset = SequenceDataset(self.dataset_path, self.max_length, preload=self.preload_sequences)
        
        # Split from training data
        random.seed(self.split_seed)
        
        # Get total size and create shuffled indices
        total_size = len(full_dataset)
        indices = list(range(total_size))
        random.shuffle(indices)
        
        # Split indices
        val_size = int(total_size * self.validation_fraction)
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
        
        # Create subsets
        self.train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        self.val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
        
        print(f"\n{'='*60}")
        print(f"Dataset Split (seed={self.split_seed}):")
        print(f"  Total sequences: {total_size}")
        print(f"  Training: {len(self.train_dataset)} ({(1-self.validation_fraction)*100:.1f}%)")
        print(f"  Validation: {len(self.val_dataset)} ({self.validation_fraction*100:.1f}%)")
        print(f"{'='*60}\n")
    
        print(f"Training sequences: {len(self.train_dataset)}")
        print(f"Validation sequences: {len(self.val_dataset)}")
    
    def _get_or_create_teacher_components(self):
        """
        Get teacher model and processors for current process/worker.
        
        This method handles both main process and worker process cases:
        - Main process: Use _main_teacher_model
        - Worker process: Use _worker_teacher_model from dataset
        
        Returns lazy-initialized components for the current context.
        """
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is not None:
            # We're in a worker process
            dataset = worker_info.dataset
            
            # Get dataset from potential RandomSplit wrapper
            if hasattr(dataset, 'dataset'):
                actual_dataset = dataset.dataset
            else:
                actual_dataset = dataset
            
            # Lazy init for this worker
            if not hasattr(actual_dataset, '_worker_teacher_model') or actual_dataset._worker_teacher_model is None:
                print(f"Worker {worker_info.id}: Initializing teacher model...")
                (actual_dataset._worker_teacher_model,
                 actual_dataset._worker_feature_processor,
                 actual_dataset._worker_data_processor,
                 actual_dataset._worker_temp_template_dir) = self._create_teacher_components()
                print(f"Worker {worker_info.id}: Teacher model ready")
            
            return (actual_dataset._worker_teacher_model,
                    actual_dataset._worker_feature_processor,
                    actual_dataset._worker_data_processor)
        else:
            # We're in the main process
            if self._main_teacher_model is None:
                print("Main process: Initializing teacher model...")
                (self._main_teacher_model,
                 self._main_feature_processor,
                 self._main_data_processor,
                 self.temp_template_dir) = self._create_teacher_components()
                print("Main process: Teacher model ready")
            
            return (self._main_teacher_model,
                    self._main_feature_processor,
                    self._main_data_processor)
    
    def _create_teacher_components(self):
        """Create teacher model, feature processor, and data processor"""
        # Create config
        config = model_config(self.config_preset, train=False, low_prec=False)
        
        # Create teacher model
        teacher_model = AlphaFold(config)
        
        # Load weights
        if self.weights_path.endswith('.npz'):
            model_basename = Path(self.weights_path).stem
            model_version = "_".join(model_basename.split("_")[1:])
            import_jax_weights_(teacher_model, self.weights_path, version=model_version)
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
            
            teacher_model.load_state_dict(state_dict, strict=False)
        
        # Move to GPU if available and set to eval mode
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        teacher_model = teacher_model.to(device)
        teacher_model.eval()
        
        # Freeze all parameters
        for param in teacher_model.parameters():
            param.requires_grad = False
        
        # Setup data pipeline
        temp_template_dir = tempfile.mkdtemp()
        dummy_cif_path = os.path.join(temp_template_dir, "dummy.cif")
        
        # Create minimal dummy CIF file
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
        
        return teacher_model, feature_processor, data_processor, temp_template_dir
    
    def _create_features_from_sequence(self, sequence: str, seq_id: str) -> Dict[str, np.ndarray]:
        """Create features from a sequence"""
        # Get processors for current worker/process
        _, feature_processor, data_processor = self._get_or_create_teacher_components()
        
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
        feature_dict = data_processor.process_fasta(
            fasta_path=tmp_fasta_path,
            alignment_dir=local_alignment_dir,
            seqemb_mode=False
        )
        
        # Cleanup
        os.remove(tmp_fasta_path)
        shutil.rmtree(temp_alignment_dir)
        
        return feature_dict
    
    def _generate_target_representations(
        self, 
        feature_dict: Dict[str, np.ndarray],
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate target representations using teacher model.
        
        Following OpenFold's uniform_recycling convention:
        - Uniformly sample num_iters from [0, 1, ..., max_recycling_iters-1]
        - Run teacher with that many recycling iterations
        - Return representations from the final iteration
        """
        # Get teacher for current worker/process
        teacher_model, feature_processor, _ = self._get_or_create_teacher_components()
        
        # Process features for model input
        processed_features = feature_processor.process_features(
            feature_dict, mode='predict', is_multimer=False
        )
        
        # Convert to tensors and move to device
        processed_features = {
            k: torch.as_tensor(v, device=device)
            for k, v in processed_features.items()
        }
        
        # Determine number of recycling iterations following OpenFold's uniform_recycling
        if self.use_recycling and self.uniform_recycling:
            # Uniformly sample from [0, 1, ..., max_recycling_iters-1]
            # Note: max_recycling_iters=3 means we sample from [0, 1, 2]
            num_recycles = np.random.randint(0, self.max_recycling_iters)
        else:
            num_recycles = 0
        
        # Override recycling in features
        processed_features['no_recycling_iters'] = torch.tensor(num_recycles, device=device)
        
        # Run teacher model (no gradients)
        with torch.no_grad():
            outputs = teacher_model(processed_features)
        
        # Extract single (s) and pair (z) representations
        # These are after the Evoformer stack
        target_s = outputs['single']  # [N, C_s] where C_s = 384
        target_z = outputs['pair']    # [N, N, C_z] where C_z = 128
        
        return target_s, target_z
    
    def collate_fn(self, batch):
        """
        Custom collate function that generates target representations on-the-fly.
        
        This function:
        1. Takes a batch of sequences
        2. Creates features for each sequence
        3. Runs teacher model to generate target representations
        4. Prepares batch for student model
        
        Note: With DDP, each worker has its own teacher model instance.
        """
        # For representation distillation, batch_size should always be 1
        assert len(batch) == 1, "Batch size must be 1 for on-the-fly teacher inference"
        
        seq_dict = batch[0]
        seq_id = seq_dict['id']
        sequence = seq_dict['sequence']
        
        # Create features from sequence
        feature_dict = self._create_features_from_sequence(sequence, seq_id)
        
        # Determine device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Generate target representations using teacher model
        target_s, target_z = self._generate_target_representations(feature_dict, device)
        
        # Prepare student batch (sequence features without teacher inference)
        # The student will learn to generate these representations from scratch
        from openfold.data import feature_pipeline
        _, feature_processor, _ = self._get_or_create_teacher_components()
        
        student_features = feature_processor.process_features(
            feature_dict, mode='train', is_multimer=False
        )
        
        # Convert to tensors
        student_batch = {
            k: torch.as_tensor(v, device=device)
            for k, v in student_features.items()
        }
        
        # Return batch dict with targets
        return {
            'student_batch': student_batch,
            'target_s': target_s,
            'target_z': target_z,
            'seq_id': seq_id,
            'seq_length': len(sequence),
        }
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            worker_init_fn=worker_init_fn if self.num_workers > 0 else None,
            persistent_workers=True if self.num_workers > 0 else False,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            worker_init_fn=worker_init_fn if self.num_workers > 0 else None,
            persistent_workers=True if self.num_workers > 0 else False,
        )
    
    def __del__(self):
        """Cleanup temporary directories"""
        if self.temp_template_dir and os.path.exists(self.temp_template_dir):
            try:
                shutil.rmtree(self.temp_template_dir)
            except:
                pass


class RepresentationDistillationModule(pl.LightningModule):
    """
    Lightning module for training adaptive blocks with representation distillation.
    
    The student model learns to match the teacher's internal representations
    using simple MSE loss.
    """
    
    def __init__(
        self,
        config,
        adaptive_config_path: str,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        warmup_steps: int = 1000,
        replace_loss_scaler: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.config = config
        self.adaptive_config_path = adaptive_config_path
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.replace_loss_scaler = replace_loss_scaler
        
        # Create student model (will be modified with adaptive blocks in on_fit_start)
        self.student_model = AlphaFold(config)
        
        self.adaptive_setup_done = False
        self.optimizer_configured = False
    
    def on_fit_start(self):
        """Setup adaptive training after model is created"""
        if not self.adaptive_setup_done and self.adaptive_config_path:
            print("\n" + "="*80)
            print("Setting up adaptive training for representation distillation")
            print("="*80)
            
            # Import here to avoid circular dependencies
            from openfold.block_replacement_scripts.adaptive_wrapper import (
                setup_adaptive_training_model,
                freeze_model_except_adaptive_components
            )
            
            # Apply adaptive blocks
            self.student_model, training_info = setup_adaptive_training_model(
                model=self.student_model,
                config_path=Path(self.adaptive_config_path),
                model_config=self.config,
            )
            
            # Freeze all except adaptive components
            trainable_params = freeze_model_except_adaptive_components(self.student_model)
            total_params = sum(p.numel() for p in self.student_model.parameters())
            
            print(f"Adaptive blocks: {len(training_info['weight_predictors'])}")
            print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")
            print("="*80 + "\n")
            
            self.adaptive_setup_done = True
            
            # Reconfigure optimizer after model structure changes
            self._reconfigure_optimizer()
            self.optimizer_configured = True
    
    def _reconfigure_optimizer(self):
        """Reconfigure optimizer after adaptive blocks are applied"""
        if not hasattr(self, 'trainer') or self.trainer is None:
            return
        
        if not hasattr(self.trainer, 'optimizers') or not self.trainer.optimizers:
            return
        
        # Get trainable parameters after adaptive setup
        trainable_params = [p for p in self.student_model.parameters() if p.requires_grad]
        
        # Create new optimizer
        new_optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        
        # Replace in trainer
        self.trainer.optimizers = [new_optimizer]
        
        # Recreate scheduler
        def lr_lambda(current_step):
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            return 1.0
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(new_optimizer, lr_lambda)
        
        # Replace scheduler in trainer
        self.trainer.lr_scheduler_configs = [{
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1,
        }]
        
        print("Optimizer reconfigured with trainable parameters after adaptive setup")
    
    def forward(self, batch):
        """Forward pass through student model"""
        return self.student_model(batch)
    
    def _compute_loss(self, pred_s, pred_z, target_s, target_z):
        """
        Compute MSE loss between predicted and target representations.
        
        Args:
            pred_s: Predicted single representation [N, C_s]
            pred_z: Predicted pair representation [N, N, C_z]
            target_s: Target single representation [N, C_s]
            target_z: Target pair representation [N, N, C_z]
        """
        # MSE loss on single representation
        loss_s = F.mse_loss(pred_s, target_s)
        
        # MSE loss on pair representation
        loss_z = F.mse_loss(pred_z, target_z)
        
        # Total loss (simple sum, can adjust weights if needed)
        total_loss = loss_s + loss_z
        
        return total_loss, loss_s, loss_z
    
    def training_step(self, batch_dict, batch_idx):
        """Training step"""
        # Fallback: ensure optimizer is configured (safety check)
        if self.adaptive_setup_done and not self.optimizer_configured:
            self._reconfigure_optimizer()
            self.optimizer_configured = True
        
        student_batch = batch_dict['student_batch']
        target_s = batch_dict['target_s']
        target_z = batch_dict['target_z']
        seq_length = batch_dict['seq_length']
        
        # Forward pass through student model
        outputs = self(student_batch)
        
        # Extract predicted representations
        pred_s = outputs['single']
        pred_z = outputs['pair']
        
        # Compute base loss (MSE on representations)
        base_loss, loss_s, loss_z = self._compute_loss(pred_s, pred_z, target_s, target_z)
        
        # Add adaptive replace loss if applicable
        if self.adaptive_setup_done and self.replace_loss_scaler > 0:
            raw_replace_loss = compute_adaptive_replace_loss(
                model=self.student_model,
                replace_loss_scaler=1.0,  # Get raw loss
                device=base_loss.device,
            )
            scaled_replace_loss = raw_replace_loss * self.replace_loss_scaler
            total_loss = base_loss + scaled_replace_loss
            
            # Log replace loss
            self.log('train/replace_loss_raw', raw_replace_loss, on_step=True, on_epoch=True)
            self.log('train/replace_loss_scaled', scaled_replace_loss, on_step=True, on_epoch=True)
        else:
            total_loss = base_loss
        
        # Log metrics
        self.log('train/loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/loss_single', loss_s, on_step=True, on_epoch=True)
        self.log('train/loss_pair', loss_z, on_step=True, on_epoch=True)
        self.log('train/seq_length', float(seq_length), on_step=False, on_epoch=True)
        
        return total_loss
    
    def validation_step(self, batch_dict, batch_idx):
        """Validation step"""
        student_batch = batch_dict['student_batch']
        target_s = batch_dict['target_s']
        target_z = batch_dict['target_z']
        
        # Forward pass through student model
        outputs = self(student_batch)
        
        # Extract predicted representations
        pred_s = outputs['single']
        pred_z = outputs['pair']
        
        # Compute base loss (MSE on representations)
        base_loss, loss_s, loss_z = self._compute_loss(pred_s, pred_z, target_s, target_z)
        
        # Add adaptive replace loss if applicable
        if self.adaptive_setup_done and self.replace_loss_scaler > 0:
            raw_replace_loss = compute_adaptive_replace_loss(
                model=self.student_model,
                replace_loss_scaler=1.0,  # Get raw loss
                device=base_loss.device,
            )
            scaled_replace_loss = raw_replace_loss * self.replace_loss_scaler
            total_loss = base_loss + scaled_replace_loss
            
            # Log replace loss
            self.log('val/replace_loss_raw', raw_replace_loss, on_step=False, on_epoch=True, sync_dist=True)
            self.log('val/replace_loss_scaled', scaled_replace_loss, on_step=False, on_epoch=True, sync_dist=True)
        else:
            total_loss = base_loss
        
        # Log metrics
        self.log('val/loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val/loss_single', loss_s, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/loss_pair', loss_z, on_step=False, on_epoch=True, sync_dist=True)
        
        return total_loss
    
    def configure_optimizers(self):
        """Configure optimizer with warmup"""
        # Only optimize trainable parameters
        trainable_params = [p for p in self.student_model.parameters() if p.requires_grad]
        
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        
        # Linear warmup scheduler
        def lr_lambda(current_step):
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            return 1.0
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            }
        }


def main():
    parser = argparse.ArgumentParser(
        description="Train adaptive blocks with representation distillation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Config file support
    parser.add_argument('--config', type=str, default=None,
                       help='Path to YAML config file (follows AFdistill/configs/ pattern)')
    
    # Data arguments
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Path to the dataset file containing sequences; can be a FASTA file or a TSV file')
    parser.add_argument('--max_length', type=int, default=256,
                       help='Maximum sequence length')
    
    # Model arguments
    parser.add_argument('--weights_path', type=str, required=False,
                       help='Path to original AF2 weights (teacher model)')
    parser.add_argument('--adaptive_config_path', type=str, required=False,
                       help='Path to adaptive training config JSON (created automatically if not provided)')
    parser.add_argument('--trained_models_dir', type=str, required=False,
                       help='Directory containing pre-trained replacement blocks (required if adaptive_config_path not provided)')
    parser.add_argument('--linear_type', type=str, default='full',
                       choices=['full', 'diagonal', 'affine'],
                       help='Linear type for adaptive blocks (must match pre-trained blocks)')
    parser.add_argument('--config_preset', type=str, default='model_2_ptm',
                       help='OpenFold config preset')
    
    # Training arguments
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=1000,
                       help='Number of warmup steps')
    parser.add_argument('--max_epochs', type=int, default=100,
                       help='Maximum number of epochs')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size (must be 1 for on-the-fly generation)')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of dataloader workers (0 for single process, >0 for DDP)')
    parser.add_argument('--gradient_clip_val', type=float, default=1.0,
                       help='Gradient clipping value')
    
    # Recycling arguments (following OpenFold convention)
    parser.add_argument('--use_recycling', action='store_true', default=True,
                       help='Use recycling in teacher model')
    parser.add_argument('--no_recycling', dest='use_recycling', action='store_false',
                       help='Disable recycling')
    parser.add_argument('--uniform_recycling', action='store_true', default=True,
                       help='Use uniform recycling (OpenFold convention)')
    parser.add_argument('--max_recycling_iters', type=int, default=3,
                       help='Maximum recycling iterations (sample from [0, ..., n-1])')
    
    # Data split arguments
    parser.add_argument('--validation_fraction', type=float, default=0.2,
                       help='Fraction of data for validation if no val_fasta provided (default: 0.2)')
    parser.add_argument('--split_seed', type=int, default=None,
                       help='Random seed for train/val split (uses --seed if not specified)')
    
    # Additional training options (for compatibility with adaptive weighting config)
    parser.add_argument('--replace_loss_scaler', type=float, default=0.0,
                       help='Scaler for replace loss (penalizes mean adaptive weights, typically 0.0 for repr distill)')
    parser.add_argument('--data_loading_strategy', type=str, default='preload_cpu',
                       choices=['on_demand', 'preload_cpu', 'preload_gpu'],
                       help='Data loading strategy (on_demand only for repr distill - targets generated on-the-fly)')
    
    # Hardware arguments
    parser.add_argument('--gpus', type=int, default=1,
                       help='Number of GPUs')
    parser.add_argument('--precision', type=str, default='bf16-mixed',
                       choices=['16', '32', 'bf16', 'bf16-mixed'],
                       help='Training precision')
    parser.add_argument('--strategy', type=str, default='ddp',
                       choices=['ddp', 'ddp_spawn', 'dp', None],
                       help='Distributed training strategy')
    
    # Logging arguments
    parser.add_argument('--output_dir', type=str, required=False,
                       help='Output directory for checkpoints and logs')
    parser.add_argument('--experiment_name', type=str, default='repr_distill',
                       help='Experiment name')
    parser.add_argument('--wandb', action='store_true', default=False,
                       help='Use Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='af2distill',
                       help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                       help='W&B entity name')
    parser.add_argument('--log_every_n_steps', type=int, default=10,
                       help='Log every N steps')
    
    # Misc arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # CRITICAL: Determine which args were explicitly provided on command line
    # This ensures command-line args take priority over config file
    provided_args = set()
    for i, arg in enumerate(sys.argv):
        if arg.startswith('--'):
            arg_name = arg[2:]  # Remove '--'
            # Handle --arg=value format
            if '=' in arg_name:
                arg_name = arg_name.split('=')[0]
            provided_args.add(arg_name)
    
    # Load config file if provided
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Merge: Command-line args override config file values
        for key, value in config_dict.items():
            # Only use config value if NOT provided on command line
            if key not in provided_args:
                if hasattr(args, key):
                    setattr(args, key, value)
    
    # Convert relative paths to absolute (relative to home directory)
    home_dir = Path.home()
    path_args = ['dataset_path', 'weights_path', 'adaptive_config_path', 
                'trained_models_dir', 'output_dir', 'config']
    
    for path_arg in path_args:
        if hasattr(args, path_arg):
            path_value = getattr(args, path_arg)
            if path_value and not Path(path_value).is_absolute():
                # Convert to absolute path relative to home
                abs_path = str(home_dir / path_value)
                setattr(args, path_arg, abs_path)
    
    # Set split_seed to seed if not specified
    if args.split_seed is None:
        args.split_seed = args.seed
    
    # Note about data_loading_strategy
    # For representation distillation:
    # - 'on_demand': Load sequences from FASTA on-the-fly (slower, lower memory)
    # - 'preload_cpu' or 'preload_gpu': Preload all sequences to memory (faster, higher memory)
    # Targets (representations) are always generated on-the-fly by teacher model
    if hasattr(args, 'data_loading_strategy'):
        print(f"Data loading strategy: {args.data_loading_strategy}")
        print("  - Sequences: {'preloaded to memory' if args.data_loading_strategy in ['preload_cpu', 'preload_gpu'] else 'loaded on-demand'}")
        print("  - Targets (representations): Always generated on-the-fly by teacher model")
    
    # Create adaptive_config_path if not provided (like train_adaptive_weighting.py does)
    if not hasattr(args, 'adaptive_config_path') or not args.adaptive_config_path:
        if not hasattr(args, 'trained_models_dir') or not args.trained_models_dir:
            raise ValueError("Either --adaptive_config_path OR --trained_models_dir is required")
        if not hasattr(args, 'linear_type') or not args.linear_type:
            raise ValueError("--linear_type is required when trained_models_dir is provided")
        
        # Create adaptive training config file (same format as train_adaptive_weighting.py)
        adaptive_config_data = {
            "trained_models_dir": args.trained_models_dir,
            "linear_type": args.linear_type,
            "replace_loss_scaler": getattr(args, 'replace_loss_scaler', 0.0),
            "log_structure_every_k_epoch": 0,  # Not used in repr distill (no structure prediction)
            "disable_per_block_logging": True,  # Not used in repr distill
            "adaptive_training": True
        }
        
        # Create output directory if needed
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Save adaptive config
        args.adaptive_config_path = os.path.join(args.output_dir, 'adaptive_training_cmd.json')
        with open(args.adaptive_config_path, 'w') as f:
            json.dump(adaptive_config_data, f, indent=2)
        
        print(f"Created adaptive config file: {args.adaptive_config_path}")
    
    # Validate required arguments
    if not args.dataset_path:
        raise ValueError("--dataset_path is required (or provide via --config)")
    if not args.weights_path:
        raise ValueError("--weights_path is required (or provide via --config)")
    if not args.output_dir:
        raise ValueError("--output_dir is required (or provide via --config)")
    
    # Set seed
    pl.seed_everything(args.seed, workers=True)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*80)
    print("REPRESENTATION DISTILLATION TRAINING")
    print("="*80)
    print(f"dataset path: {args.dataset_path}")
    print(f"Max length: {args.max_length}")
    print(f"Weights path: {args.weights_path}")
    print(f"Adaptive config: {args.adaptive_config_path}")
    print(f"Output dir: {args.output_dir}")
    print(f"Recycling: use_recycling={args.use_recycling}, uniform={args.uniform_recycling}, max_iters={args.max_recycling_iters}")
    print(f"Hardware: {args.gpus} GPU(s), precision={args.precision}, strategy={args.strategy}")
    print(f"Num workers: {args.num_workers}")
    print("="*80 + "\n")
    
    # Create model config
    is_low_precision = args.precision in ["bf16-mixed", "16", "bf16", "16-true", "16-mixed"]
    config = model_config(args.config_preset, train=True, low_prec=is_low_precision)
    
    # Create data module
    datamodule = RepresentationDistillationDataModule(
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_length=args.max_length,
        config_preset=args.config_preset,
        weights_path=args.weights_path,
        use_recycling=args.use_recycling,
        uniform_recycling=args.uniform_recycling,
        max_recycling_iters=args.max_recycling_iters,
        validation_fraction=args.validation_fraction,
        split_seed=args.split_seed,
        data_loading_strategy=args.data_loading_strategy,
    )
    
    # Create model
    model = RepresentationDistillationModule(
        config=config,
        adaptive_config_path=args.adaptive_config_path,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        replace_loss_scaler=args.replace_loss_scaler,
    )
    
    # Setup callbacks
    callbacks = []
    
    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, 'checkpoints'),
        filename='repr_distill-{epoch:02d}-{val/loss:.4f}',
        monitor='val/loss',
        mode='min',
        save_top_k=3,
        save_last=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    # Setup logger
    loggers = []
    if args.wandb:
        wandb_logger = WandbLogger(
            name=args.experiment_name,
            project=args.wandb_project,
            entity=args.wandb_entity,
            save_dir=args.output_dir,
        )
        loggers.append(wandb_logger)
    
    # Setup strategy for DDP
    if args.gpus > 1:
        strategy = args.strategy
    else:
        strategy = None  # Single GPU doesn't need strategy
    
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
        deterministic=True,
    )
    
    # Train
    print("\nStarting training...")
    trainer.fit(model, datamodule=datamodule)
    
    print("\n" + "="*80)
    print("Training complete!")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print("="*80)


if __name__ == '__main__':
    main()
