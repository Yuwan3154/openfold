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
from pytorch_lightning.strategies import DDPStrategy
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
from openfold.block_replacement_scripts.adaptive_wrapper import (
    setup_adaptive_training_model,
    compute_adaptive_replace_loss,
    compute_block_match_loss,
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
    
    def __init__(self, dataset_file: str, min_length: Optional[int] = None, max_length: Optional[int] = None, preload: bool = True):
        """
        Args:
            dataset_file: Path to FASTA or TSV file
            min_length: Minimum sequence length (sequences shorter than this are filtered)
            max_length: Maximum sequence length (sequences longer than this are filtered)
            preload: If True, load all sequences into memory. If False, load on-demand.
        """
        self.dataset_file = dataset_file
        self.min_length = min_length
        self.max_length = max_length
        self.preload = preload
        
        # Detect file format
        self.file_format = self._detect_format(dataset_file)
        print(f"Detected file format: {self.file_format}")
        
        if preload:
            # Preload all sequences into memory
            self.sequences = self._load_sequences(dataset_file)
            # Filter by length if specified
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
        else:
            # On-demand mode: just index the file
            self.sequence_offsets = self._index_file(dataset_file)
            # Filter by length if specified (requires reading sequences)
            if min_length:
                original_count = len(self.sequence_offsets)
                self.sequence_offsets = [
                    offset for offset in self.sequence_offsets
                    if len(self._read_sequence_at_offset(offset)['sequence']) >= min_length
                ]
                if original_count > len(self.sequence_offsets):
                    print(f"Filtered {original_count - len(self.sequence_offsets)} sequences shorter than {min_length}")
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
        min_length: Optional[int] = None,
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
        self.min_length = min_length
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
        full_dataset = SequenceDataset(self.dataset_path, self.min_length, self.max_length, preload=self.preload_sequences)
        
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
        
        # Move to GPU and set to eval mode
        # In DDP, use the current process's device (cuda:0 for rank 0, cuda:1 for rank 1, etc.)
        if torch.cuda.is_available():
            # Get device from LOCAL_RANK environment variable (set by DDP)
            # Fall back to current_device() if LOCAL_RANK not set
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            device = torch.device(f'cuda:{local_rank}')
        else:
            device = torch.device('cpu')
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
        
        # Add placeholder ground truth features for sequence-only training
        # These are required by some data transforms but won't affect representation extraction
        num_res = len(sequence)
        feature_dict['all_atom_positions'] = np.zeros((num_res, 37, 3), dtype=np.float32)
        feature_dict['all_atom_mask'] = np.ones((num_res, 37), dtype=np.float32)  # Assume all atoms present
        feature_dict['resolution'] = np.array([0.0], dtype=np.float32)  # Placeholder resolution
        feature_dict['is_distillation'] = np.array(0.0, dtype=np.float32)  # Not distillation data
        feature_dict['release_date'] = np.array(['2025-01-01'.encode('utf-8')], dtype=object)  # Placeholder date
        
        # Cleanup
        os.remove(tmp_fasta_path)
        shutil.rmtree(temp_alignment_dir)
        
        return feature_dict
    
    def _generate_target_representations_from_processed(
        self, 
        processed_batch: Dict[str, torch.Tensor],
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate target representations using teacher model from already-processed features.
        
        For representation distillation, we always use 0 recycles for efficiency
        since we only need the Evoformer outputs, not structure predictions.
        
        Args:
            processed_batch: Already processed features (as tensors on device)
            device: Device to run on
        
        Returns:
            target_s: Single representation [batch, N, C_s]
            target_z: Pair representation [batch, N, N, C_z]
        """
        # Get teacher for current worker/process
        teacher_model, _, _ = self._get_or_create_teacher_components()
        
        # Make a copy to avoid modifying the student batch
        teacher_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v 
                        for k, v in processed_batch.items()}
        
        # Add flag to request representations only (skip structure module)
        teacher_batch['return_representations'] = True
        
        # Run teacher model (no gradients)
        with torch.no_grad():
            outputs = teacher_model(teacher_batch)
        
        # Extract single (s) and pair (z) representations
        # These are after the Evoformer stack
        # Shapes: [batch, N, C_s] and [batch, N, N, C_z]
        target_s = outputs['single']
        target_z = outputs['pair']
        
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
        
        # Determine device - use current device for DDP compatibility
        if torch.cuda.is_available():
            # Get device from LOCAL_RANK environment variable (set by DDP)
            # Fall back to 0 if LOCAL_RANK not set (single GPU case)
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            device = torch.device(f'cuda:{local_rank}')
        else:
            device = torch.device('cpu')
        
        # Process features once for both teacher and student
        # IMPORTANT: Use same mode to ensure identical feature processing
        _, feature_processor, _ = self._get_or_create_teacher_components()
        processed_features = feature_processor.process_features(
            feature_dict, mode='train', is_multimer=False
        )
        
        # Convert to tensors
        processed_batch = {
            k: torch.as_tensor(v, device=device)
            for k, v in processed_features.items()
        }
        # Prepare student batch (copy of processed features)
        student_batch = processed_batch.copy()
                
        # Generate target representations using teacher model
        # Teacher uses the same processed features as student
        target_s, target_z = self._generate_target_representations_from_processed(
            processed_batch, device
        )
        
        # Add flag to request representations only (skip structure module)
        student_batch['return_representations'] = True
        
        # Create proper sequence mask based on actual sequence length
        # The features might be padded to max_length, but we only want loss on actual sequence
        actual_seq_len = len(sequence)
        if 'seq_mask' in student_batch:
            # Get the padded sequence length from the feature shape
            padded_len = student_batch['seq_mask'].shape[-1]  # Last dim after recycling
            if actual_seq_len < padded_len:
                # Create mask: 1 for actual sequence, 0 for padding
                # Shape should match seq_mask: [..., N] where last dim is sequence length
                mask = torch.zeros_like(student_batch['seq_mask'])
                mask[..., :actual_seq_len] = 1.0
                student_batch['seq_mask'] = mask
        
        # Return batch dict with targets
        return {
            'student_batch': student_batch,
            'target_s': target_s,
            'target_z': target_z,
            'seq_id': seq_id,
            'seq_length': actual_seq_len,
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
        adaptive_config_path: str = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        warmup_steps: int = 1000,
        replace_loss_scaler: float = 0.0,
        weights_path: str = None,
        resume_adaptive_checkpoint: str = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.config = config
        self.adaptive_config_path = adaptive_config_path
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.replace_loss_scaler = replace_loss_scaler
        self.weights_path = weights_path
        self.resume_adaptive_checkpoint = resume_adaptive_checkpoint
        
        # Create base student model
        self.student_model = AlphaFold(config)
        
        # Determine initialization path
        # Priority: resume_adaptive_checkpoint > weights_path + adaptive_config_path
        if resume_adaptive_checkpoint:
            # Path 2: Resume from checkpoint with adaptive blocks already applied
            print(f"\n{'='*80}")
            print(f"Resuming from adaptive checkpoint: {resume_adaptive_checkpoint}")
            print(f"{'='*80}")
            
            # Apply adaptive blocks (structure only, weights loaded by Lightning)
            if not adaptive_config_path:
                raise ValueError("adaptive_config_path is required when resuming from checkpoint")
            
            self.student_model, training_info = setup_adaptive_training_model(
                model=self.student_model,
                config_path=Path(adaptive_config_path),
                model_config=self.config,
            )
            
            # Freeze all except adaptive components
            trainable_params = freeze_model_except_adaptive_components(self.student_model)
            total_params = sum(p.numel() for p in self.student_model.parameters())
            
            print(f"Adaptive blocks: {len(training_info['weight_predictors'])}")
            print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")
            print(f"Note: Weights will be loaded from checkpoint by Lightning")
            print(f"{'='*80}\n")
            
            self.adaptive_setup_done = True
            
        elif weights_path and adaptive_config_path:
            # Path 1: Initial training from base weights
            print(f"\n{'='*80}")
            print(f"Initial training: Loading base weights and applying adaptive blocks")
            print(f"{'='*80}")
            
            # Load base weights first
            self._load_base_weights(weights_path)
            
            # Apply adaptive blocks immediately
            self.student_model, training_info = setup_adaptive_training_model(
                model=self.student_model,
                config_path=Path(adaptive_config_path),
                model_config=self.config,
            )
            
            # Freeze all except adaptive components
            trainable_params = freeze_model_except_adaptive_components(self.student_model)
            total_params = sum(p.numel() for p in self.student_model.parameters())
            
            print(f"Adaptive blocks: {len(training_info['weight_predictors'])}")
            print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")
            print(f"{'='*80}\n")
            
            self.adaptive_setup_done = True
            
        else:
            raise ValueError(
                "Must provide either:\n"
                "  1. --resume_adaptive_checkpoint (for resuming training), OR\n"
                "  2. --weights_path AND --adaptive_config_path (for initial training)"
            )
    
    def _load_base_weights(self, weights_path: str):
        """Load base model weights (JAX or PyTorch)"""
        print(f"Loading base weights: {weights_path}")
        
        if weights_path.endswith('.npz'):
            # JAX weights
            model_basename = Path(weights_path).stem
            model_version = "_".join(model_basename.split("_")[1:])
            import_jax_weights_(self.student_model, weights_path, version=model_version)
        else:
            # PyTorch weights
            checkpoint = torch.load(weights_path, map_location='cpu')
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            
            # Remove 'model.' prefix if present
            if any(k.startswith('model.') for k in state_dict.keys()):
                state_dict = {k[6:]: v for k, v in state_dict.items() if k.startswith('model.')}
            
            self.student_model.load_state_dict(state_dict, strict=False)
        
        print("Base weights loaded successfully")
    
    def forward(self, batch):
        """Forward pass through student model"""
        return self.student_model(batch)
    
    def _compute_loss(self, pred_s, pred_z, target_s, target_z, seq_mask=None):
        """
        Compute MSE loss between predicted and target representations.
        
        Args:
            pred_s: Predicted single representation [batch, N, C_s] or [N, C_s]
            pred_z: Predicted pair representation [batch, N, N, C_z] or [N, N, C_z]
            target_s: Target single representation [batch, N, C_s] or [N, C_s]
            target_z: Target pair representation [batch, N, N, C_z] or [N, N, C_z]
            seq_mask: Optional mask for valid positions
                     Can be [N], [N, recycle], [batch, N], or [batch, N, recycle]
        """
        # Ensure both have batch dimension
        if pred_s.ndim == 2:
            pred_s = pred_s.unsqueeze(0)
        if target_s.ndim == 2:
            target_s = target_s.unsqueeze(0)
        if pred_z.ndim == 3:
            pred_z = pred_z.unsqueeze(0)
        if target_z.ndim == 3:
            target_z = target_z.unsqueeze(0)
        
        # MSE loss on single representation with optional masking
        if seq_mask is not None:
            # Handle seq_mask which might have recycling dimension
            # seq_mask could be: [N], [N, recycle], [batch, N], or [batch, N, recycle]
            
            # First, handle recycling dimension if present
            if seq_mask.ndim >= 2 and seq_mask.shape[-1] <= 10:  # Heuristic: recycling dim is small
                if seq_mask.ndim == 2 and seq_mask.shape[0] > seq_mask.shape[1]:
                    # [N, recycle_dim] -> [N]
                    seq_mask = seq_mask[:, 0]
                elif seq_mask.ndim == 3:
                    # [batch, N, recycle_dim] -> [batch, N]
                    seq_mask = seq_mask[..., 0]
            
            # Now handle batch dimension
            if seq_mask.ndim == 1:
                seq_mask = seq_mask.unsqueeze(0)  # [N] -> [1, N]
            
            # Now seq_mask is [batch, N]
            mask_s = seq_mask.unsqueeze(-1)  # [batch, N, 1]
            
            # Compute masked MSE
            diff_s = (pred_s - target_s) ** 2  # [batch, N, C_s]
            masked_diff_s = diff_s * mask_s
            
            # Sum over all dimensions, then divide by (num_valid_positions * num_channels)
            num_valid_positions = mask_s.sum()
            num_channels = pred_s.shape[-1]
            loss_s = masked_diff_s.sum() / (num_valid_positions * num_channels + 1e-8)
        else:
            loss_s = F.mse_loss(pred_s, target_s)
        
        # MSE loss on pair representation with optional masking
        if seq_mask is not None:
            # seq_mask is already [batch, N] from above
            # Create pair mask: [batch, N, N]
            mask_z = seq_mask.unsqueeze(-1) * seq_mask.unsqueeze(-2)  # [batch, N, N]
            mask_z = mask_z.unsqueeze(-1)  # [batch, N, N, 1]
            
            # Compute masked MSE
            diff_z = (pred_z - target_z) ** 2  # [batch, N, N, C_z]
            masked_diff_z = diff_z * mask_z
            
            num_valid_pairs = mask_z.sum()
            num_channels_z = pred_z.shape[-1]
            loss_z = masked_diff_z.sum() / (num_valid_pairs * num_channels_z + 1e-8)
        else:
            loss_z = F.mse_loss(pred_z, target_z)
        
        # Total loss (simple sum, can adjust weights if needed)
        total_loss = loss_s + loss_z
        
        return total_loss, loss_s, loss_z
    
    def training_step(self, batch_dict, batch_idx):
        """Training step"""
        student_batch = batch_dict['student_batch']
        target_s = batch_dict['target_s']
        target_z = batch_dict['target_z']
        seq_length = batch_dict['seq_length']
        
        # Forward pass through student model
        outputs = self(student_batch)
        
        # Extract predicted representations
        pred_s = outputs['single']
        pred_z = outputs['pair']
        
        # Get sequence mask from batch (accounts for padding)
        seq_mask = student_batch.get('seq_mask', None)
        
        # Compute base loss (MSE on representations with masking)
        base_loss, loss_s, loss_z = self._compute_loss(
            pred_s, pred_z, target_s, target_z, seq_mask=seq_mask
        )
        
        # Add adaptive losses if applicable
        if self.adaptive_setup_done and self.replace_loss_scaler > 0:
            # Compute replace loss (penalizes mean adaptive weights)
            raw_replace_loss = compute_adaptive_replace_loss(
                model=self.student_model,
                replace_loss_scaler=1.0,  # Get raw loss
                device=base_loss.device,
            )
            
            # Compute block match loss (encourages replacement to match original)
            block_match_loss = compute_block_match_loss(
                model=self.student_model,
                device=base_loss.device,
            )
            
            # Scale and combine losses
            # Both replace_loss and block_match_loss are scaled by replace_loss_scaler
            scaled_replace_loss = raw_replace_loss * self.replace_loss_scaler
            scaled_block_match_loss = block_match_loss * self.replace_loss_scaler
            total_loss = base_loss + scaled_replace_loss + scaled_block_match_loss
            
            # Log losses
            self.log('train/replace_loss_raw', raw_replace_loss, on_step=True, on_epoch=True, batch_size=1)
            self.log('train/replace_loss_scaled', scaled_replace_loss, on_step=True, on_epoch=True, batch_size=1)
            self.log('train/block_match_loss_raw', block_match_loss, on_step=True, on_epoch=True, batch_size=1)
            self.log('train/block_match_loss_scaled', scaled_block_match_loss, on_step=True, on_epoch=True, batch_size=1)
        else:
            total_loss = base_loss
        
        # Log metrics
        self.log('train/loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=1)
        self.log('train/loss_single', loss_s, on_step=True, on_epoch=True, batch_size=1)
        self.log('train/loss_pair', loss_z, on_step=True, on_epoch=True, batch_size=1)
        self.log('train/seq_length', float(seq_length), on_step=False, on_epoch=True, batch_size=1)
        
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
        
        # Get sequence mask from batch (accounts for padding)
        seq_mask = student_batch.get('seq_mask', None)
        
        # Compute base loss (MSE on representations with masking)
        base_loss, loss_s, loss_z = self._compute_loss(pred_s, pred_z, target_s, target_z, seq_mask=seq_mask)
        
        # Add adaptive losses if applicable
        if self.adaptive_setup_done and self.replace_loss_scaler > 0:
            # Compute replace loss (penalizes mean adaptive weights)
            raw_replace_loss = compute_adaptive_replace_loss(
                model=self.student_model,
                replace_loss_scaler=1.0,  # Get raw loss
                device=base_loss.device,
            )
            
            # Compute block match loss (encourages replacement to match original)
            block_match_loss = compute_block_match_loss(
                model=self.student_model,
                device=base_loss.device,
            )
            
            # Scale and combine losses
            scaled_replace_loss = raw_replace_loss * self.replace_loss_scaler
            scaled_block_match_loss = block_match_loss * self.replace_loss_scaler
            total_loss = base_loss + scaled_replace_loss + scaled_block_match_loss
            
            # Log losses with explicit batch_size
            self.log('val/replace_loss_raw', raw_replace_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=1)
            self.log('val/replace_loss_scaled', scaled_replace_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=1)
            self.log('val/block_match_loss_raw', block_match_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=1)
            self.log('val/block_match_loss_scaled', scaled_block_match_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=1)
        else:
            total_loss = base_loss
        
        # Log metrics with explicit batch_size
        self.log('val/loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=1)
        self.log('val/loss_single', loss_s, on_step=False, on_epoch=True, sync_dist=True, batch_size=1)
        self.log('val/loss_pair', loss_z, on_step=False, on_epoch=True, sync_dist=True, batch_size=1)
        
        return total_loss
    
    def configure_optimizers(self):
        """Configure optimizer with warmup scheduler"""
        
        # Collect trainable parameters (adaptive components only after freezing)
        trainable_params = [p for p in self.student_model.parameters() if p.requires_grad]
        
        num_trainable = sum(p.numel() for p in trainable_params)
        total_params = sum(p.numel() for p in self.student_model.parameters())
        print(f"Configuring optimizer with {num_trainable:,} / {total_params:,} trainable parameters")
        
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
    parser.add_argument('--dataset_path', type=str, required=False,
                       help='Path to the dataset file containing sequences; can be a FASTA file or a TSV file')
    parser.add_argument('--min_length', type=int, default=50,
                       help='Minimum sequence length')
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
    parser.add_argument('--gpus', type=int,
                       help='Number of GPUs')
    parser.add_argument('--precision', type=str,
                       choices=['16', '32', 'bf16', 'bf16-mixed'],
                       help='Training precision')
    parser.add_argument('--strategy', type=str,
                       choices=['ddp', 'ddp_spawn', 'dp', None],
                       help='Distributed training strategy')
    parser.add_argument('--distributed_backend', type=str,
                       choices=['nccl', 'gloo', 'mpi'],
                       help='Distributed backend (nccl for GPU, gloo for CPU/compatibility)')
    
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
    
    # Checkpoint resuming
    parser.add_argument('--resume_adaptive_checkpoint', type=str, default=None,
                       help='Path to Lightning checkpoint (.ckpt) with adaptive blocks already applied. Takes priority over --weights_path.')
    
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
            # If command line arg was provided, it takes precedence (already set by argparse)
    
    # Convert relative paths to absolute (relative to home directory)
    home_dir = Path.home()
    path_args = ['dataset_path', 'weights_path', 'adaptive_config_path', 
                'trained_models_dir', 'output_dir', 'config', 'resume_adaptive_checkpoint']
    
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
    
    # Set defaults for required fields if not provided
    if not hasattr(args, 'gpus') or args.gpus is None:
        args.gpus = 1
    if not hasattr(args, 'precision') or args.precision is None:
        args.precision = 'bf16-mixed'
    if not hasattr(args, 'strategy') or args.strategy is None:
        args.strategy = 'ddp'
    
    # Note about data_loading_strategy
    # For representation distillation:
    # - 'on_demand': Load sequences from FASTA on-the-fly (slower, lower memory)
    # - 'preload_cpu' or 'preload_gpu': Preload all sequences to memory (faster, higher memory)
    # Targets (representations) are always generated on-the-fly by teacher model
    if hasattr(args, 'data_loading_strategy'):
        print(f"Data loading strategy: {args.data_loading_strategy}")
        print("  - Sequences: {'preloaded to memory' if args.data_loading_strategy in ['preload_cpu', 'preload_gpu'] else 'loaded on-demand'}")
        print("  - Targets (representations): Always generated on-the-fly by teacher model")
    
    # Validate required arguments based on training mode
    if not args.dataset_path:
        raise ValueError("--dataset_path is required (or provide via --config)")
    if not args.output_dir:
        raise ValueError("--output_dir is required (or provide via --config)")
    
    # Two paths: resume from checkpoint OR initial training
    if args.resume_adaptive_checkpoint:
        # Path 2: Resuming from checkpoint
        print(f"\n{'='*60}")
        print("Mode: Resuming from adaptive checkpoint")
        print(f"Checkpoint: {args.resume_adaptive_checkpoint}")
        print(f"{'='*60}\n")
        
        # Still need adaptive_config_path to know the model structure
        if not hasattr(args, 'adaptive_config_path') or not args.adaptive_config_path:
            if not hasattr(args, 'trained_models_dir') or not args.trained_models_dir:
                raise ValueError("When resuming, either --adaptive_config_path OR --trained_models_dir is required")
            if not hasattr(args, 'linear_type') or not args.linear_type:
                raise ValueError("--linear_type is required when trained_models_dir is provided")
            
            # Create adaptive training config file
            adaptive_config_data = {
                "trained_models_dir": args.trained_models_dir,
                "linear_type": args.linear_type,
                "replace_loss_scaler": getattr(args, 'replace_loss_scaler', 0.0),
                "log_structure_every_k_epoch": 0,
                "disable_per_block_logging": True,
                "adaptive_training": True
            }
            
            os.makedirs(args.output_dir, exist_ok=True)
            args.adaptive_config_path = os.path.join(args.output_dir, 'adaptive_training_cmd.json')
            with open(args.adaptive_config_path, 'w') as f:
                json.dump(adaptive_config_data, f, indent=2)
            print(f"Created adaptive config file: {args.adaptive_config_path}")
        
    else:
        # Path 1: Initial training from base weights
        print(f"\n{'='*60}")
        print("Mode: Initial training from base weights")
        print(f"{'='*60}\n")
        
        if not args.weights_path:
            raise ValueError("--weights_path is required for initial training (or provide via --config)")
        
        # Create adaptive_config_path if not provided
        if not hasattr(args, 'adaptive_config_path') or not args.adaptive_config_path:
            if not hasattr(args, 'trained_models_dir') or not args.trained_models_dir:
                raise ValueError("Either --adaptive_config_path OR --trained_models_dir is required")
            if not hasattr(args, 'linear_type') or not args.linear_type:
                raise ValueError("--linear_type is required when trained_models_dir is provided")
            
            # Create adaptive training config file
            adaptive_config_data = {
                "trained_models_dir": args.trained_models_dir,
                "linear_type": args.linear_type,
                "replace_loss_scaler": getattr(args, 'replace_loss_scaler', 0.0),
                "log_structure_every_k_epoch": 0,
                "disable_per_block_logging": True,
                "adaptive_training": True
            }
            
            os.makedirs(args.output_dir, exist_ok=True)
            args.adaptive_config_path = os.path.join(args.output_dir, 'adaptive_training_cmd.json')
            with open(args.adaptive_config_path, 'w') as f:
                json.dump(adaptive_config_data, f, indent=2)
            print(f"Created adaptive config file: {args.adaptive_config_path}")
    
    # Set float32 matmul precision for Tensor Cores (addresses warning about bf16-mixed)
    # This is recommended when using bf16-mixed precision
    if args.precision in ["bf16-mixed", "bf16"]:
        torch.set_float32_matmul_precision('medium')  # or 'high' for more performance
        print("Set torch.set_float32_matmul_precision('medium') for Tensor Cores")
    
    # Set seed
    pl.seed_everything(args.seed, workers=True)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*80)
    print("REPRESENTATION DISTILLATION TRAINING")
    print("="*80)
    print(f"dataset path: {args.dataset_path}")
    print(f"Min length: {args.min_length}")
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
        min_length=args.min_length,
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
        weights_path=args.weights_path,
        resume_adaptive_checkpoint=args.resume_adaptive_checkpoint,
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
    
    # Learning rate monitor (only if logger is configured)
    if args.wandb:
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
    
    # IMPORTANT: Force num_workers=0 to avoid CUDA forking issues with teacher model
    # The teacher model needs to run on GPU, but dataloader workers + CUDA + fork don't work together
    if args.num_workers > 0:
        print(f"WARNING: Forcing num_workers=0 (was {args.num_workers}) to avoid CUDA forking issues")
        print("         Teacher model must run on GPU, which is incompatible with forked workers")
        args.num_workers = 0
    
    # Setup strategy for DDP with proper backend
    if args.gpus > 1:
        # Get distributed backend from config (defaults to nccl for GPU)
        distributed_backend = getattr(args, 'distributed_backend', 'nccl')
        if args.strategy == 'ddp':
            strategy = DDPStrategy(process_group_backend=distributed_backend)
        else:
            strategy = args.strategy
    else:
        # Single GPU: Force 'auto' strategy to avoid unnecessary DDP overhead
        # DDP with 1 GPU spawns extra worker processes that waste memory
        if args.strategy in ['ddp', 'ddp_spawn']:
            print(f"WARNING: Changing strategy from '{args.strategy}' to 'auto' for single GPU")
            print("         DDP with 1 GPU creates unnecessary worker processes")
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
        deterministic=True,
    )
    
    # Train
    print("\nStarting training...")
    trainer.fit(
        model, 
        datamodule=datamodule,
        ckpt_path=args.resume_adaptive_checkpoint if args.resume_adaptive_checkpoint else None
    )
    
    print("\n" + "="*80)
    print("Training complete!")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print("="*80)


if __name__ == '__main__':
    main()
