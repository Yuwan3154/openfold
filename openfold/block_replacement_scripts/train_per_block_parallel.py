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
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from openfold.block_replacement_scripts import _torch_pytree_compat  # noqa: F401
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
from openfold.block_replacement_scripts.dilated_conv_evoformer_replacement import (
    DilatedConvEvoformerReplacement,
)
from openfold.block_replacement_scripts.block_data_io import (
    load_block_sample,
    load_merged_block_samples,
    sanitize_id,
    save_block_sample,
)

MIN_BLOCK_DATA_CACHE_FREE_BYTES = 100 * 1024**3  # 100 GiB


def get_free_bytes_on_filesystem(path: Path) -> int:
    """
    Return free bytes on the filesystem hosting `path`.
    If `path` doesn't exist, walk up to the nearest existing parent.
    """
    p = path
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(str(p)).free


class TeacherModelWithBlockCapture:
    """
    Wrapper for teacher AlphaFold model that captures intermediate block outputs.
    
    Uses forward hooks to capture inputs and outputs of each Evoformer block during
    the forward pass. All captured tensors are detached to prevent gradient accumulation.
    """
    
    def __init__(
        self,
        model: AlphaFold,
        device: torch.device,
    ):
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
        self.forward_calls = 0
        
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
        self.forward_calls += 1
        
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
        block_data_dir: Optional[str] = None,
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
        self.block_data_dir = block_data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.min_length = min_length
        self.max_length = max_length
        self.validation_fraction = validation_fraction
        self.split_seed = split_seed
        
        # Setup data and feature processors only in teacher mode.
        # Cache-only mode does not require feature processing.
        if self.block_data_dir is None:
            self.data_processor, self.feature_processor = self._setup_data_pipeline()
        else:
            self.data_processor = None
            self.feature_processor = None
    
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
        
        # Cache-only batching: collate ids + lengths and build padding masks.
        if self.block_data_dir is not None:
            seq_ids = [s["id"] for s in batch]
            seq_lens = [len(s["sequence"]) for s in batch]
            bsz = len(seq_lens)
            if bsz == 0:
                raise ValueError("Empty batch")
            n_max = self.max_length if self.max_length is not None else max(seq_lens)
            seq_mask = torch.zeros((bsz, n_max), dtype=torch.float32)
            for i, n in enumerate(seq_lens):
                if n > n_max:
                    n = n_max
                    seq_lens[i] = n
                if n > 0:
                    seq_mask[i, :n] = 1.0
            msa_mask = seq_mask.unsqueeze(1)  # [B, 1, Nmax]
            return {
                "seq_id": seq_ids,
                "seq_length": torch.tensor(seq_lens, dtype=torch.long),
                "seq_mask": seq_mask,
                "msa_mask": msa_mask,
            }
        
        # Teacher mode: batch_size must be 1 (single-sample pipeline).
        if self.batch_size != 1 or len(batch) != 1:
            raise ValueError(
                f"Teacher mode requires batch_size=1, got batch_size={self.batch_size} len(batch)={len(batch)}"
            )

        seq_data = batch[0]
        seq_id = seq_data["id"]
        sequence = seq_data["sequence"]
        
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
            persistent_workers=False,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            persistent_workers=False,
        )
    
    def __del__(self):
        """Cleanup temporary directory"""
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
        block_data_dir: Optional[str] = None,
        block_data_format: str = "pt",
        block_data_save_dtype: str = "bf16",
        block_data_quantization: str = "none",
        replacement_type: str = "linear",
        linear_type: str = "full",
        kernel_size: int = 3,
        dilations: str = "1,2,4",
        dilation_pattern: Optional[str] = None,
        dilation_repeats: int = 1,
        replacement_mode: str = "per_block",
        replacement_checkpoint_subdir: Optional[str] = None,
        allow_random_init: bool = False,
        compile_replacement: bool = False,
        resume_checkpoint_path: Optional[str] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.config = config
        self.config_preset = config_preset
        self.weights_path = weights_path
        self.trained_models_dir = trained_models_dir
        self.block_data_dir = block_data_dir
        self.block_data_format = str(block_data_format)
        self.block_data_save_dtype = str(block_data_save_dtype)
        self.block_data_quantization = str(block_data_quantization)
        self.replacement_type = replacement_type
        self.linear_type = linear_type
        self.kernel_size = int(kernel_size)
        if dilation_pattern is None:
            dilation_pattern_vals = tuple(
                int(d) for d in str(dilations).split(",") if str(d).strip() != ""
            )
        else:
            dilation_pattern_vals = tuple(
                int(d) for d in str(dilation_pattern).split(",") if str(d).strip() != ""
            )
        self.dilation_pattern = dilation_pattern_vals
        self.dilation_repeats = int(dilation_repeats)
        self.dilations = tuple(self.dilation_pattern) * self.dilation_repeats
        self.replacement_mode = replacement_mode
        self.replacement_checkpoint_subdir = replacement_checkpoint_subdir
        self.allow_random_init = bool(allow_random_init)
        self.compile_replacement = bool(compile_replacement)
        self.resume_checkpoint_path = resume_checkpoint_path
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Get dimensions from config
        self.c_m = config.model.evoformer_stack.c_m
        self.c_z = config.model.evoformer_stack.c_z
        
        # Create teacher model with block capture
        self.teacher = None
        self._teacher_model = None
        if self.block_data_dir is None:
            print("Loading teacher model...", flush=True)
            self._teacher_model = self._load_teacher_model()
        else:
            print(
                "Cache-only mode enabled (block_data_dir is set). Teacher model will NOT be initialized.",
                flush=True,
            )
        
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
        
        # Ensure teacher is never trained
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        
        return model
    
    def _load_replacement_blocks(self):
        """Load all pretrained replacement blocks from disk"""
        blocks_loaded_from_ckpt = 0
        blocks_random_init = 0
        blocks_failed = 0

        if self.resume_checkpoint_path is not None:
            # When resuming, Lightning will restore weights from the run checkpoint.
            # Avoid unnecessary I/O and confusing "checkpoint not found" warnings.
            for block_idx in range(0, 48):
                if self.replacement_type == "linear":
                    replacement_block = SimpleEvoformerReplacement(
                        c_m=self.c_m,
                        c_z=self.c_z,
                        linear_type=self.linear_type,
                    )
                else:
                    replacement_block = DilatedConvEvoformerReplacement(
                        c_m=self.c_m,
                        c_z=self.c_z,
                        kernel_size=self.kernel_size,
                        dilations=self.dilations,
                        dilation_pattern=self.dilation_pattern,
                        dilation_repeats=self.dilation_repeats,
                        mode=self.replacement_mode,
                    )
                self.replacement_blocks[str(block_idx)] = replacement_block
                blocks_random_init += 1

            print(
                f"Resuming from {self.resume_checkpoint_path}; "
                "initialized replacement blocks for checkpoint restore.",
                flush=True,
            )
            print(
                f"Replacement blocks: loaded_from_ckpt={blocks_loaded_from_ckpt}, "
                f"random_init={blocks_random_init}, missing={blocks_failed}",
                flush=True,
            )
            return

        if self.replacement_checkpoint_subdir is not None:
            checkpoint_subdir = self.replacement_checkpoint_subdir
        elif self.replacement_type == "linear":
            checkpoint_subdir = self.linear_type
        elif self.replacement_type == "conv":
            d_str = "-".join(str(d) for d in self.dilations)
            checkpoint_subdir = f"conv_{self.replacement_mode}_k{self.kernel_size}_d{d_str}"
        else:
            raise ValueError(
                f"Invalid replacement_type: {self.replacement_type}. Expected 'linear' or 'conv'."
            )
        
        for block_idx in range(0, 48):
            block_path = (
                Path(self.trained_models_dir)
                / f"block_{block_idx:02d}"
                / checkpoint_subdir
                / "best_model.ckpt"
            )
            
            if not block_path.exists():
                print(f"Warning: Block {block_idx} checkpoint not found at {block_path}")
                if not self.allow_random_init:
                    blocks_failed += 1
                    continue
            
            # Create replacement block
            if self.replacement_type == "linear":
                replacement_block = SimpleEvoformerReplacement(
                    c_m=self.c_m,
                    c_z=self.c_z,
                    linear_type=self.linear_type,
                )
            else:
                replacement_block = DilatedConvEvoformerReplacement(
                    c_m=self.c_m,
                    c_z=self.c_z,
                    kernel_size=self.kernel_size,
                    dilations=self.dilations,
                    dilation_pattern=self.dilation_pattern,
                    dilation_repeats=self.dilation_repeats,
                    mode=self.replacement_mode,
                )

            # Load checkpoint if present
            if block_path.exists():
                checkpoint = torch.load(block_path, map_location="cpu")
                if "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                else:
                    state_dict = checkpoint

                # Strip 'replacement_block.' prefix if present
                if any(k.startswith("replacement_block.") for k in state_dict.keys()):
                    state_dict = {
                        k.replace("replacement_block.", ""): v for k, v in state_dict.items()
                    }

                replacement_block.load_state_dict(state_dict, strict=True)
                blocks_loaded_from_ckpt += 1
            else:
                blocks_random_init += 1
            
            # Store in ModuleDict
            self.replacement_blocks[str(block_idx)] = replacement_block
        
        total_blocks = blocks_loaded_from_ckpt + blocks_random_init
        print(
            f"Replacement blocks: loaded_from_ckpt={blocks_loaded_from_ckpt}, "
            f"random_init={blocks_random_init}, missing={blocks_failed}"
        )
        
        if total_blocks == 0:
            raise ValueError(f"No replacement blocks loaded from {self.trained_models_dir}")
    
    def setup(self, stage: Optional[str] = None):
        """Setup teacher model on correct device"""
        # If caching is enabled, initialize teacher lazily only when we hit a cache miss.
        if self.teacher is None and self.block_data_dir is None:
            if self._teacher_model is None:
                self._teacher_model = self._load_teacher_model()
            device = self.device if hasattr(self, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.teacher = TeacherModelWithBlockCapture(
                self._teacher_model,
                device,
            )
            print(f"Teacher model initialized on {device}")

    def on_fit_start(self):
        # In cache-only mode, ensure teacher is never initialized.
        if self.block_data_dir is not None:
            if self.teacher is not None or self._teacher_model is not None:
                raise RuntimeError("Cache-only mode forbids teacher initialization, but teacher is present")

            # Preflight: verify all required cache files exist before training.
            # Do this on rank 0 and broadcast outcome to avoid DDP hangs.
            is_dist = torch.distributed.is_available() and torch.distributed.is_initialized()
            global_rank = int(getattr(self, "global_rank", 0))
            is_rank0 = (global_rank == 0)

            ok = True
            missing: list[str] = []
            if is_rank0:
                dm = getattr(self.trainer, "datamodule", None)
                if dm is None:
                    raise RuntimeError("Cache-only mode requires a datamodule for preflight cache validation")

                cache_dir = Path(self.block_data_dir)
                ext = self.block_data_format
                blocks_to_train = sorted(int(k) for k in self.replacement_blocks.keys())

                datasets = []
                if hasattr(dm, "train_dataset") and dm.train_dataset is not None:
                    datasets.append(("train", dm.train_dataset))
                if hasattr(dm, "val_dataset") and dm.val_dataset is not None:
                    datasets.append(("val", dm.val_dataset))

                is_merged = ext == "merged.safetensors"
                for split_name, ds in datasets:
                    for item in ds:
                        seq_id = item["id"]
                        sid = sanitize_id(str(seq_id))
                        if is_merged:
                            cache_path = cache_dir / f"{sid}.safetensors"
                            if not cache_path.exists():
                                ok = False
                                if len(missing) < 20:
                                    missing.append(f"{split_name}: {cache_path}")
                        else:
                            for block_idx in blocks_to_train:
                                cache_path = cache_dir / f"block_{block_idx:02d}" / f"{sid}.{ext}"
                                if not cache_path.exists():
                                    ok = False
                                    if len(missing) < 20:
                                        missing.append(f"{split_name}: {cache_path}")

            ok_tensor = torch.tensor([1 if ok else 0], dtype=torch.int32)
            if is_dist:
                torch.distributed.broadcast(ok_tensor, src=0)

            if int(ok_tensor.item()) == 0:
                if is_rank0:
                    msg = "Cache preflight failed; missing cache files (first 20):\n" + "\n".join(missing)
                    raise FileNotFoundError(msg)
                raise RuntimeError("Cache preflight failed on rank0; see rank0 logs for missing cache paths")

        if self.compile_replacement and self.device.type == "cuda":
            import torch._dynamo as dynamo

            dynamo.config.suppress_errors = True
            for block_idx in list(self.replacement_blocks.keys()):
                self.replacement_blocks[block_idx] = torch.compile(
                    self.replacement_blocks[block_idx],
                    mode="reduce-overhead",
                    dynamic=True,
                )
            print(
                f"Compiled {len(self.replacement_blocks)} replacement blocks with mode=reduce-overhead dynamic=True",
                flush=True,
            )
    
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
        log_batch_size = 1
        
        # Extract metadata
        seq_length = batch.pop("seq_length")
        seq_id = batch.pop("seq_id")
        cache_dir = None
        safe_seq_id = None
        if self.block_data_dir:
            cache_dir = Path(self.block_data_dir)
            if isinstance(seq_id, (list, tuple)):
                safe_seq_id = [sanitize_id(str(s)) for s in seq_id]
            else:
                safe_seq_id = sanitize_id(str(seq_id))
        cache_can_write = None
        
        # Get masks from batch
        # msa_mask should be [batch, n_seq, n_res]
        msa_mask = batch.get('msa_mask', None)
        # seq_mask is [batch, n_res], need to expand to pair_mask [batch, n_res, n_res]
        seq_mask = batch.get('seq_mask', None)
        if seq_mask is not None:
            pair_mask = seq_mask.unsqueeze(-1) * seq_mask.unsqueeze(-2)
        else:
            pair_mask = None
        
        # Run teacher model lazily only if we have a cache miss
        block_data = None
        teacher_used = False
        cache_hits = 0
        cache_misses = 0

        # For merged format, load all blocks for each sample once before the loop.
        # Use map_location="cpu" to leverage safetensors mmap (instant open);
        # the existing .to(device, non_blocking=True) calls in the block loop
        # handle GPU transfer more efficiently than loading directly to GPU.
        merged_all = None  # sid -> dict[block_idx] -> {input: ..., output: ...}
        is_merged = cache_dir is not None and self.block_data_format == "merged.safetensors"
        if is_merged and safe_seq_id is not None:
            merged_all = {}
            sids = safe_seq_id if isinstance(safe_seq_id, list) else [safe_seq_id]
            for sid in sids:
                merged_path = cache_dir / f"{sid}.safetensors"
                if not merged_path.exists():
                    raise FileNotFoundError(f"Missing merged cache file: {merged_path}")
                merged_all[sid] = load_merged_block_samples(merged_path, map_location="cpu")
        
        # Train each replacement block sequentially
        total_loss = 0.0
        num_blocks = 0
        
        for block_idx in range(0, 48):
            if str(block_idx) not in self.replacement_blocks:
                continue
            
            cached = None
            if merged_all is not None:
                # Merged format: look up from pre-loaded dict
                if isinstance(safe_seq_id, list):
                    cached = [merged_all[sid].get(block_idx) for sid in safe_seq_id]
                    if any(c is None for c in cached):
                        raise FileNotFoundError(f"Block {block_idx} missing in merged cache for some samples")
                else:
                    cached = merged_all[safe_seq_id].get(block_idx)
                    if cached is None:
                        raise FileNotFoundError(f"Block {block_idx} missing in merged cache for {safe_seq_id}")
            elif cache_dir is not None and safe_seq_id is not None:
                ext = self.block_data_format
                cache_map_location = self.device if self.block_data_format == "df11.safetensors" else "cpu"
                if isinstance(safe_seq_id, list):
                    cached = []
                    for sid in safe_seq_id:
                        cache_path = cache_dir / f"block_{block_idx:02d}" / f"{sid}.{ext}"
                        if not cache_path.exists():
                            raise FileNotFoundError(f"Missing cache file: {cache_path}")
                        cached.append(load_block_sample(cache_path, map_location=cache_map_location))
                else:
                    cache_path = cache_dir / f"block_{block_idx:02d}" / f"{safe_seq_id}.{ext}"
                    if not cache_path.exists():
                        raise FileNotFoundError(f"Missing cache file: {cache_path}")
                    cached = load_block_sample(cache_path, map_location=cache_map_location)

            if cached is not None:
                cache_hits += 1
            else:
                # Teacher path (single-sample only when block_data_dir is not set)
                cache_misses += 1
                if self.block_data_dir is not None:
                    raise RuntimeError("Cache-only mode: unexpected missing cached data")

                if block_data is None:
                    teacher_used = True
                    if self.teacher is None:
                        if self._teacher_model is None:
                            print("Loading teacher model...", flush=True)
                            self._teacher_model = self._load_teacher_model()
                        self.teacher = TeacherModelWithBlockCapture(self._teacher_model, self.device)
                        print(f"Teacher model initialized on {self.device}")
                    block_data = self.teacher.forward(batch)
                if block_idx not in block_data:
                    continue

                # Get block input/output from teacher (already single representation)
                m_in_single, z_in = block_data[block_idx]["input"]
                m_target_single, z_target = block_data[block_idx]["output"]

                # Cache to disk if enabled and not already present
                if cache_dir is not None and safe_seq_id is not None and not isinstance(safe_seq_id, list):
                    ext = self.block_data_format
                    cache_path = cache_dir / f"block_{block_idx:02d}" / f"{safe_seq_id}.{ext}"
                    if not cache_path.exists():
                        if cache_can_write is None:
                            free_bytes = get_free_bytes_on_filesystem(cache_dir)
                            cache_can_write = free_bytes >= MIN_BLOCK_DATA_CACHE_FREE_BYTES
                        if not cache_can_write:
                            continue

                        sample = {
                            "input": {"m": m_in_single.detach().cpu(), "z": z_in.detach().cpu()},
                            "output": {"m": m_target_single.detach().cpu(), "z": z_target.detach().cpu()},
                            "chain_id": str(seq_id),
                            "block_idx": int(block_idx),
                        }
                        tmp_tag = f".tmp.{os.getpid()}.{int(getattr(self, 'global_rank', 0))}"
                        suffix = f".{self.block_data_format}"
                        base_name = (
                            cache_path.name[: -len(suffix)] if cache_path.name.endswith(suffix) else cache_path.name
                        )
                        tmp_path = cache_path.with_name(base_name + tmp_tag + suffix)
                        save_block_sample(
                            sample,
                            tmp_path,
                            save_dtype=self.block_data_save_dtype,  # type: ignore[arg-type]
                            quantization=self.block_data_quantization,  # type: ignore[arg-type]
                        )
                        os.replace(tmp_path, cache_path)

            if isinstance(cached, list):
                # Cache-only batched mode: pad to Nmax and run once.
                if not isinstance(seq_length, torch.Tensor) or seq_length.ndim != 1:
                    raise ValueError("Cache-only mode expects seq_length as 1D tensor")
                bsz = int(seq_length.shape[0])
                log_batch_size = bsz
                if seq_mask is None:
                    raise ValueError("Cache-only mode requires seq_mask in batch")
                n_max = int(seq_mask.shape[1])

                m_list = []
                z_list = []
                mt_list = []
                zt_list = []
                for c in cached:
                    m = c["input"]["m"]
                    z = c["input"]["z"]
                    mt = c["output"]["m"]
                    zt = c["output"]["z"]
                    if m.dim() == 3 and m.shape[0] == 1:
                        m = m.squeeze(0)
                    if z.dim() == 4 and z.shape[0] == 1:
                        z = z.squeeze(0)
                    if mt.dim() == 3 and mt.shape[0] == 1:
                        mt = mt.squeeze(0)
                    if zt.dim() == 4 and zt.shape[0] == 1:
                        zt = zt.squeeze(0)
                    m_list.append(m)
                    z_list.append(z)
                    mt_list.append(mt)
                    zt_list.append(zt)

                c_m = int(m_list[0].shape[-1])
                c_z = int(z_list[0].shape[-1])
                m_in_padded = torch.zeros((bsz, n_max, c_m), dtype=m_list[0].dtype, device=self.device)
                m_tgt_padded = torch.zeros((bsz, n_max, c_m), dtype=mt_list[0].dtype, device=self.device)
                z_in_padded = torch.zeros((bsz, n_max, n_max, c_z), dtype=z_list[0].dtype, device=self.device)
                z_tgt_padded = torch.zeros((bsz, n_max, n_max, c_z), dtype=zt_list[0].dtype, device=self.device)

                seq_mask = seq_mask.clone()

                for i in range(bsz):
                    n_i = int(seq_length[i].item())
                    if n_i > n_max:
                        n_i = n_max
                        seq_length[i] = n_i
                        seq_mask[i, n_i:] = 0.0
                    n_cache = int(m_list[i].shape[0])
                    if z_list[i].dim() >= 2:
                        n_cache = min(n_cache, int(z_list[i].shape[0]), int(z_list[i].shape[1]))
                    if n_i > n_cache:
                        n_i = n_cache
                        seq_length[i] = n_i
                        seq_mask[i, n_i:] = 0.0
                    if n_i == 0:
                        continue
                    m_in_padded[i, :n_i, :] = m_list[i][:n_i]
                    m_tgt_padded[i, :n_i, :] = mt_list[i][:n_i]
                    z_in_padded[i, :n_i, :n_i, :] = z_list[i][:n_i, :n_i]
                    z_tgt_padded[i, :n_i, :n_i, :] = zt_list[i][:n_i, :n_i]

                msa_mask = seq_mask.unsqueeze(1)
                pair_mask = seq_mask.unsqueeze(-1) * seq_mask.unsqueeze(-2)

                m_in_msa = m_in_padded.unsqueeze(1)
                z_in_for_block = z_in_padded
                m_target_single = m_tgt_padded
                z_target = z_tgt_padded
                needs_batch_dim = False
            elif cached is not None:
                m_in_single = cached["input"]["m"].to(self.device, non_blocking=True)
                z_in = cached["input"]["z"].to(self.device, non_blocking=True)
                m_target_single = cached["output"]["m"].to(self.device, non_blocking=True)
                z_target = cached["output"]["z"].to(self.device, non_blocking=True)
                needs_batch_dim = m_in_single.dim() == 2
                if needs_batch_dim:
                    # Add batch + n_seq dims: [N, C] -> [1, 1, N, C]
                    m_in_msa = m_in_single.unsqueeze(0).unsqueeze(0)
                else:
                    # Add n_seq dim: [B, N, C] -> [B, 1, N, C]
                    m_in_msa = m_in_single.unsqueeze(-3)

                # Cache files store z as [N, N, C_z] (no batch dim). Always add batch dim when missing.
                z_in_for_block = z_in.unsqueeze(0) if z_in.dim() == 3 else z_in
            else:
                # Teacher path (single-sample): m_in_single / z_in / targets already set above.
                needs_batch_dim = m_in_single.dim() == 2
                if needs_batch_dim:
                    m_in_msa = m_in_single.unsqueeze(0).unsqueeze(0)
                else:
                    m_in_msa = m_in_single.unsqueeze(-3)
                z_in_for_block = z_in.unsqueeze(0) if z_in.dim() == 3 else z_in

            # Run replacement block
            replacement_block = self.replacement_blocks[str(block_idx)]
            m_pred_msa, z_pred = replacement_block(
                m_in_msa, z_in_for_block, msa_mask, pair_mask,
                chunk_size=None,
                use_deepspeed_evo_attention=False,
                use_lma=False,
                use_flash=False,
                inplace_safe=False,
                _mask_trans=True
            )
            
            # Extract single representation from output: [..., 1, N, C] -> [..., N, C]
            m_pred_single = m_pred_msa[..., 0, :, :]
            z_pred_for_loss = z_pred
            if needs_batch_dim:
                m_pred_single = m_pred_single.squeeze(0)
            if z_target.dim() == 3 and z_pred_for_loss.dim() == 4 and z_pred_for_loss.shape[0] == 1:
                z_pred_for_loss = z_pred_for_loss.squeeze(0)
            
            # Compute loss
            if cache_dir is not None and isinstance(seq_length, torch.Tensor) and seq_length.ndim == 1:
                # Masked MSE (padding-safe) for cache-only batched mode
                seq_mask_for_loss = seq_mask
                if seq_mask_for_loss is None:
                    raise ValueError("Cache-only mode requires seq_mask in batch")
                pair_mask_for_loss = seq_mask_for_loss.unsqueeze(-1) * seq_mask_for_loss.unsqueeze(-2)

                c_m = int(m_pred_single.shape[-1])
                c_z = int(z_pred_for_loss.shape[-1])
                denom_m = float(seq_mask_for_loss.sum().item())
                denom_z = float(pair_mask_for_loss.sum().item())
                if denom_m == 0.0 or denom_z == 0.0:
                    loss_single = torch.tensor(0.0, device=self.device)
                    loss_pair = torch.tensor(0.0, device=self.device)
                else:
                    loss_single = (
                        ((m_pred_single - m_target_single) ** 2) * seq_mask_for_loss.unsqueeze(-1)
                    ).sum() / (seq_mask_for_loss.sum() * c_m)
                    loss_pair = (
                        ((z_pred_for_loss - z_target) ** 2) * pair_mask_for_loss.unsqueeze(-1)
                    ).sum() / (pair_mask_for_loss.sum() * c_z)
            else:
                # Original behavior
                loss_single = F.mse_loss(m_pred_single, m_target_single)
                loss_pair = F.mse_loss(z_pred_for_loss, z_target)
            
            # Combine losses
            block_loss = loss_single + loss_pair
            total_loss += block_loss
            num_blocks += 1
            
            # Log per-block loss for all 46 blocks
            self.log(
                f'train/block_{block_idx:02d}_loss',
                block_loss,
                on_step=True,
                on_epoch=True,
                batch_size=log_batch_size,
                sync_dist=True,
            )
            
            # Clear intermediate tensors to free memory
            del m_in_msa, m_pred_msa, m_pred_single, z_pred_for_loss, block_loss
        
        # Average loss across all blocks
        if num_blocks > 0:
            avg_loss = total_loss / num_blocks
        else:
            avg_loss = torch.tensor(0.0, device=self.device)
        
        # Log overall metrics
        self.log('train/loss', avg_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=log_batch_size, sync_dist=True)
        self.log('train/num_blocks_trained', float(num_blocks), on_step=False, on_epoch=True, batch_size=log_batch_size, sync_dist=True)
        if isinstance(seq_length, torch.Tensor):
            self.log('train/seq_length', float(seq_length.float().mean().item()), on_step=False, on_epoch=True, batch_size=log_batch_size, sync_dist=True)
        else:
            self.log('train/seq_length', float(seq_length), on_step=False, on_epoch=True, batch_size=log_batch_size, sync_dist=True)
        
        # Clear teacher outputs to free GPU memory (if used)
        if block_data is not None:
            del block_data
        torch.cuda.empty_cache()

        if os.environ.get("AFDISTILL_CACHE_DEBUG") == "1" and batch_idx == 0 and int(getattr(self, "global_rank", 0)) == 0:
            tfc = self.teacher.forward_calls if self.teacher is not None else 0
            print(
                f"[cache_debug][train] seq_id={seq_id} hits={cache_hits} misses={cache_misses} "
                f"teacher_used={teacher_used} teacher_forward_calls={tfc}",
                flush=True,
            )
        
        return avg_loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step - same as training but without gradient updates"""
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        log_batch_size = 1
        
        # Extract metadata
        seq_length = batch.pop("seq_length")
        seq_id = batch.pop("seq_id")
        cache_dir = None
        safe_seq_id = None
        if self.block_data_dir:
            cache_dir = Path(self.block_data_dir)
            if isinstance(seq_id, (list, tuple)):
                safe_seq_id = [sanitize_id(str(s)) for s in seq_id]
            else:
                safe_seq_id = sanitize_id(str(seq_id))
        cache_can_write = None
        
        # Get masks from batch
        # msa_mask should be [batch, n_seq, n_res]
        msa_mask = batch.get('msa_mask', None)
        # seq_mask is [batch, n_res], need to expand to pair_mask [batch, n_res, n_res]
        seq_mask = batch.get('seq_mask', None)
        if seq_mask is not None:
            pair_mask = seq_mask.unsqueeze(-1) * seq_mask.unsqueeze(-2)
        else:
            pair_mask = None
        
        # Run teacher model lazily only if we have a cache miss
        block_data = None
        teacher_used = False
        cache_hits = 0
        cache_misses = 0

        # For merged format, load all blocks for each sample once before the loop.
        # Use map_location="cpu" for mmap; .to(device) in block loop handles transfer.
        merged_all = None
        is_merged = cache_dir is not None and self.block_data_format == "merged.safetensors"
        if is_merged and safe_seq_id is not None:
            merged_all = {}
            sids = safe_seq_id if isinstance(safe_seq_id, list) else [safe_seq_id]
            for sid in sids:
                merged_path = cache_dir / f"{sid}.safetensors"
                if not merged_path.exists():
                    raise FileNotFoundError(f"Missing merged cache file: {merged_path}")
                merged_all[sid] = load_merged_block_samples(merged_path, map_location="cpu")

        # Evaluate each replacement block
        total_loss = 0.0
        num_blocks = 0
        
        for block_idx in range(0, 48):
            if str(block_idx) not in self.replacement_blocks:
                continue
            
            cached = None
            if merged_all is not None:
                if isinstance(safe_seq_id, list):
                    cached = [merged_all[sid].get(block_idx) for sid in safe_seq_id]
                    if any(c is None for c in cached):
                        raise FileNotFoundError(f"Block {block_idx} missing in merged cache for some samples")
                else:
                    cached = merged_all[safe_seq_id].get(block_idx)
                    if cached is None:
                        raise FileNotFoundError(f"Block {block_idx} missing in merged cache for {safe_seq_id}")
            elif cache_dir is not None and safe_seq_id is not None:
                ext = self.block_data_format
                cache_map_location = self.device if self.block_data_format == "df11.safetensors" else "cpu"
                if isinstance(safe_seq_id, list):
                    cached = []
                    for sid in safe_seq_id:
                        cache_path = cache_dir / f"block_{block_idx:02d}" / f"{sid}.{ext}"
                        if not cache_path.exists():
                            raise FileNotFoundError(f"Missing cache file: {cache_path}")
                        cached.append(load_block_sample(cache_path, map_location=cache_map_location))
                else:
                    cache_path = cache_dir / f"block_{block_idx:02d}" / f"{safe_seq_id}.{ext}"
                    if not cache_path.exists():
                        raise FileNotFoundError(f"Missing cache file: {cache_path}")
                    cached = load_block_sample(cache_path, map_location=cache_map_location)

            if cached is not None:
                cache_hits += 1
            else:
                # Teacher path (single-sample only when block_data_dir is not set)
                cache_misses += 1
                if self.block_data_dir is not None:
                    raise RuntimeError("Cache-only mode: unexpected missing cached data")

                if block_data is None:
                    teacher_used = True
                    if self.teacher is None:
                        if self._teacher_model is None:
                            print("Loading teacher model...", flush=True)
                            self._teacher_model = self._load_teacher_model()
                        self.teacher = TeacherModelWithBlockCapture(self._teacher_model, self.device)
                        print(f"Teacher model initialized on {self.device}")
                    block_data = self.teacher.forward(batch)
                if block_idx not in block_data:
                    continue

                # Get block input/output from teacher (already single representation)
                m_in_single, z_in = block_data[block_idx]["input"]
                m_target_single, z_target = block_data[block_idx]["output"]

                # Cache to disk if enabled and not already present
                if cache_dir is not None and safe_seq_id is not None and not isinstance(safe_seq_id, list):
                    ext = self.block_data_format
                    cache_path = cache_dir / f"block_{block_idx:02d}" / f"{safe_seq_id}.{ext}"
                    if not cache_path.exists():
                        if cache_can_write is None:
                            free_bytes = get_free_bytes_on_filesystem(cache_dir)
                            cache_can_write = free_bytes >= MIN_BLOCK_DATA_CACHE_FREE_BYTES
                        if not cache_can_write:
                            continue

                        sample = {
                            "input": {"m": m_in_single.detach().cpu(), "z": z_in.detach().cpu()},
                            "output": {"m": m_target_single.detach().cpu(), "z": z_target.detach().cpu()},
                            "chain_id": str(seq_id),
                            "block_idx": int(block_idx),
                        }
                        tmp_tag = f".tmp.{os.getpid()}.{int(getattr(self, 'global_rank', 0))}"
                        suffix = f".{self.block_data_format}"
                        base_name = (
                            cache_path.name[: -len(suffix)] if cache_path.name.endswith(suffix) else cache_path.name
                        )
                        tmp_path = cache_path.with_name(base_name + tmp_tag + suffix)
                        save_block_sample(
                            sample,
                            tmp_path,
                            save_dtype=self.block_data_save_dtype,  # type: ignore[arg-type]
                            quantization=self.block_data_quantization,  # type: ignore[arg-type]
                        )
                        os.replace(tmp_path, cache_path)

            if isinstance(cached, list):
                # Cache-only batched mode: pad to Nmax and run once.
                if not isinstance(seq_length, torch.Tensor) or seq_length.ndim != 1:
                    raise ValueError("Cache-only mode expects seq_length as 1D tensor")
                bsz = int(seq_length.shape[0])
                log_batch_size = bsz
                if seq_mask is None:
                    raise ValueError("Cache-only mode requires seq_mask in batch")
                n_max = int(seq_mask.shape[1])

                m_list = []
                z_list = []
                mt_list = []
                zt_list = []
                for c in cached:
                    m = c["input"]["m"]
                    z = c["input"]["z"]
                    mt = c["output"]["m"]
                    zt = c["output"]["z"]
                    if m.dim() == 3 and m.shape[0] == 1:
                        m = m.squeeze(0)
                    if z.dim() == 4 and z.shape[0] == 1:
                        z = z.squeeze(0)
                    if mt.dim() == 3 and mt.shape[0] == 1:
                        mt = mt.squeeze(0)
                    if zt.dim() == 4 and zt.shape[0] == 1:
                        zt = zt.squeeze(0)
                    m_list.append(m)
                    z_list.append(z)
                    mt_list.append(mt)
                    zt_list.append(zt)

                c_m = int(m_list[0].shape[-1])
                c_z = int(z_list[0].shape[-1])
                m_in_padded = torch.zeros((bsz, n_max, c_m), dtype=m_list[0].dtype, device=self.device)
                m_tgt_padded = torch.zeros((bsz, n_max, c_m), dtype=mt_list[0].dtype, device=self.device)
                z_in_padded = torch.zeros((bsz, n_max, n_max, c_z), dtype=z_list[0].dtype, device=self.device)
                z_tgt_padded = torch.zeros((bsz, n_max, n_max, c_z), dtype=zt_list[0].dtype, device=self.device)

                seq_mask = seq_mask.clone()

                for i in range(bsz):
                    n_i = int(seq_length[i].item())
                    if n_i > n_max:
                        n_i = n_max
                        seq_length[i] = n_i
                        seq_mask[i, n_i:] = 0.0
                    n_cache = int(m_list[i].shape[0])
                    if z_list[i].dim() >= 2:
                        n_cache = min(n_cache, int(z_list[i].shape[0]), int(z_list[i].shape[1]))
                    if n_i > n_cache:
                        n_i = n_cache
                        seq_length[i] = n_i
                        seq_mask[i, n_i:] = 0.0
                    if n_i == 0:
                        continue
                    m_in_padded[i, :n_i, :] = m_list[i][:n_i]
                    m_tgt_padded[i, :n_i, :] = mt_list[i][:n_i]
                    z_in_padded[i, :n_i, :n_i, :] = z_list[i][:n_i, :n_i]
                    z_tgt_padded[i, :n_i, :n_i, :] = zt_list[i][:n_i, :n_i]

                msa_mask = seq_mask.unsqueeze(1)
                pair_mask = seq_mask.unsqueeze(-1) * seq_mask.unsqueeze(-2)

                m_in_msa = m_in_padded.unsqueeze(1)
                z_in_for_block = z_in_padded
                m_target_single = m_tgt_padded
                z_target = z_tgt_padded
                needs_batch_dim = False
            elif cached is not None:
                m_in_single = cached["input"]["m"].to(self.device, non_blocking=True)
                z_in = cached["input"]["z"].to(self.device, non_blocking=True)
                m_target_single = cached["output"]["m"].to(self.device, non_blocking=True)
                z_target = cached["output"]["z"].to(self.device, non_blocking=True)
                needs_batch_dim = m_in_single.dim() == 2
                if needs_batch_dim:
                    # Add batch + n_seq dims: [N, C] -> [1, 1, N, C]
                    m_in_msa = m_in_single.unsqueeze(0).unsqueeze(0)
                else:
                    # Add n_seq dim: [B, N, C] -> [B, 1, N, C]
                    m_in_msa = m_in_single.unsqueeze(-3)

                # Cache files store z as [N, N, C_z] (no batch dim). Always add batch dim when missing.
                z_in_for_block = z_in.unsqueeze(0) if z_in.dim() == 3 else z_in
            else:
                # Teacher path (single-sample): m_in_single / z_in / targets already set above.
                needs_batch_dim = m_in_single.dim() == 2
                if needs_batch_dim:
                    m_in_msa = m_in_single.unsqueeze(0).unsqueeze(0)
                else:
                    m_in_msa = m_in_single.unsqueeze(-3)
                z_in_for_block = z_in.unsqueeze(0) if z_in.dim() == 3 else z_in

            # Run replacement block
            replacement_block = self.replacement_blocks[str(block_idx)]
            m_pred_msa, z_pred = replacement_block(
                m_in_msa, z_in_for_block, msa_mask, pair_mask,
                chunk_size=None,
                use_deepspeed_evo_attention=False,
                use_lma=False,
                use_flash=False,
                inplace_safe=False,
                _mask_trans=True
            )
            
            # Extract single representation from output: [..., 1, N, C] -> [..., N, C]
            m_pred_single = m_pred_msa[..., 0, :, :]
            z_pred_for_loss = z_pred
            if needs_batch_dim:
                m_pred_single = m_pred_single.squeeze(0)
            if z_target.dim() == 3 and z_pred_for_loss.dim() == 4 and z_pred_for_loss.shape[0] == 1:
                z_pred_for_loss = z_pred_for_loss.squeeze(0)
            
            # Compute loss
            if cache_dir is not None and isinstance(seq_length, torch.Tensor) and seq_length.ndim == 1:
                seq_mask_for_loss = seq_mask
                if seq_mask_for_loss is None:
                    raise ValueError("Cache-only mode requires seq_mask in batch")
                pair_mask_for_loss = seq_mask_for_loss.unsqueeze(-1) * seq_mask_for_loss.unsqueeze(-2)

                c_m = int(m_pred_single.shape[-1])
                c_z = int(z_pred_for_loss.shape[-1])
                denom_m = float(seq_mask_for_loss.sum().item())
                denom_z = float(pair_mask_for_loss.sum().item())
                if denom_m == 0.0 or denom_z == 0.0:
                    loss_single = torch.tensor(0.0, device=self.device)
                    loss_pair = torch.tensor(0.0, device=self.device)
                else:
                    loss_single = (
                        ((m_pred_single - m_target_single) ** 2) * seq_mask_for_loss.unsqueeze(-1)
                    ).sum() / (seq_mask_for_loss.sum() * c_m)
                    loss_pair = (
                        ((z_pred_for_loss - z_target) ** 2) * pair_mask_for_loss.unsqueeze(-1)
                    ).sum() / (pair_mask_for_loss.sum() * c_z)
            else:
                loss_single = F.mse_loss(m_pred_single, m_target_single)
                loss_pair = F.mse_loss(z_pred_for_loss, z_target)
            
            block_loss = loss_single + loss_pair
            total_loss += block_loss
            num_blocks += 1
            
            # Log per-block loss for all 46 blocks
            self.log(
                f'val/block_{block_idx:02d}_loss',
                block_loss,
                on_step=False,
                on_epoch=True,
                batch_size=log_batch_size,
                sync_dist=True,
            )
            
            # Clear intermediate tensors
            del m_in_msa, m_pred_msa, m_pred_single, z_pred_for_loss, block_loss
        
        # Average loss
        if num_blocks > 0:
            avg_loss = total_loss / num_blocks
        else:
            avg_loss = torch.tensor(0.0, device=self.device)
        
        # Log validation metrics
        self.log('val/loss', avg_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=log_batch_size, sync_dist=True)
        self.log('val/num_blocks_evaluated', float(num_blocks), on_step=False, on_epoch=True, batch_size=log_batch_size, sync_dist=True)
        
        # Clear teacher outputs to free GPU memory (if used)
        if block_data is not None:
            del block_data
        torch.cuda.empty_cache()

        if os.environ.get("AFDISTILL_CACHE_DEBUG") == "1" and batch_idx == 0 and int(getattr(self, "global_rank", 0)) == 0:
            tfc = self.teacher.forward_calls if self.teacher is not None else 0
            print(
                f"[cache_debug][val] seq_id={seq_id} hits={cache_hits} misses={cache_misses} "
                f"teacher_used={teacher_used} teacher_forward_calls={tfc}",
                flush=True,
            )
        
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
    # If launched with torchrun/torch.distributed (TorchElastic), pin each process to its LOCAL_RANK GPU
    # as early as possible. Otherwise, any early CUDA initialization may happen on cuda:0 and leave
    # behind a ~260MiB context on GPU0 for every rank.
    local_rank_env = os.environ.get("LOCAL_RANK")
    world_size_env = os.environ.get("WORLD_SIZE")
    if local_rank_env is not None and world_size_env is not None and torch.cuda.is_available():
        torch.cuda.set_device(int(local_rank_env))

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

    # Enforce mode constraints early.
    if getattr(args, "block_data_dir", None) is None and int(args.batch_size) != 1:
        raise ValueError("Teacher mode (no block_data_dir) requires --batch_size 1")
    
    # Create data module
    print("Setting up data module...")
    data_module = PerBlockDataModule(
        dataset_path=args.dataset_path,
        config_preset=args.config_preset,
        block_data_dir=getattr(args, "block_data_dir", None),
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
        block_data_dir=getattr(args, "block_data_dir", None),
        block_data_format=getattr(args, "block_data_format", "pt"),
        block_data_save_dtype=getattr(args, "block_data_save_dtype", "bf16"),
        block_data_quantization=getattr(args, "block_data_quantization", "none"),
        replacement_type=args.replacement_type,
        linear_type=args.linear_type,
        kernel_size=args.kernel_size,
        dilations=args.dilations,
        dilation_pattern=args.dilation_pattern,
        dilation_repeats=args.dilation_repeats,
        replacement_mode=args.replacement_mode,
        replacement_checkpoint_subdir=args.replacement_checkpoint_subdir,
        allow_random_init=args.allow_random_init,
        compile_replacement=args.compile_replacement,
        resume_checkpoint_path=args.resume_checkpoint_path,
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
    
    # Setup strategy / devices.
    #
    # If launched via torchrun (TorchElastic), Lightning will NOT spawn subprocesses itself
    # (TorchElasticEnvironment.creates_processes_externally=True), but it *will* validate that
    # devices * num_nodes == WORLD_SIZE.
    launched_with_torchrun = (
        world_size_env is not None and int(world_size_env) > 1 and local_rank_env is not None
    )
    if launched_with_torchrun:
        world_size = int(world_size_env)  # total processes across all nodes
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size)))  # processes per node
        if local_world_size <= 0:
            local_world_size = world_size
        num_nodes = max(1, world_size // local_world_size)

        accelerator = "gpu" if torch.cuda.is_available() else "cpu"

        # Trainer(devices=...) is "devices per node", so this should match LOCAL_WORLD_SIZE.
        if args.gpus != local_world_size:
            print(
                f"Detected torchrun environment (WORLD_SIZE={world_size}, LOCAL_WORLD_SIZE={local_world_size}). "
                f"Overriding --gpus {args.gpus} -> {local_world_size} to match torchrun.",
                flush=True,
            )
        devices = local_world_size if accelerator == "gpu" else 1

        strategy = DDPStrategy(
            find_unused_parameters=False,
            process_group_backend=args.distributed_backend,
        )
    else:
        num_nodes = 1
        accelerator = "gpu" if args.gpus > 0 else "cpu"
        devices = args.gpus if args.gpus > 0 else 1
        if args.gpus > 1:
            strategy = DDPStrategy(
                find_unused_parameters=False,
                process_group_backend=args.distributed_backend,
            )
        else:
            strategy = "auto"
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=accelerator,
        devices=devices,
        num_nodes=num_nodes,
        strategy=strategy,
        precision=args.precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
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
    if args.resume_checkpoint_path is not None:
        ckpt = torch.load(args.resume_checkpoint_path, map_location="cpu")
        # If the checkpoint has optimizer_states, it's a full Lightning checkpoint
        # and we can let Lightning resume normally.
        if "optimizer_states" in ckpt and len(ckpt["optimizer_states"]) > 0:
            del ckpt
            trainer.fit(model, datamodule=data_module, ckpt_path=args.resume_checkpoint_path)
        else:
            # Weights-only checkpoint: load state_dict manually, train from scratch.
            sd = ckpt.get("state_dict", ckpt)
            missing, unexpected = model.load_state_dict(sd, strict=False)
            del ckpt, sd
            if missing:
                print(f"  Weights-only resume: {len(missing)} missing keys (first 5: {missing[:5]})")
            if unexpected:
                print(f"  Weights-only resume: {len(unexpected)} unexpected keys (first 5: {unexpected[:5]})")
            print(f"Loaded weights-only checkpoint from {args.resume_checkpoint_path}", flush=True)
            trainer.fit(model, datamodule=data_module)
    else:
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
    parser.add_argument(
        "--block_data_dir",
        type=str,
        default=None,
        help="Optional cache directory for per-block teacher I/O. If set, training will load cached block data "
             "from block_{XX}/{seq_id}.{block_data_format} when available; otherwise it will run the teacher and cache it.",
    )
    parser.add_argument(
        "--block_data_format",
        type=str,
        default="pt",
        choices=["pt", "pt.gz", "safetensors", "safetensors.gz", "safetensors.znn", "df11.safetensors", "merged.safetensors"],
        help="Cache file format used inside block_data_dir. 'merged.safetensors' expects one file per sample in a flat dir.",
    )
    parser.add_argument(
        "--block_data_save_dtype",
        type=str,
        default="bf16",
        choices=["float32", "bf16"],
        help="Floating-point dtype used when writing cached block data",
    )
    parser.add_argument(
        "--block_data_quantization",
        type=str,
        default="none",
        choices=["none"],
        help="Optional quantization applied before writing cached block data",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a Lightning checkpoint to resume from. If set, overrides auto-resume.",
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        default=False,
        help="Disable auto-resume from output_dir/checkpoints/last.ckpt",
    )
    
    # Model arguments
    parser.add_argument('--config_preset', type=str, default='model_1_ptm',
                       help='Model config preset')
    parser.add_argument(
        '--replacement_type',
        type=str,
        default='linear',
        choices=['linear', 'conv'],
        help="Replacement block type: 'linear' (SimpleEvoformerReplacement) or 'conv' (DilatedConvEvoformerReplacement)",
    )
    parser.add_argument('--linear_type', type=str, default='full',
                       choices=['full', 'diagonal', 'affine'],
                       help='Type of linear layers in replacement blocks')
    parser.add_argument(
        "--kernel_size",
        type=int,
        default=3,
        help="Kernel size for dilated convolution replacement blocks (conv mode)",
    )
    parser.add_argument(
        "--dilations",
        type=str,
        default="1,2,4",
        help="Comma-separated dilations for conv replacement blocks (e.g. '1,2,4,8,16')",
    )
    parser.add_argument(
        "--dilation_pattern",
        type=str,
        default=None,
        help="Optional dilation pattern to repeat for shared_proj mode (e.g. '1,2,4,8').",
    )
    parser.add_argument(
        "--dilation_repeats",
        type=int,
        default=1,
        help="Number of times to repeat the dilation pattern.",
    )
    parser.add_argument(
        "--replacement_mode",
        type=str,
        default="per_block",
        choices=["per_block", "shared_proj"],
        help="Convolutional replacement architecture mode (conv mode)",
    )
    parser.add_argument(
        "--replacement_checkpoint_subdir",
        type=str,
        default=None,
        help="Override checkpoint subdirectory under each block_{XX}/. "
             "If unset, uses linear_type for linear or a conv_* tag for conv.",
    )
    parser.add_argument(
        "--allow_random_init",
        action="store_true",
        default=False,
        help="If set, initialize missing replacement blocks randomly instead of skipping them.",
    )
    
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
    parser.add_argument(
        '--accumulate_grad_batches',
        type=int,
        default=1,
        help='Number of gradient accumulation steps',
    )
    parser.add_argument('--gradient_clip_val', type=float, default=1.0,
                       help='Gradient clipping value')

    # Compilation arguments
    parser.add_argument(
        "--compile_replacement",
        action="store_true",
        default=False,
        help="Compile replacement blocks with torch.compile (mode=reduce-overhead)",
    )
    
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

    # Resolve resume checkpoint path with highest priority:
    # 1) explicit --resume_from_checkpoint
    # 2) output_dir/checkpoints/last.ckpt (unless --no_resume)
    # 3) newest .ckpt in output_dir/checkpoints/ (fallback if last.ckpt missing)
    args.resume_checkpoint_path = None
    if args.resume_from_checkpoint is not None:
        args.resume_checkpoint_path = args.resume_from_checkpoint
    elif not args.no_resume:
        ckpt_dir = Path(args.output_dir) / "checkpoints"
        last_ckpt = ckpt_dir / "last.ckpt"
        if last_ckpt.exists():
            args.resume_checkpoint_path = str(last_ckpt)
        elif ckpt_dir.exists():
            ckpt_files = sorted(
                [p for p in ckpt_dir.glob("*.ckpt") if p.is_file()],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if len(ckpt_files) > 0:
                args.resume_checkpoint_path = str(ckpt_files[0])
    if args.resume_checkpoint_path is not None:
        print(f"Will resume from checkpoint: {args.resume_checkpoint_path}", flush=True)
    
    # Validate arguments
    if getattr(args, "block_data_dir", None) is None and int(args.batch_size) != 1:
        raise ValueError("Teacher mode (no block_data_dir) requires --batch_size 1")
    
    main(args, parser)

