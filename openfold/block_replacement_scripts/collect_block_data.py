#!/usr/bin/env python3
"""
Data collection script for Evoformer block input/output pairs.

This script collects input/output pairs for all 48 Evoformer blocks during forward passes,
using recycle 0 values. The data will be used to train replacement blocks.
"""

import argparse
import os
import sys
import json
import pickle
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
import yaml
import warnings
warnings.filterwarnings('ignore')

# Add openfold to path
sys.path.append(str(Path.home() / 'openfold'))

from openfold.block_replacement_scripts import _torch_pytree_compat  # noqa: F401
from openfold.config import model_config
from openfold.model.model import AlphaFold
from openfold.data import feature_pipeline, data_pipeline, templates
from openfold.utils.import_weights import import_jax_weights_
from openfold.utils.tensor_utils import tensor_tree_map
from openfold.block_replacement_scripts.block_data_io import save_block_sample, sanitize_id

from enhanced_data_utils import EnhancedStructureFinder
from evaluation_features_utils import extract_sequence_from_structure


def get_dist_info() -> Tuple[int, int, int]:
    """
    Return (rank, world_size, local_rank) from torchrun-style environment variables.

    Defaults to single-process (0/1) when not launched under torchrun.
    """
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world_size, local_rank


def cast_floating_tensors(obj: Any, dtype: torch.dtype) -> Any:
    """
    Recursively cast floating-point torch.Tensors inside nested dict/list/tuple structures.
    Non-tensor leaves (e.g., strings/ints) are returned unchanged.
    """
    if isinstance(obj, torch.Tensor):
        if torch.is_floating_point(obj) and obj.dtype != dtype:
            return obj.to(dtype=dtype)
        return obj
    if isinstance(obj, dict):
        return {k: cast_floating_tensors(v, dtype) for k, v in obj.items()}
    if isinstance(obj, list):
        return [cast_floating_tensors(v, dtype) for v in obj]
    if isinstance(obj, tuple):
        return tuple(cast_floating_tensors(v, dtype) for v in obj)
    return obj


class EvoformerBlockDataCollector:
    """Collects input/output pairs for all Evoformer blocks"""
    
    def __init__(self, args):
        self.args = args
        self.rank, self.world_size, self.local_rank = get_dist_info()
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.device = torch.device("cpu")
        
        # Setup paths
        self.home_dir = Path.home()
        self.csv_path = (self.home_dir / args.csv_path) if args.csv_path is not None else None
        self.tsv_path = (self.home_dir / args.tsv_path) if args.tsv_path is not None else None
        self.pdb_dir = (self.home_dir / args.pdb_dir) if args.pdb_dir is not None else None
        self.output_dir = self.home_dir / args.output_dir
        self.weights_path = self.home_dir / args.weights_path
        self.chain_to_sequence: Dict[str, str] = {}
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Data storage (no hooks needed)
        
        # Setup data pipeline
        self.data_processor, self.feature_processor = self._setup_data_pipeline()
        
        print(f"Initialized data collector:")
        print(f"  Rank: {self.rank}/{self.world_size} (local_rank={self.local_rank})")
        print(f"  Device: {self.device}")
        if self.tsv_path is not None:
            print(f"  TSV path: {self.tsv_path}")
            print(f"  PDB directory: (unused)")
        else:
            print(f"  CSV path: {self.csv_path}")
            print(f"  PDB directory: {self.pdb_dir}")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Weights: {self.weights_path}")
        print(f"  Config preset: {self.args.config_preset}")
        if self.args.min_length is not None or self.args.max_length is not None:
            print(f"  Length filter: min={self.args.min_length} max={self.args.max_length}")
        print(f"  Save dtype: {self.args.save_dtype}")
        print(f"  Save format: {self.args.save_format}")
        print(f"  Save quantization: {self.args.save_quantization}")
        print()

    def _setup_data_pipeline(self):
        """Setup the data and feature processing pipelines"""
        config = model_config(self.args.config_preset, train=False, low_prec=False)
        
        # Create dummy template directory
        temp_template_dir = tempfile.mkdtemp()
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
        
        self.temp_template_dir = temp_template_dir
        
        data_processor = data_pipeline.DataPipeline(template_featurizer=template_featurizer)
        feature_processor = feature_pipeline.FeaturePipeline(config.data)
        
        return data_processor, feature_processor

    def __del__(self):
        """Cleanup temporary directory"""
        if hasattr(self, 'temp_template_dir'):
            try:
                shutil.rmtree(self.temp_template_dir)
            except:
                pass

    def _load_model(self) -> AlphaFold:
        """Load the pretrained model"""
        print("Loading pretrained model...")
        
        config = model_config(self.args.config_preset, train=False, low_prec=False)
        model = AlphaFold(config)
        
        # Load weights
        if self.weights_path.suffix == ".npz":
            model_basename = self.weights_path.stem
            model_version = "_".join(model_basename.split("_")[1:])
            import_jax_weights_(model, str(self.weights_path), version=model_version)
        else:
            checkpoint = torch.load(self.weights_path, map_location="cpu")
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            
            if any(k.startswith('model.') for k in state_dict.keys()):
                state_dict = {k[6:]: v for k, v in state_dict.items() if k.startswith('model.')}
            
            model.load_state_dict(state_dict, strict=False)
        
        model = model.to(self.device)
        model.eval()
        
        print(f"Model loaded successfully")
        return model


    def _load_chain_list(self) -> List[str]:
        """Load chain list (CSV + structures) or (TSV id/sequence) without train/val split"""
        if self.tsv_path is not None:
            print("Loading chain list from TSV...")
            chains: List[str] = []
            dropped_too_short = 0
            dropped_too_long = 0
            min_len = self.args.min_length
            max_len = self.args.max_length
            with open(self.tsv_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split("\t")
                    if len(parts) < 2:
                        continue
                    chain_id, seq = parts[0], parts[1]
                    L = len(seq)
                    if min_len is not None and L < int(min_len):
                        dropped_too_short += 1
                        continue
                    if max_len is not None and L > int(max_len):
                        dropped_too_long += 1
                        continue
                    chains.append(chain_id)
                    self.chain_to_sequence[chain_id] = seq

            print(f"  Total chains in TSV: {len(chains)}")
            if dropped_too_short or dropped_too_long:
                print(
                    f"  Dropped by length: too_short={dropped_too_short} too_long={dropped_too_long}",
                )
            if self.args.max_proteins and self.args.max_proteins < len(chains):
                chains = chains[:self.args.max_proteins]
                print(f"  Limited to: {len(chains)} proteins")
            if self.world_size > 1:
                chains = chains[self.rank::self.world_size]
                print(f"  Shard: rank {self.rank}/{self.world_size} -> {len(chains)} proteins")
            print(f"  Total chains to process: {len(chains)}")
            return chains

        print("Loading chain list from CSV...")
        df = pd.read_csv(self.csv_path)
        all_chains = df["natives_rcsb"].dropna().tolist()
        
        # Filter by available structure files
        structure_finder = EnhancedStructureFinder(
            str(self.pdb_dir),
            [".cif", ".pdb", ".core"],
            None,
        )
        
        available_chains = []
        for chain in all_chains:
            try:
                structure_finder.find_structure_path(chain)
                available_chains.append(chain)
            except ValueError:
                continue
        
        print(f"  Total chains in CSV: {len(all_chains)}")
        print(f"  Chains with available structures: {len(available_chains)}")
        
        # Limit if specified
        if self.args.max_proteins and self.args.max_proteins < len(available_chains):
            available_chains = available_chains[: self.args.max_proteins]
            print(f"  Limited to: {len(available_chains)} proteins")
        if self.world_size > 1:
            available_chains = available_chains[self.rank::self.world_size]
            print(f"  Shard: rank {self.rank}/{self.world_size} -> {len(available_chains)} proteins")
        
        print(f"  Total chains to process: {len(available_chains)}")
        return available_chains

    def _create_features_from_sequence(self, chain_id: str, sequence: str) -> Dict[str, np.ndarray]:
        """Create features for a protein from its amino-acid sequence"""
        safe_chain_id = sanitize_id(chain_id)
        tmp_fasta_path = os.path.join(os.getcwd(), f"tmp_{os.getpid()}_{safe_chain_id}.fasta")
        with open(tmp_fasta_path, "w") as fp:
            fp.write(f">{chain_id}\n{sequence}")
        
        # Create local alignment directory
        temp_alignment_dir = tempfile.mkdtemp()
        local_alignment_dir = os.path.join(temp_alignment_dir, safe_chain_id)
        os.makedirs(local_alignment_dir, exist_ok=True)
        
        # Create minimal MSA file
        msa_path = os.path.join(local_alignment_dir, "output.a3m")
        with open(msa_path, "w") as f:
            f.write(f">{chain_id}\n{sequence}\n")
        
        # Process features
        feature_dict = self.data_processor.process_fasta(
            fasta_path=tmp_fasta_path,
            alignment_dir=local_alignment_dir,
            seqemb_mode=False,
        )
        
        # Cleanup
        os.remove(tmp_fasta_path)
        shutil.rmtree(temp_alignment_dir)
        return feature_dict

    def _create_features(self, chain_id: str) -> Dict[str, np.ndarray]:
        """Create features for a protein from an on-disk structure file"""
        structure_finder = EnhancedStructureFinder(
            str(self.pdb_dir),
            [".cif", ".pdb", ".core"],
            None
        )
        
        structure_path, file_id, chain_id_only, ext = structure_finder.find_structure_path(chain_id)
        sequence = extract_sequence_from_structure(structure_path, chain_id_only)
        return self._create_features_from_sequence(chain_id, sequence)

    def _run_inference_with_hooks(self, model: AlphaFold, feature_dict: Dict[str, np.ndarray]):
        """Run inference and collect block data via the built-in outputs mechanism"""
        
        # Process features
        processed_feature_dict = self.feature_processor.process_features(
            feature_dict, mode='predict', is_multimer=False
        )
        
        # Convert to tensors
        processed_feature_dict = {
            k: torch.as_tensor(v, device=self.device)
            for k, v in processed_feature_dict.items()
        }
        
        # Run inference - OpenFold will automatically capture intermediate outputs
        # when they are available (the model has built-in support for this)
        with torch.no_grad():
            outputs = model(processed_feature_dict)
        
        return outputs

    def collect_data(self):
        """Main data collection function"""
        print("=== Evoformer Block Data Collection ===")
        print()
        
        # Load model
        model = self._load_model()
        
        try:
            # Load chain list (no train/val split)
            all_chains = self._load_chain_list()
            
            # Process all data
            print(f"\nProcessing data ({len(all_chains)} proteins) on rank {self.rank}/{self.world_size}...")
            for i, chain_id in enumerate(
                tqdm(all_chains, desc=f"Processing proteins (rank {self.rank}/{self.world_size})")
            ):
                # Check if data already exists for this protein
                if self._check_protein_data_exists(chain_id):
                    print(f"  ✅ Data already exists for {chain_id} - skipping")
                    continue
                
                # Create features and run inference
                if self.tsv_path is not None:
                    sequence = self.chain_to_sequence[chain_id]
                    features = self._create_features_from_sequence(chain_id, sequence)
                else:
                    features = self._create_features(chain_id)
                outputs = self._run_inference_with_hooks(model, features)
                
                # Save data for this protein
                self._save_protein_data(chain_id, outputs)
                
                # Clear GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            print("\nData collection completed!")
            
        finally:
            # Clean up
            pass
        
        # Save metadata
        self._save_metadata(all_chains)

    def _check_protein_data_exists(self, chain_id: str) -> bool:
        """Check if data has already been collected for this protein (all requested blocks present)."""
        safe_chain_id = sanitize_id(chain_id)
        
        for block_idx in self.args.blocks:
            block_dir = self.output_dir / f"block_{block_idx:02d}"
            pt_file = block_dir / f"{safe_chain_id}.pt"
            pt_gz_file = block_dir / f"{safe_chain_id}.pt.gz"
            st_file = block_dir / f"{safe_chain_id}.safetensors"
            st_gz_file = block_dir / f"{safe_chain_id}.safetensors.gz"
            st_znn_file = block_dir / f"{safe_chain_id}.safetensors.znn"
            df11_file = block_dir / f"{safe_chain_id}.df11.safetensors"
            pkl_file = block_dir / f"{safe_chain_id}.pkl"  # Backward compatibility
            
            if not (
                pt_file.exists()
                or pt_gz_file.exists()
                or st_file.exists()
                or st_gz_file.exists()
                or st_znn_file.exists()
                or df11_file.exists()
                or pkl_file.exists()
            ):
                return False

        return True

    def _save_protein_data(self, chain_id: str, outputs: Dict[str, torch.Tensor]):
        """Save data for a single protein using built-in intermediate outputs"""
        safe_chain_id = sanitize_id(chain_id)
        
        # Extract recycle 0 data (we want only the first recycle)
        recycle_0_keys = [k for k in outputs.keys() if k.startswith('recycle_0_block_')]
        
        if not recycle_0_keys:
            print(f"    Warning: No recycle_0_block outputs found for {chain_id}")
            print(f"    Available keys: {list(outputs.keys())[:10]}...")  # Show first 10 keys
            return
        
        print(f"    Found {len(recycle_0_keys)} block outputs for {chain_id}")
        
        saved_blocks = 0
        # Process each block
        for block_key in recycle_0_keys:
            block_idx = int(block_key.split("_")[-1])
            
            # Create block-specific directory
            block_dir = self.output_dir / f"block_{block_idx:02d}"
            os.makedirs(block_dir, exist_ok=True)
            
            # Get the block outputs (tuple of m, z)
            block_output = outputs[block_key]
            m_out, z_out = block_output[0], block_output[1]
            
            if block_idx not in self.args.blocks:
                continue

            # For inputs, we need the output of the previous block (or evoformer input for block 0)
            if block_idx == 0:
                evo_input_key = "recycle_0_evoformer_input"
                if evo_input_key not in outputs:
                    print(f"    Warning: No evoformer input found for block 0 of {chain_id}")
                    continue
                evo_input = outputs[evo_input_key]
                m_in, z_in = evo_input[0], evo_input[1]
            else:
                # Input is the output of the previous block
                prev_block_key = f"recycle_0_block_{block_idx-1}"
                if prev_block_key not in outputs:
                    continue
                prev_block_output = outputs[prev_block_key]
                m_in, z_in = prev_block_output[0], prev_block_output[1]
            
            # Save input/output pair
            data = {
                "input": {
                    "m": m_in.detach().cpu()[:1],
                    "z": z_in.detach().cpu()
                },
                "output": {
                    "m": m_out.detach().cpu()[:1],
                    "z": z_out.detach().cpu()
                },
                "chain_id": chain_id,
                "block_idx": block_idx
            }
            
            save_dtype = "bf16" if self.args.save_dtype == "bf16" else "float32"
            if self.args.save_format == "pt":
                save_path = block_dir / f"{safe_chain_id}.pt"
            elif self.args.save_format == "pt.gz":
                save_path = block_dir / f"{safe_chain_id}.pt.gz"
            elif self.args.save_format == "safetensors":
                save_path = block_dir / f"{safe_chain_id}.safetensors"
            elif self.args.save_format == "safetensors.gz":
                save_path = block_dir / f"{safe_chain_id}.safetensors.gz"
            elif self.args.save_format == "safetensors.znn":
                save_path = block_dir / f"{safe_chain_id}.safetensors.znn"
            elif self.args.save_format == "df11.safetensors":
                save_path = block_dir / f"{safe_chain_id}.df11.safetensors"
            else:
                raise ValueError(f"Unsupported save_format: {self.args.save_format}")

            save_block_sample(
                data,
                save_path,
                save_dtype=save_dtype,
                quantization=self.args.save_quantization,
            )
            
            saved_blocks += 1
        
        print(f"    Saved data for {saved_blocks} blocks")

    def _save_metadata(self, all_chains: List[str]):
        """Save metadata about the collected data"""
        
        metadata = {
            "all_chains": all_chains,
            "total_blocks": 48,
            "weights_path": str(self.weights_path),
            "config": self.args.config_preset,
            "save_dtype": self.args.save_dtype,
            "save_format": self.args.save_format,
            "save_quantization": self.args.save_quantization,
            "min_length": self.args.min_length,
            "max_length": self.args.max_length,
            "rank": int(self.rank),
            "world_size": int(self.world_size),
        }

        suffix_parts: List[str] = []
        if self.args.run_tag is not None and str(self.args.run_tag).strip() != "":
            suffix_parts.append(str(self.args.run_tag))
        elif self.world_size > 1:
            suffix_parts.append(f"rank{self.rank}")
        suffix = "" if len(suffix_parts) == 0 else "." + ".".join(suffix_parts)
        
        with open(self.output_dir / f"metadata{suffix}.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create summary
        summary = []
        for block_idx in range(48):
            block_dir = self.output_dir / f"block_{block_idx:02d}"
            
            if block_dir.exists():
                sample_count = (
                    len(list(block_dir.glob("*.pt")))
                    + len(list(block_dir.glob("*.pt.gz")))
                    + len(list(block_dir.glob("*.safetensors")))
                    + len(list(block_dir.glob("*.safetensors.gz")))
                    + len(list(block_dir.glob("*.safetensors.znn")))
                )
            else:
                sample_count = 0
            
            summary.append({
                "block_idx": block_idx,
                "total_samples": sample_count
            })
        
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(self.output_dir / f"data_summary{suffix}.csv", index=False)
        
        print(f"\nData collection summary:")
        print(f"  Total blocks: 48")
        print(f"  Total proteins: {len(all_chains)}")
        print(f"  Data saved to: {self.output_dir}")
        print(f"  Summary saved to: {self.output_dir / f'data_summary{suffix}.csv'}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect input/output pairs for all Evoformer blocks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--csv_path", type=str, default=None,
        help="Path to CSV file containing chain list (relative to home directory). Requires --pdb_dir."
    )
    parser.add_argument(
        "--tsv_path", type=str, default=None,
        help="Path to TSV file containing 'id\\tsequence' per line (relative to home directory). Ignores --pdb_dir."
    )
    parser.add_argument(
        "--pdb_dir", type=str, default=None,
        help="Directory containing PDB/mmCIF files (relative to home directory). Required if using --csv_path."
    )
    parser.add_argument(
        "--weights_path", type=str, required=True,
        help="Path to pretrained weights (relative to home directory)"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for collected data (relative to home directory)"
    )
    parser.add_argument(
        "--config_preset",
        type=str,
        default="model_1_ptm",
        help="OpenFold model config preset (should match weights used during training)",
    )
    parser.add_argument(
        "--train_config",
        type=str,
        default=None,
        help="Optional YAML config file (relative to home directory) to populate min_length/max_length defaults",
    )
    parser.add_argument(
        "--min_length",
        type=int,
        default=None,
        help="Optional minimum sequence length filter (applied in TSV mode)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Optional maximum sequence length filter (applied in TSV mode)",
    )
    parser.add_argument(
        "--run_tag",
        type=str,
        default=None,
        help="Optional tag to suffix metadata/data_summary outputs (useful when running multiple 1-GPU processes)",
    )
    parser.add_argument(
        "--max_proteins", type=int, default=None,
        help="Maximum number of proteins to process (None for all)"
    )
    parser.add_argument(
        "--blocks", type=int, nargs="+", 
        default=list(range(1, 47)),
        help="Block indices to process (default: all blocks 1-46)"
    )
    parser.add_argument(
        "--save_dtype",
        type=str,
        choices=["float32", "bf16"],
        default="float32",
        help="Dtype used when saving floating-point tensors in .pt files",
    )
    parser.add_argument(
        "--save_format",
        type=str,
        choices=["pt", "pt.gz", "safetensors", "safetensors.gz", "safetensors.znn", "df11.safetensors"],
        default="pt",
        help="On-disk format for saved block data",
    )
    parser.add_argument(
        "--save_quantization",
        type=str,
        choices=["none"],
        default="none",
        help="Optional quantization applied before serialization (currently only 'none')",
    )
    
    args = parser.parse_args()
    if args.train_config is not None:
        cfg_path = Path.home() / args.train_config
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        if args.min_length is None and "min_length" in cfg:
            args.min_length = int(cfg["min_length"])
        if args.max_length is None and "max_length" in cfg:
            args.max_length = int(cfg["max_length"])
    if args.tsv_path is None and args.csv_path is None:
        raise ValueError("Must provide either --tsv_path or --csv_path")
    if args.tsv_path is None and args.pdb_dir is None:
        raise ValueError("--pdb_dir is required when using --csv_path")
    
    # Run data collection
    collector = EvoformerBlockDataCollector(args)
    collector.collect_data()


if __name__ == "__main__":
    main()
