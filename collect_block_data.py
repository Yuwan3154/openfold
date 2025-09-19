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
import warnings
warnings.filterwarnings('ignore')

# Add openfold to path
sys.path.append(str(Path.home() / 'openfold'))

from openfold.config import model_config
from openfold.model.model import AlphaFold
from openfold.data import feature_pipeline, data_pipeline, templates
from openfold.utils.import_weights import import_jax_weights_
from openfold.utils.tensor_utils import tensor_tree_map

from enhanced_data_utils import EnhancedStructureFinder
from evaluation_features_utils import extract_sequence_from_structure


class EvoformerBlockDataCollector:
    """Collects input/output pairs for all Evoformer blocks"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup paths
        self.home_dir = Path.home()
        self.csv_path = self.home_dir / args.csv_path
        self.pdb_dir = self.home_dir / args.pdb_dir
        self.output_dir = self.home_dir / args.output_dir
        self.weights_path = self.home_dir / args.weights_path
        
        # Create output directories
        self.data_dir = self.output_dir / "block_data"
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Hook storage for each block
        self.block_data = {i: {"inputs": [], "outputs": []} for i in range(48)}
        self.hooks = []
        
        # Setup data pipeline
        self.data_processor, self.feature_processor = self._setup_data_pipeline()
        
        print(f"Initialized data collector:")
        print(f"  Device: {self.device}")
        print(f"  CSV path: {self.csv_path}")
        print(f"  PDB directory: {self.pdb_dir}")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Weights: {self.weights_path}")
        print()

    def _setup_data_pipeline(self):
        """Setup the data and feature processing pipelines"""
        config = model_config("model_2_ptm", train=False, low_prec=False)
        
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
        
        config = model_config("model_2_ptm", train=False, low_prec=False)
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

    def _setup_hooks(self, model: AlphaFold):
        """Setup forward hooks for all Evoformer blocks"""
        print("Setting up hooks for all 48 Evoformer blocks...")
        
        def create_hook(block_idx):
            def hook_fn(module, input, output):
                # Store input and output for this block
                # input is a tuple, output is a tuple (m, z)
                m_in, z_in = input[0], input[1]
                m_out, z_out = output[0], output[1]
                
                # Store copies on CPU to save GPU memory
                self.block_data[block_idx]["inputs"].append({
                    "m": m_in.detach().cpu().clone(),
                    "z": z_in.detach().cpu().clone()
                })
                self.block_data[block_idx]["outputs"].append({
                    "m": m_out.detach().cpu().clone(), 
                    "z": z_out.detach().cpu().clone()
                })
            return hook_fn
        
        # Register hooks for all blocks
        for i, block in enumerate(model.evoformer.blocks):
            hook = block.register_forward_hook(create_hook(i))
            self.hooks.append(hook)
        
        print(f"Registered hooks for {len(self.hooks)} blocks")

    def _remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def _load_chain_list(self) -> Tuple[List[str], List[str]]:
        """Load chain list and create train/val split"""
        print("Loading chain list from CSV...")
        
        df = pd.read_csv(self.csv_path)
        all_chains = df['natives_rcsb'].dropna().tolist()
        
        # Filter by available structure files
        structure_finder = EnhancedStructureFinder(
            str(self.pdb_dir),
            [".cif", ".pdb", ".core"],
            None
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
            available_chains = available_chains[:self.args.max_proteins]
            print(f"  Limited to: {len(available_chains)} proteins")
        
        # Create 80/20 train/val split
        np.random.seed(42)
        indices = np.random.permutation(len(available_chains))
        split_idx = int(0.8 * len(available_chains))
        
        train_chains = [available_chains[i] for i in indices[:split_idx]]
        val_chains = [available_chains[i] for i in indices[split_idx:]]
        
        print(f"  Train set: {len(train_chains)} proteins")
        print(f"  Validation set: {len(val_chains)} proteins")
        
        return train_chains, val_chains

    def _create_features(self, chain_id: str) -> Dict[str, np.ndarray]:
        """Create features for a protein"""
        structure_finder = EnhancedStructureFinder(
            str(self.pdb_dir),
            [".cif", ".pdb", ".core"],
            None
        )
        
        structure_path, file_id, chain_id_only, ext = structure_finder.find_structure_path(chain_id)
        sequence = extract_sequence_from_structure(structure_path, chain_id_only)
                
        tmp_fasta_path = os.path.join(os.getcwd(), f"tmp_{os.getpid()}_{chain_id}.fasta")
        with open(tmp_fasta_path, "w") as fp:
            fp.write(f">{chain_id}\n{sequence}")
        
        # Create local alignment directory
        temp_alignment_dir = tempfile.mkdtemp()
        local_alignment_dir = os.path.join(temp_alignment_dir, chain_id)
        os.makedirs(local_alignment_dir, exist_ok=True)
        
        # Create minimal MSA file
        msa_path = os.path.join(local_alignment_dir, "output.a3m")
        with open(msa_path, 'w') as f:
            f.write(f">{chain_id}\n{sequence}\n")
        
        # Process features
        feature_dict = self.data_processor.process_fasta(
            fasta_path=tmp_fasta_path,
            alignment_dir=local_alignment_dir,
            seqemb_mode=True
        )
        
        # Cleanup
        os.remove(tmp_fasta_path)
        shutil.rmtree(temp_alignment_dir)
        
        return feature_dict

    def _run_inference_with_hooks(self, model: AlphaFold, feature_dict: Dict[str, np.ndarray]):
        """Run inference and collect block data via hooks"""
        
        # Process features
        processed_feature_dict = self.feature_processor.process_features(
            feature_dict, mode='predict', is_multimer=False
        )
        
        # Convert to tensors
        processed_feature_dict = {
            k: torch.as_tensor(v, device=self.device)
            for k, v in processed_feature_dict.items()
        }
        
        # Run inference (only first recycle - recycle 0)
        with torch.no_grad():
            # Set recycles to 1 to get only recycle 0
            original_recycles = model.config.model.recycle_features
            model.config.model.recycle_features = False  # This ensures only one pass
            
            outputs = model(processed_feature_dict)
            
            # Restore original setting
            model.config.model.recycle_features = original_recycles
        
        return outputs

    def collect_data(self):
        """Main data collection function"""
        print("=== Evoformer Block Data Collection ===")
        print()
        
        # Load model and setup hooks
        model = self._load_model()
        self._setup_hooks(model)
        
        try:
            # Load chain lists
            train_chains, val_chains = self._load_chain_list()
            
            # Process training data
            print(f"\nProcessing training data ({len(train_chains)} proteins)...")
            for i, chain_id in enumerate(tqdm(train_chains, desc="Training proteins")):
                try:
                    # Clear block data for this protein
                    for block_idx in range(48):
                        self.block_data[block_idx]["inputs"].clear()
                        self.block_data[block_idx]["outputs"].clear()
                    
                    # Create features and run inference
                    features = self._create_features(chain_id)
                    outputs = self._run_inference_with_hooks(model, features)
                    
                    # Save data for this protein
                    self._save_protein_data(chain_id, "train")
                    
                    # Clear GPU cache
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    print(f"Error processing {chain_id}: {e}")
                    continue
            
            # Process validation data
            print(f"\nProcessing validation data ({len(val_chains)} proteins)...")
            for i, chain_id in enumerate(tqdm(val_chains, desc="Validation proteins")):
                try:
                    # Clear block data for this protein
                    for block_idx in range(48):
                        self.block_data[block_idx]["inputs"].clear()
                        self.block_data[block_idx]["outputs"].clear()
                    
                    # Create features and run inference
                    features = self._create_features(chain_id)
                    outputs = self._run_inference_with_hooks(model, features)
                    
                    # Save data for this protein
                    self._save_protein_data(chain_id, "val")
                    
                    # Clear GPU cache
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    print(f"Error processing {chain_id}: {e}")
                    continue
            
            print("\nData collection completed!")
            
        finally:
            # Always remove hooks
            self._remove_hooks()
        
        # Save metadata
        self._save_metadata(train_chains, val_chains)

    def _save_protein_data(self, chain_id: str, split: str):
        """Save data for a single protein"""
        
        for block_idx in range(48):
            if not self.block_data[block_idx]["inputs"]:
                continue  # Skip if no data collected
            
            # Create block-specific directory
            block_dir = self.data_dir / f"block_{block_idx:02d}" / split
            os.makedirs(block_dir, exist_ok=True)
            
            # Save input/output pair
            data = {
                "input": self.block_data[block_idx]["inputs"][0],  # Should be only one
                "output": self.block_data[block_idx]["outputs"][0],
                "chain_id": chain_id
            }
            
            save_path = block_dir / f"{chain_id}.pkl"
            with open(save_path, 'wb') as f:
                pickle.dump(data, f)

    def _save_metadata(self, train_chains: List[str], val_chains: List[str]):
        """Save metadata about the collected data"""
        
        metadata = {
            "train_chains": train_chains,
            "val_chains": val_chains,
            "total_blocks": 48,
            "weights_path": str(self.weights_path),
            "config": "model_2_ptm"
        }
        
        with open(self.data_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create summary
        summary = []
        for block_idx in range(48):
            train_dir = self.data_dir / f"block_{block_idx:02d}" / "train"
            val_dir = self.data_dir / f"block_{block_idx:02d}" / "val"
            
            train_count = len(list(train_dir.glob("*.pkl"))) if train_dir.exists() else 0
            val_count = len(list(val_dir.glob("*.pkl"))) if val_dir.exists() else 0
            
            summary.append({
                "block_idx": block_idx,
                "train_samples": train_count,
                "val_samples": val_count
            })
        
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(self.data_dir / "data_summary.csv", index=False)
        
        print(f"\nData collection summary:")
        print(f"  Total blocks: 48")
        print(f"  Train proteins: {len(train_chains)}")
        print(f"  Val proteins: {len(val_chains)}")
        print(f"  Data saved to: {self.data_dir}")
        print(f"  Summary saved to: {self.data_dir / 'data_summary.csv'}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect input/output pairs for all Evoformer blocks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--csv_path", type=str, required=True,
        help="Path to CSV file containing chain list (relative to home directory)"
    )
    parser.add_argument(
        "--pdb_dir", type=str, required=True,
        help="Directory containing PDB/mmCIF files (relative to home directory)"
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
        "--max_proteins", type=int, default=None,
        help="Maximum number of proteins to process (None for all)"
    )
    
    args = parser.parse_args()
    
    # Run data collection
    collector = EvoformerBlockDataCollector(args)
    collector.collect_data()


if __name__ == "__main__":
    main()
