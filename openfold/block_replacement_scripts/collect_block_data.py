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
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Data storage (no hooks needed)
        
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


    def _load_chain_list(self) -> List[str]:
        """Load chain list without train/val split"""
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
        
        print(f"  Total chains to process: {len(available_chains)}")
        
        return available_chains

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
            print(f"\nProcessing data ({len(all_chains)} proteins)...")
            for i, chain_id in enumerate(tqdm(all_chains, desc="Processing proteins")):
                # Check if data already exists for this protein
                if self._check_protein_data_exists(chain_id):
                    print(f"  ✅ Data already exists for {chain_id} - skipping")
                    continue
                
                # Create features and run inference
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
        """Check if data has already been collected for this protein"""
        
        # Check if at least one block file exists for this protein
        for block_idx in range(1, 47):  # Skip first and last blocks
            block_dir = self.output_dir / f"block_{block_idx:02d}"
            pt_file = block_dir / f"{chain_id}.pt"
            pkl_file = block_dir / f"{chain_id}.pkl"  # Backward compatibility
            
            if pt_file.exists() or pkl_file.exists():
                return True
        
        return False

    def _save_protein_data(self, chain_id: str, outputs: Dict[str, torch.Tensor]):
        """Save data for a single protein using built-in intermediate outputs"""
        
        # Extract recycle 0 data (we want only the first recycle)
        recycle_0_keys = [k for k in outputs.keys() if k.startswith('recycle_0_block_')]
        
        if not recycle_0_keys:
            print(f"    Warning: No recycle_0_block outputs found for {chain_id}")
            print(f"    Available keys: {list(outputs.keys())[:10]}...")  # Show first 10 keys
            return
        
        print(f"    Found {len(recycle_0_keys)} block outputs for {chain_id}")
        
        saved_blocks = 0
        # Process each block
        for block_idx in range(48):
            block_key = f"recycle_0_block_{block_idx}"
            
            if block_key not in outputs:
                continue  # Skip if this block wasn't captured
            
            # Create block-specific directory
            block_dir = self.output_dir / f"block_{block_idx:02d}"
            os.makedirs(block_dir, exist_ok=True)
            
            # Get the block outputs (tuple of m, z)
            block_output = outputs[block_key]
            m_out, z_out = block_output[0], block_output[1]
            
            # For inputs, we need the output of the previous block (or initial embeddings for block 0)
            if block_idx == 0:
                # For block 0, we don't have "inputs" from a previous block
                # Skip block 0 for now, or handle specially
                continue
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
                    "m": m_in.detach().cpu(),
                    "z": z_in.detach().cpu()
                },
                "output": {
                    "m": m_out.detach().cpu(),
                    "z": z_out.detach().cpu()
                },
                "chain_id": chain_id,
                "block_idx": block_idx
            }
            
            save_path = block_dir / f"{chain_id}.pt"
            torch.save(data, save_path)
            
            saved_blocks += 1
        
        print(f"    Saved data for {saved_blocks} blocks")

    def _save_metadata(self, all_chains: List[str]):
        """Save metadata about the collected data"""
        
        metadata = {
            "all_chains": all_chains,
            "total_blocks": 48,
            "weights_path": str(self.weights_path),
            "config": "model_2_ptm"
        }
        
        with open(self.output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create summary
        summary = []
        for block_idx in range(48):
            block_dir = self.output_dir / f"block_{block_idx:02d}"
            
            sample_count = len(list(block_dir.glob("*.pt"))) if block_dir.exists() else 0
            
            summary.append({
                "block_idx": block_idx,
                "total_samples": sample_count
            })
        
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(self.output_dir / "data_summary.csv", index=False)
        
        print(f"\nData collection summary:")
        print(f"  Total blocks: 48")
        print(f"  Total proteins: {len(all_chains)}")
        print(f"  Data saved to: {self.output_dir}")
        print(f"  Summary saved to: {self.output_dir / 'data_summary.csv'}")


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
    parser.add_argument(
        "--blocks", type=int, nargs="+", 
        default=list(range(1, 47)),
        help="Block indices to process (default: all blocks 1-46)"
    )
    
    args = parser.parse_args()
    
    # Run data collection
    collector = EvoformerBlockDataCollector(args)
    collector.collect_data()


if __name__ == "__main__":
    main()
