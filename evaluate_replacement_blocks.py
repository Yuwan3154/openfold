#!/usr/bin/env python3
"""
Evaluation script for trained replacement blocks.

This script tests the fitted replacement blocks in the actual OpenFold model
and compares their performance against the "block removed" baseline using TM scores.
"""

import argparse
import os
import sys
import json
import subprocess
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

from custom_evoformer_replacement import SimpleEvoformerReplacement
from enhanced_data_utils import EnhancedStructureFinder
from evaluation_features_utils import extract_sequence_from_structure
from train_replacement_blocks import ReplacementBlockTrainer


class ReplacementBlockEvaluator:
    """Evaluates trained replacement blocks in the full model"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup paths
        self.home_dir = Path.home()
        self.csv_path = self.home_dir / args.csv_path
        self.pdb_dir = self.home_dir / args.pdb_dir
        self.weights_path = self.home_dir / args.weights_path
        self.trained_models_dir = self.home_dir / args.trained_models_dir
        self.output_dir = self.home_dir / args.output_dir
        
        # Create output directories
        self.predictions_dir = self.output_dir / "predictions"
        self.analysis_dir = self.output_dir / "analysis"
        
        for dir_path in [self.predictions_dir, self.analysis_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Setup data pipeline
        self.data_processor, self.feature_processor = self._setup_data_pipeline()
        
        # Model dimensions
        self.c_m = 256
        self.c_z = 128
        
        print(f"Initialized evaluator:")
        print(f"  Device: {self.device}")
        print(f"  CSV path: {self.csv_path}")
        print(f"  PDB directory: {self.pdb_dir}")
        print(f"  Weights: {self.weights_path}")
        print(f"  Trained models: {self.trained_models_dir}")
        print(f"  Output directory: {self.output_dir}")
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

    def _load_base_model(self) -> AlphaFold:
        """Load the base pretrained model"""
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
        
        return model

    def _load_trained_replacement_block(self, block_idx: int, linear_type: str) -> Optional[SimpleEvoformerReplacement]:
        """Load a trained replacement block"""
        
        checkpoint_path = self.trained_models_dir / f"block_{block_idx:02d}" / linear_type / "best_model.ckpt"
        
        if not checkpoint_path.exists():
            return None
        
        # Load the PyTorch Lightning checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Create the replacement block with same config as training
        replacement_block = SimpleEvoformerReplacement(
            c_m=self.c_m,
            c_z=self.c_z,
            m_hidden_dim=self.args.hidden_dim,
            z_hidden_dim=self.args.hidden_dim,
            linear_type=linear_type,
            gating=True,
            residual=True
        )
        
        # Extract the replacement block state dict from the Lightning module
        lightning_state_dict = checkpoint['state_dict']
        replacement_state_dict = {}
        
        for key, value in lightning_state_dict.items():
            if key.startswith('replacement_block.'):
                new_key = key[len('replacement_block.'):]
                replacement_state_dict[new_key] = value
        
        replacement_block.load_state_dict(replacement_state_dict)
        replacement_block = replacement_block.to(self.device)
        replacement_block.eval()
        
        return replacement_block

    def _create_model_with_replacement(self, block_idx: int, linear_type: str) -> Optional[AlphaFold]:
        """Create model with a specific block replaced"""
        
        # Load base model
        model = self._load_base_model()
        
        # Load trained replacement block
        replacement_block = self._load_trained_replacement_block(block_idx, linear_type)
        
        if replacement_block is None:
            return None
        
        # Replace the block
        model.evoformer.blocks[block_idx] = replacement_block
        
        return model

    def _create_model_with_removed_block(self, block_idx: int) -> AlphaFold:
        """Create model with a specific block removed"""
        
        model = self._load_base_model()
        
        # Remove the block
        new_blocks = nn.ModuleList([
            block for i, block in enumerate(model.evoformer.blocks) 
            if i != block_idx
        ])
        
        model.evoformer.blocks = new_blocks
        
        return model

    def _load_chain_list(self) -> List[str]:
        """Load validation chain list"""
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
        
        # Use validation split (last 20%)
        split_idx = int(0.8 * len(available_chains))
        val_chains = available_chains[split_idx:]
        
        # Limit if specified
        if self.args.max_proteins and self.args.max_proteins < len(val_chains):
            val_chains = val_chains[:self.args.max_proteins]
        
        print(f"  Validation chains: {len(val_chains)}")
        
        return val_chains

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

    def _run_inference(self, model: AlphaFold, feature_dict: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """Run inference on a single protein"""
        
        # Process features
        processed_feature_dict = self.feature_processor.process_features(
            feature_dict, mode='predict', is_multimer=False
        )
        
        # Convert to tensors
        processed_feature_dict = {
            k: torch.as_tensor(v, device=self.device)
            for k, v in processed_feature_dict.items()
        }
        
        # Run inference
        with torch.no_grad():
            outputs = model(processed_feature_dict)
        
        return outputs

    def _save_structure(self, outputs: Dict[str, torch.Tensor], chain_id: str, output_path: Path):
        """Save predicted structure as PDB file"""
        
        final_atom_positions = outputs["final_atom_positions"].cpu().numpy()
        
        # Remove batch dimension if present
        if final_atom_positions.ndim == 4:
            final_atom_positions = final_atom_positions[0]
        
        final_atom_mask = outputs.get("final_atom_mask", None)
        if final_atom_mask is not None:
            final_atom_mask = final_atom_mask.cpu().numpy()
            if final_atom_mask.ndim > 2:
                final_atom_mask = final_atom_mask[0]
        
        # Get confidence scores
        plddt = outputs["plddt"].cpu().numpy()
        if plddt.ndim > 1:
            plddt = plddt[0]
        
        # Get sequence length
        seq_len = final_atom_positions.shape[0]
        
        # Create PDB string
        pdb_lines = []
        pdb_lines.append("HEADER    PREDICTED STRUCTURE")
        pdb_lines.append(f"TITLE     PREDICTION FOR {chain_id}")
        
        atom_index = 1
        for res_index in range(seq_len):
            res_num = res_index + 1
            
            # Write main chain atoms (N, CA, C, O)
            for atom_name, atom_idx in [("N", 0), ("CA", 1), ("C", 2), ("O", 3)]:
                if final_atom_mask is None:
                    include_atom = True
                elif final_atom_mask.ndim == 2 and final_atom_mask.shape[1] > atom_idx:
                    include_atom = final_atom_mask[res_index, atom_idx]
                elif final_atom_mask.ndim == 1:
                    include_atom = final_atom_mask[res_index] 
                else:
                    include_atom = True
                
                if include_atom:
                    coord = final_atom_positions[res_index, atom_idx]
                    if plddt.ndim == 0:
                        confidence = float(plddt)
                    elif plddt.ndim == 1 and len(plddt) > res_index:
                        confidence = plddt[res_index]
                    else:
                        confidence = 50.0
                    
                    pdb_line = (
                        f"ATOM  {atom_index:5d}  {atom_name:3s} ALA A{res_num:4d}    "
                        f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}"
                        f"  1.00{confidence:6.2f}           {atom_name[0]}"
                    )
                    pdb_lines.append(pdb_line)
                    atom_index += 1
        
        pdb_lines.append("END")
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write('\n'.join(pdb_lines))

    def _calculate_tm_score(self, pred_pdb: Path, true_pdb: Path) -> float:
        """Calculate TM-score using USalign"""
        
        cmd = ["USalign", str(pred_pdb), str(true_pdb)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            return 0.0
        
        # Parse TM-score from output
        for line in result.stdout.split('\n'):
            if "TM-score=" in line and "normalized by length of Structure_1" in line:
                try:
                    tm_score = float(line.split("TM-score=")[1].split()[0])
                    return tm_score
                except (IndexError, ValueError):
                    continue
        
        return 0.0

    def evaluate_all_models(self):
        """Main evaluation function"""
        
        print("=== Evaluating Replacement Blocks in Full Model ===")
        print()
        
        # Load chain list
        chain_list = self._load_chain_list()
        
        # Determine which models to evaluate
        linear_types = ["full", "diagonal", "affine"]
        
        if self.args.test_blocks:
            block_indices = self.args.test_blocks
        else:
            # Find all available trained models
            block_indices = []
            for block_dir in self.trained_models_dir.glob("block_*"):
                try:
                    block_idx = int(block_dir.name.split("_")[1])
                    block_indices.append(block_idx)
                except:
                    continue
            block_indices = sorted(set(block_indices))
        
        print(f"Evaluating blocks: {block_indices}")
        print(f"Linear types: {linear_types}")
        print(f"Validation proteins: {len(chain_list)}")
        print()
        
        # Store all results
        all_results = []
        
        # Evaluate each combination
        for block_idx in tqdm(block_indices, desc="Blocks"):
            print(f"\nEvaluating Block {block_idx:02d}...")
            
            # Evaluate block removed baseline
            print(f"  Evaluating removed baseline...")
            removed_results = self._evaluate_single_model(
                block_idx, "removed", chain_list
            )
            all_results.extend(removed_results)
            
            # Evaluate each linear type
            for linear_type in linear_types:
                print(f"  Evaluating {linear_type} replacement...")
                replacement_results = self._evaluate_single_model(
                    block_idx, linear_type, chain_list
                )
                all_results.extend(replacement_results)
        
        # Save and analyze results
        self._save_and_analyze_results(all_results)
        
        print(f"\n=== Evaluation Complete ===")
        print(f"Results saved to: {self.output_dir}")

    def _evaluate_single_model(self, block_idx: int, model_type: str, chain_list: List[str]) -> List[Dict]:
        """Evaluate a single model configuration"""
        
        # Create model
        if model_type == "removed":
            model = self._create_model_with_removed_block(block_idx)
        else:
            model = self._create_model_with_replacement(block_idx, model_type)
        
        if model is None:
            print(f"    Could not load model, skipping...")
            return []
        
        # Create output directory
        model_dir = self.predictions_dir / f"block_{block_idx:02d}_{model_type}"
        os.makedirs(model_dir, exist_ok=True)
        
        results = []
        
        # Evaluate on each protein
        for chain_id in tqdm(chain_list, desc=f"  {model_type}", leave=False):
            try:
                # Create features
                features = self._create_features(chain_id)
                
                # Run inference
                outputs = self._run_inference(model, features)
                
                # Save prediction
                pred_pdb_path = model_dir / f"{chain_id}.pdb"
                self._save_structure(outputs, chain_id, pred_pdb_path)
                
                # Get ground truth structure path
                structure_finder = EnhancedStructureFinder(
                    str(self.pdb_dir),
                    [".cif", ".pdb", ".core"],
                    None
                )
                true_structure_path, _, _, _ = structure_finder.find_structure_path(chain_id)
                
                # Calculate TM-score
                tm_score = self._calculate_tm_score(pred_pdb_path, Path(true_structure_path))
                
                # Extract pTM score
                ptm_score = 0.0
                if "ptm_score" in outputs:
                    ptm_tensor = outputs["ptm_score"]
                    if ptm_tensor.numel() == 1:
                        ptm_score = ptm_tensor.item()
                    elif ptm_tensor.numel() > 1:
                        ptm_score = ptm_tensor.mean().item()
                
                results.append({
                    "block_idx": block_idx,
                    "model_type": model_type,
                    "chain_id": chain_id,
                    "tm_score": tm_score,
                    "ptm_score": ptm_score,
                    "prediction_path": str(pred_pdb_path)
                })
                
                # Clear GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"    Error evaluating {chain_id}: {e}")
                continue
        
        return results

    def _save_and_analyze_results(self, all_results: List[Dict]):
        """Save results and perform analysis"""
        
        # Save detailed results
        with open(self.analysis_dir / "detailed_results.json", 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Create DataFrame for analysis
        df = pd.DataFrame(all_results)
        df.to_csv(self.analysis_dir / "evaluation_results.csv", index=False)
        
        if len(df) == 0:
            print("No results to analyze")
            return
        
        # Analyze results
        print("\n=== Results Analysis ===")
        
        # Overall statistics by model type
        print("\nOverall performance by model type:")
        model_stats = df.groupby('model_type').agg({
            'tm_score': ['mean', 'std', 'count'],
            'ptm_score': ['mean', 'std']
        }).round(4)
        print(model_stats)
        
        # Performance by block index
        print("\nPerformance by block index (mean TM-score):")
        block_stats = df.groupby(['block_idx', 'model_type'])['tm_score'].mean().unstack().round(4)
        print(block_stats)
        
        # Best replacement type per block
        print("\nBest replacement type per block:")
        replacement_df = df[df['model_type'] != 'removed']
        if len(replacement_df) > 0:
            best_per_block = replacement_df.groupby('block_idx').apply(
                lambda x: x.loc[x.groupby('model_type')['tm_score'].mean().idxmax()]
            )[['model_type', 'tm_score']].groupby('block_idx').first()
            print(best_per_block)
        
        # Save analysis summary
        with open(self.analysis_dir / "analysis_summary.txt", 'w') as f:
            f.write("Replacement Block Evaluation Results\n")
            f.write("=" * 40 + "\n\n")
            f.write("Overall performance by model type:\n")
            f.write(str(model_stats))
            f.write("\n\nPerformance by block index:\n")
            f.write(str(block_stats))
            if len(replacement_df) > 0:
                f.write("\n\nBest replacement type per block:\n")
                f.write(str(best_per_block))


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained replacement blocks in full model",
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
        "--trained_models_dir", type=str, required=True,
        help="Directory containing trained replacement models (relative to home directory)"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for evaluation results (relative to home directory)"
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=256,
        help="Hidden dimension used in replacement blocks"
    )
    parser.add_argument(
        "--max_proteins", type=int, default=None,
        help="Maximum number of proteins to evaluate (None for all validation set)"
    )
    parser.add_argument(
        "--test_blocks", type=int, nargs="+", default=None,
        help="Specific block indices to evaluate (for testing, default: all available)"
    )
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluator = ReplacementBlockEvaluator(args)
    evaluator.evaluate_all_models()


if __name__ == "__main__":
    main()
