#!/usr/bin/env python3
"""
Comprehensive evaluation script for OpenFold model architectures.

This script evaluates three model configurations:
1. Original architecture with pretrained weights
2. Fine-tuned model with Evoformer block replacement 
3. Original architecture with specified Evoformer block removed

For each model, it:
- Runs inference on a list of proteins from CSV+folder structure
- Saves predicted structures to organized directories
- Calculates TM-scores using USalign against ground truth
- Extracts pTM scores from model predictions
- Generates comprehensive analysis plots and metrics
"""

import argparse
import os
import sys
import json
import subprocess
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
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
from openfold.data.data_pipeline import DataPipeline
from openfold.data.parsers import parse_fasta
from openfold.data.mmcif_parsing import parse as parse_mmcif
from openfold.data.mmcif_parsing import mmcif_loop_to_list
from openfold.data.tools import hhsearch, hmmsearch, jackhmmer
from openfold.np import residue_constants, protein
from openfold.utils.import_weights import import_jax_weights_
from openfold.utils.tensor_utils import (
    tensor_tree_map,
    masked_mean,
)

# Import our custom modules
from openfold.block_replacement_scripts.custom_evoformer_replacement import (
    replace_evoformer_block,
    freeze_all_except_replaced_block
)
from openfold.block_replacement_scripts.enhanced_data_utils import (
    EnhancedStructureFinder,
    build_chain_ids_from_structures_and_list
)
from openfold.block_replacement_scripts.evaluation_features_utils import features_from_chain_id


class ModelArchitectureEvaluator:
    """Evaluates different OpenFold model architectures"""
    
    def __init__(self, args):
        """Initialize the evaluator with command line arguments"""
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup paths
        self.home_dir = Path.home()
        self.csv_path = self.home_dir / args.csv_path
        self.pdb_dir = self.home_dir / args.pdb_dir
        self.output_dir = self.home_dir / args.output_dir
        self.original_weights = self.home_dir / args.original_weights
        self.finetuned_weights = self.home_dir / args.finetuned_weights if args.finetuned_weights else None
        
        # Create output directories
        self.predictions_dir = self.output_dir / "predictions"
        self.analysis_dir = self.output_dir / "analysis"
        self.plots_dir = self.output_dir / "plots"
        
        for dir_path in [self.predictions_dir, self.analysis_dir, self.plots_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Model configurations
        self.models = {}
        self.results = {}
        
        # Setup data pipeline (like official OpenFold)
        self.data_processor, self.feature_processor = self._setup_data_pipeline()
        
        print(f"Initialized evaluator:")
        print(f"  Device: {self.device}")
        print(f"  CSV path: {self.csv_path}")
        print(f"  PDB directory: {self.pdb_dir}")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Original weights: {self.original_weights}")
        if self.finetuned_weights:
            print(f"  Fine-tuned weights: {self.finetuned_weights}")
        print()

    def _setup_data_pipeline(self):
        """Setup the data and feature processing pipelines like official OpenFold"""
        # Create config - use model_2_ptm to enable pTM head
        config = model_config("model_2_ptm", train=False, low_prec=False)
        
        # Set up template featurizer with no templates (for single sequence mode)
        # Create a dummy template directory with a minimal CIF file
        temp_template_dir = tempfile.mkdtemp()
        dummy_cif_path = os.path.join(temp_template_dir, "dummy.cif")
        
        # Create a minimal dummy CIF file
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
            mmcif_dir=temp_template_dir,  # Provide valid path even though max_hits=0
            max_template_date="2025-01-01",
            max_hits=0,  # No templates for single sequence mode
            kalign_binary_path="/usr/bin/kalign",
            release_dates_path=None,
            obsolete_pdbs_path=None
        )
        
        # Store temp dir for cleanup later
        self.temp_template_dir = temp_template_dir
        
        # Create data processor
        data_processor = data_pipeline.DataPipeline(
            template_featurizer=template_featurizer,
        )
        
        # Create feature processor
        feature_processor = feature_pipeline.FeaturePipeline(config.data)
        
        return data_processor, feature_processor
    
    def __del__(self):
        """Cleanup temporary directory"""
        if hasattr(self, 'temp_template_dir'):
            import shutil
            try:
                shutil.rmtree(self.temp_template_dir)
            except:
                pass

    def _load_chain_list(self) -> List[str]:
        """Load chain list from CSV file"""
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
        
        # Limit number of proteins if specified
        if self.args.max_proteins and self.args.max_proteins < len(available_chains):
            available_chains = available_chains[:self.args.max_proteins]
            print(f"  Will evaluate on: {len(available_chains)} proteins (limited by --max_proteins)")
        else:
            print(f"  Will evaluate on: {len(available_chains)} proteins")
        
        return available_chains

    def _create_model_configs(self) -> Dict[str, Dict]:
        """Create configurations for the three model architectures"""
        
        # Use model_2_ptm config to enable pTM head
        base_config = model_config("model_2_ptm", train=False, low_prec=False)
        
        configs = {
            "original": {
                "name": "Original Architecture",
                "config": base_config,
                "weights_path": self.original_weights,
                "modifications": None,
                "output_dir": self.predictions_dir / "original"
            },
            "removed_block": {
                "name": f"Block {self.args.replace_block_index} Removed",
                "config": base_config,
                "weights_path": self.original_weights,
                "modifications": "remove_block",
                "output_dir": self.predictions_dir / "removed_block"
            },
            "replaced_block": {
                "name": f"Block {self.args.replace_block_index} Replaced (Pretrained)",
                "config": base_config,
                "weights_path": self.original_weights,
                "modifications": "replace_block",
                "output_dir": self.predictions_dir / "replaced_block"
            }
        }
        
        # Add fine-tuned model if weights are provided
        if self.finetuned_weights:
            configs["finetuned"] = {
                "name": f"Fine-tuned (Block {self.args.replace_block_index} Replaced)",
                "config": base_config,
                "weights_path": self.finetuned_weights,
                "modifications": "replace_block_finetuned",
                "output_dir": self.predictions_dir / "finetuned"
            }
        
        # Create output directories
        for config in configs.values():
            os.makedirs(config["output_dir"], exist_ok=True)
        
        return configs

    def _load_model(self, config_info: Dict) -> AlphaFold:
        """Load a model with specified configuration"""
        
        print(f"Loading model: {config_info['name']}")
        
        # Create model
        model = AlphaFold(config_info["config"])
        
        # Apply modifications if needed
        if config_info["modifications"] == "remove_block":
            model = self._remove_evoformer_block(model, self.args.replace_block_index)
        elif config_info["modifications"] in ["replace_block", "replace_block_finetuned"]:
            # Replace block (this will be done after loading weights for compatibility)
            pass
        
        # Load weights
        if config_info["weights_path"].suffix == ".npz":
            # JAX weights
            model_basename = config_info["weights_path"].stem
            model_version = "_".join(model_basename.split("_")[1:])
            import_jax_weights_(model, str(config_info["weights_path"]), version=model_version)
        else:
            # PyTorch weights
            checkpoint = torch.load(config_info["weights_path"], map_location="cpu")
            
            # Load state dict first (before modifications)
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            
            # Remove 'model.' prefix if present
            if any(k.startswith('model.') for k in state_dict.keys()):
                state_dict = {k[6:]: v for k, v in state_dict.items() if k.startswith('model.')}
            
            # Load weights into the original model structure first
            print(f"  Loading weights...")
            model.load_state_dict(state_dict, strict=False)
            
            # Apply modifications AFTER loading weights
            if config_info["modifications"] in ["replace_block", "replace_block_finetuned"]:
                print(f"  Applying block replacement...")
                c_m = config_info["config"].model.evoformer_stack.c_m
                c_z = config_info["config"].model.evoformer_stack.c_z
                model = replace_evoformer_block(
                    model, 
                    self.args.replace_block_index, 
                    c_m, c_z, 
                    self.args.replacement_hidden_dim
                )
                print(f"  Replaced Evoformer block {self.args.replace_block_index} with custom layer")
        
        model = model.to(self.device)
        model.eval()
        
        print(f"  Model loaded successfully")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        return model

    def _remove_evoformer_block(self, model: AlphaFold, block_index: int) -> AlphaFold:
        """Remove a specific Evoformer block from the model"""
        
        if not (0 < block_index < len(model.evoformer.blocks) - 1):
            raise ValueError("Block index must not be the first or last block.")
        
        # Create new list without the specified block
        new_blocks = nn.ModuleList([
            block for i, block in enumerate(model.evoformer.blocks) 
            if i != block_index
        ])
        
        model.evoformer.blocks = new_blocks
        
        print(f"  Removed Evoformer block {block_index}")
        print(f"  Remaining blocks: {len(new_blocks)}")
        
        return model

    def _create_features(self, chain_id: str) -> Dict[str, np.ndarray]:
        """Create features using the official OpenFold data pipeline"""
        
        # Extract sequence from structure file
        structure_finder = EnhancedStructureFinder(
            str(self.pdb_dir),
            [".cif", ".pdb", ".core"],
            None
        )
        
        # Get structure path
        structure_path, file_id, chain_id_only, ext = structure_finder.find_structure_path(chain_id)
        
        # Extract sequence from structure
        from evaluation_features_utils import extract_sequence_from_structure
        sequence = extract_sequence_from_structure(structure_path, chain_id_only)
                
        tmp_fasta_path = os.path.join(os.getcwd(), f"tmp_{os.getpid()}_{chain_id}.fasta")
        with open(tmp_fasta_path, "w") as fp:
            fp.write(f">{chain_id}\n{sequence}")
        
        # Create local alignment directory (following official pattern)
        temp_alignment_dir = tempfile.mkdtemp()
        local_alignment_dir = os.path.join(temp_alignment_dir, chain_id)
        os.makedirs(local_alignment_dir, exist_ok=True)
        
        # Create minimal MSA file (for seqemb mode)
        msa_path = os.path.join(local_alignment_dir, "output.a3m")
        with open(msa_path, 'w') as f:
            f.write(f">{chain_id}\n{sequence}\n")
        
        # Use official OpenFold data pipeline with seqemb_mode=True
        feature_dict = self.data_processor.process_fasta(
            fasta_path=tmp_fasta_path,
            alignment_dir=local_alignment_dir,
            seqemb_mode=True  # Single sequence mode like official script
        )
        
        # Cleanup
        os.remove(tmp_fasta_path)
        shutil.rmtree(temp_alignment_dir)
        
        return feature_dict

    def _run_inference(self, model: AlphaFold, feature_dict: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """Run inference on a single protein following official OpenFold pattern"""
        
        # Process features using OpenFold's feature processor (exactly like the official script)
        processed_feature_dict = self.feature_processor.process_features(
            feature_dict, mode='predict', is_multimer=False
        )
        
        # Convert to tensors and move to device (exactly like the official script)
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
        if final_atom_positions.ndim == 4:  # [batch, N_res, 37, 3]
            final_atom_positions = final_atom_positions[0]
        elif final_atom_positions.ndim == 3:  # [N_res, 37, 3] - already correct
            pass
        else:
            print(f"    Warning: Unexpected final_atom_positions shape: {final_atom_positions.shape}")
        
        final_atom_mask = outputs.get("final_atom_mask", None)
        if final_atom_mask is not None:
            final_atom_mask = final_atom_mask.cpu().numpy()
            if final_atom_mask.ndim > 2:
                final_atom_mask = final_atom_mask[0]  # Remove batch dim
        
        # Get confidence scores
        plddt = outputs["plddt"].cpu().numpy()
        if plddt.ndim > 1:
            plddt = plddt[0]  # Remove batch dim
        
        # Get sequence length
        seq_len = final_atom_positions.shape[0]
        
        # Create a simple PDB string
        pdb_lines = []
        pdb_lines.append("HEADER    PREDICTED STRUCTURE")
        pdb_lines.append(f"TITLE     PREDICTION FOR {chain_id}")
        
        atom_index = 1
        for res_index in range(seq_len):
            res_num = res_index + 1
            
            # Write main chain atoms (N, CA, C, O)
            for atom_name, atom_idx in [("N", 0), ("CA", 1), ("C", 2), ("O", 3)]:
                # Check mask properly based on its dimensions after batch removal
                if final_atom_mask is None:
                    include_atom = True
                elif final_atom_mask.ndim == 2 and final_atom_mask.shape[1] > atom_idx:
                    # 2D mask: [N_res, 37]
                    include_atom = final_atom_mask[res_index, atom_idx]
                elif final_atom_mask.ndim == 1:
                    # 1D mask: [N_res] - residue level mask
                    include_atom = final_atom_mask[res_index] 
                else:
                    # Unknown format, include atom
                    include_atom = True
                
                if include_atom:
                    coord = final_atom_positions[res_index, atom_idx]
                    # Handle plddt dimensions correctly
                    if plddt.ndim == 0:
                        confidence = float(plddt)
                    elif plddt.ndim == 1 and len(plddt) > res_index:
                        confidence = plddt[res_index]
                    else:
                        confidence = 50.0  # Default confidence
                    
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

    def _extract_ptm_score(self, outputs: Dict[str, torch.Tensor]) -> float:
        """Extract pTM score from model outputs"""
        
        if "ptm_score" in outputs:
            ptm_tensor = outputs["ptm_score"]
            if ptm_tensor.numel() == 1:
                return ptm_tensor.item()
            elif ptm_tensor.numel() > 1:
                # Take mean if multiple values
                return ptm_tensor.mean().item()
        elif "ptm" in outputs:
            return outputs["ptm"].item()
        elif "predicted_tm" in outputs:
            return outputs["predicted_tm"].item()
        else:
            return 0.0  # Placeholder if no pTM found

    def _calculate_tm_score(self, pred_pdb: Path, true_pdb: Path) -> float:
        """Calculate TM-score using USalign"""
        
        # Run USalign
        cmd = ["USalign", str(pred_pdb), str(true_pdb)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            print(f"USalign failed for {pred_pdb.name}: {result.stderr}")
            return 0.0
        
        # Parse TM-score from output
        # Look for lines like: "TM-score= 0.53210 (normalized by length of Structure_1: L=105, d0=3.76)"
        for line in result.stdout.split('\n'):
            if "TM-score=" in line and "normalized by length of Structure_1" in line:
                # Extract TM-score (format: TM-score= 0.xxxxx)
                try:
                    tm_score = float(line.split("TM-score=")[1].split()[0])
                    return tm_score
                except (IndexError, ValueError) as e:
                    continue
        
        return 0.0

    def evaluate_models(self):
        """Main evaluation function"""
        
        print("=== OpenFold Model Architecture Evaluation ===")
        print()
        
        # Load chain list
        chain_list = self._load_chain_list()
        
        # Create model configurations
        model_configs = self._create_model_configs()
        
        # Load models
        print("Loading models...")
        for model_name, config_info in model_configs.items():
            self.models[model_name] = self._load_model(config_info)
            self.results[model_name] = {
                "predictions": {},
                "ptm_scores": {},
                "tm_scores": {},
                "config": config_info
            }
        print()
        
        # Run evaluation on each protein
        print(f"Running evaluation on {len(chain_list)} proteins...")
        
        for i, chain_id in enumerate(tqdm(chain_list, desc="Evaluating proteins")):
            print(f"\nEvaluating {chain_id} ({i+1}/{len(chain_list)})")
            
            # Create features
            features = self._create_features(chain_id)
            
            # Get ground truth structure path
            structure_finder = EnhancedStructureFinder(
                str(self.pdb_dir),
                [".cif", ".pdb", ".core"],
                None
            )
            true_structure_path, _, _, _ = structure_finder.find_structure_path(chain_id)
            
            # Run inference for each model
            for model_name, model in self.models.items():
                # Clear GPU cache before inference
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Run inference
                outputs = self._run_inference(model, features)
                
                # Save predicted structure
                pred_pdb_path = self.results[model_name]["config"]["output_dir"] / f"{chain_id}.pdb"
                self._save_structure(outputs, chain_id, pred_pdb_path)
                
                # Extract pTM score
                ptm_score = self._extract_ptm_score(outputs)
                
                # Calculate TM-score against ground truth
                tm_score = self._calculate_tm_score(pred_pdb_path, Path(true_structure_path))
                
                # Store results
                self.results[model_name]["predictions"][chain_id] = str(pred_pdb_path)
                self.results[model_name]["ptm_scores"][chain_id] = ptm_score
                self.results[model_name]["tm_scores"][chain_id] = tm_score
                
        
        print("\nEvaluation completed!")
        
        # Generate analysis and plots
        self._analyze_results()
        self._create_plots()
        
        # Save results
        self._save_results()

    def _analyze_results(self):
        """Analyze the evaluation results"""
        
        print("\n=== Analysis Results ===")
        
        analysis = {}
        
        for model_name, results in self.results.items():
            ptm_scores = list(results["ptm_scores"].values())
            tm_scores = list(results["tm_scores"].values())
            
            if len(ptm_scores) == 0:
                continue
            
            # Basic statistics
            stats = {
                "n_proteins": len(ptm_scores),
                "mean_ptm": np.mean(ptm_scores),
                "std_ptm": np.std(ptm_scores),
                "mean_tm": np.mean(tm_scores),
                "std_tm": np.std(tm_scores),
                "median_ptm": np.median(ptm_scores),
                "median_tm": np.median(tm_scores),
            }
            
            # Correlation between pTM and TM-score
            if len(ptm_scores) > 1:
                pearson_r, pearson_p = pearsonr(ptm_scores, tm_scores)
                spearman_r, spearman_p = spearmanr(ptm_scores, tm_scores)
                
                stats.update({
                    "pearson_r": pearson_r,
                    "pearson_p": pearson_p,
                    "spearman_r": spearman_r,
                    "spearman_p": spearman_p,
                })
            
            analysis[model_name] = stats
            
            print(f"\n{results['config']['name']}:")
            print(f"  Proteins evaluated: {stats['n_proteins']}")
            print(f"  Mean pTM: {stats['mean_ptm']:.3f} ± {stats['std_ptm']:.3f}")
            print(f"  Mean TM-score: {stats['mean_tm']:.3f} ± {stats['std_tm']:.3f}")
            if 'pearson_r' in stats:
                print(f"  pTM-TM correlation (Pearson): {stats['pearson_r']:.3f} (p={stats['pearson_p']:.3e})")
                print(f"  pTM-TM correlation (Spearman): {stats['spearman_r']:.3f} (p={stats['spearman_p']:.3e})")
        
        self.analysis = analysis

    def _create_plots(self):
        """Create comprehensive analysis plots"""
            
        print("\nGenerating plots...")
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Overall performance comparison
        self._plot_performance_comparison()
        
        # 2. pTM vs TM-score correlation for each model
        self._plot_ptm_tm_correlation()
        
        # 3. Distribution plots
        self._plot_score_distributions()
        
        # 4. Model ranking analysis
        self._plot_ranking_analysis()
        
        print("  All plots saved to:", self.plots_dir)

    def _plot_performance_comparison(self):
        """Plot overall performance comparison between models"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        model_names = []
        ptm_means = []
        ptm_stds = []
        tm_means = []
        tm_stds = []
        
        for model_name, analysis in self.analysis.items():
            model_names.append(self.results[model_name]["config"]["name"])
            ptm_means.append(analysis["mean_ptm"])
            ptm_stds.append(analysis["std_ptm"])
            tm_means.append(analysis["mean_tm"])
            tm_stds.append(analysis["std_tm"])
        
        # pTM scores
        ax1.bar(model_names, ptm_means, yerr=ptm_stds, capsize=5, alpha=0.7)
        ax1.set_ylabel("pTM Score")
        ax1.set_title("Model Performance: pTM Scores")
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # TM-scores
        ax2.bar(model_names, tm_means, yerr=tm_stds, capsize=5, alpha=0.7)
        ax2.set_ylabel("TM-Score")
        ax2.set_title("Model Performance: TM-Scores")
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.savefig(self.plots_dir / "performance_comparison.pdf", bbox_inches='tight')
        plt.close()

    def _plot_ptm_tm_correlation(self):
        """Plot pTM vs TM-score correlation for each model"""
        
        n_models = len(self.results)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5))
        if n_models == 1:
            axes = [axes]
        
        for i, (model_name, results) in enumerate(self.results.items()):
            ax = axes[i]
            
            ptm_scores = list(results["ptm_scores"].values())
            tm_scores = list(results["tm_scores"].values())
            
            if len(ptm_scores) == 0:
                continue
            
            # Scatter plot
            ax.scatter(ptm_scores, tm_scores, alpha=0.6, s=50)
            
            # Fit line (only if we have variation in the data)
            if len(ptm_scores) > 1 and np.std(ptm_scores) > 1e-10 and np.std(tm_scores) > 1e-10:
                try:
                    z = np.polyfit(ptm_scores, tm_scores, 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(min(ptm_scores), max(ptm_scores), 100)
                    ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
                except np.linalg.LinAlgError:
                    pass  # Skip fitting if numerical issues
                
                # Add correlation info
                pearson_r = self.analysis[model_name].get("pearson_r", 0)
                if not np.isnan(pearson_r):
                    ax.text(0.05, 0.95, f"r = {pearson_r:.3f}", transform=ax.transAxes, 
                           bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
            
            ax.set_xlabel("pTM Score")
            ax.set_ylabel("TM-Score")
            ax.set_title(f"{results['config']['name']}")
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "ptm_tm_correlation.png", dpi=300, bbox_inches='tight')
        plt.savefig(self.plots_dir / "ptm_tm_correlation.pdf", bbox_inches='tight')
        plt.close()

    def _plot_score_distributions(self):
        """Plot score distributions for each model with mean and std in legend"""
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
        
        # Prepare data
        ptm_data = []
        tm_data = []
        labels = []
        ptm_stats = []
        tm_stats = []
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.results)))
        
        for i, (model_name, results) in enumerate(self.results.items()):
            ptm_scores = list(results["ptm_scores"].values())
            tm_scores = list(results["tm_scores"].values())
            
            if len(ptm_scores) == 0:
                continue
            
            ptm_data.append(ptm_scores)
            tm_data.append(tm_scores)
            labels.append(results["config"]["name"])
            
            # Calculate stats
            ptm_mean, ptm_std = np.mean(ptm_scores), np.std(ptm_scores)
            tm_mean, tm_std = np.mean(tm_scores), np.std(tm_scores)
            ptm_stats.append(f"{ptm_mean:.3f} ± {ptm_std:.3f}")
            tm_stats.append(f"{tm_mean:.3f} ± {tm_std:.3f}")
        
        # pTM distributions with histograms
        if ptm_data:
            for i, (data, label, stat) in enumerate(zip(ptm_data, labels, ptm_stats)):
                ax1.hist(data, bins=15, alpha=0.7, label=f"{label}: {stat}", 
                        color=colors[i], density=True)
            ax1.set_xlabel("pTM Score")
            ax1.set_ylabel("Density")
            ax1.set_title("pTM Score Distributions (mean ± std)")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # TM-score distributions with histograms  
        if tm_data:
            for i, (data, label, stat) in enumerate(zip(tm_data, labels, tm_stats)):
                ax2.hist(data, bins=15, alpha=0.7, label=f"{label}: {stat}",
                        color=colors[i], density=True)
            ax2.set_xlabel("TM-Score")
            ax2.set_ylabel("Density")
            ax2.set_title("TM-Score Distributions (mean ± std)")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "score_distributions.png", dpi=300, bbox_inches='tight')
        plt.savefig(self.plots_dir / "score_distributions.pdf", bbox_inches='tight')
        plt.close()

    def _plot_ranking_analysis(self):
        """Plot ranking analysis: how well do pTM scores rank structures"""
        
        # Combine all data for ranking analysis
        all_data = []
        
        for model_name, results in self.results.items():
            for chain_id in results["ptm_scores"]:
                all_data.append({
                    "model": results["config"]["name"],
                    "chain_id": chain_id,
                    "ptm": results["ptm_scores"][chain_id],
                    "tm": results["tm_scores"][chain_id]
                })
        
        if not all_data:
            return
        
        df = pd.DataFrame(all_data)
        
        # Create ranking comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Overall correlation heatmap
        ax1 = axes[0, 0]
        corr_matrix = []
        model_names = df['model'].unique()
        
        for model1 in model_names:
            row = []
            for model2 in model_names:
                # Get common proteins
                model1_data = df[df['model'] == model1]
                model2_data = df[df['model'] == model2]
                common_chains = set(model1_data['chain_id']) & set(model2_data['chain_id'])
                
                if len(common_chains) > 1:
                    m1_tm = [model1_data[model1_data['chain_id'] == c]['tm'].iloc[0] for c in common_chains]
                    m2_tm = [model2_data[model2_data['chain_id'] == c]['tm'].iloc[0] for c in common_chains]
                    corr = np.corrcoef(m1_tm, m2_tm)[0, 1]
                else:
                    corr = 0
                row.append(corr)
            corr_matrix.append(row)
        
        sns.heatmap(corr_matrix, annot=True, xticklabels=model_names, yticklabels=model_names, 
                   cmap='coolwarm', center=0, ax=ax1)
        ax1.set_title("TM-Score Correlation Between Models")
        
        # 2. pTM-TM correlation by model
        ax2 = axes[0, 1]
        for model in model_names:
            model_data = df[df['model'] == model]
            if len(model_data) > 1:
                ax2.scatter(model_data['ptm'], model_data['tm'], label=model, alpha=0.6, s=30)
        
        ax2.set_xlabel("pTM Score")
        ax2.set_ylabel("TM-Score")
        ax2.set_title("pTM vs TM-Score by Model")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Ranking quality assessment
        ax3 = axes[1, 0]
        ranking_correlations = []
        
        for model in model_names:
            model_data = df[df['model'] == model]
            if len(model_data) > 1:
                # Calculate ranking correlation (Spearman)
                spearman_r, _ = spearmanr(model_data['ptm'], model_data['tm'])
                ranking_correlations.append(spearman_r)
            else:
                ranking_correlations.append(0)
        
        bars = ax3.bar(model_names, ranking_correlations, alpha=0.7)
        ax3.set_ylabel("Spearman Correlation (pTM vs TM-score)")
        ax3.set_title("Ranking Quality by Model")
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, corr in zip(bars, ranking_correlations):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{corr:.3f}', ha='center', va='bottom')
        
        # 4. Performance summary table
        ax4 = axes[1, 1]
        ax4.axis('tight')
        ax4.axis('off')
        
        # Create summary table
        summary_data = []
        for model in model_names:
            model_data = df[df['model'] == model]
            if len(model_data) > 0:
                summary_data.append([
                    model,
                    len(model_data),
                    f"{model_data['ptm'].mean():.3f}",
                    f"{model_data['tm'].mean():.3f}",
                    f"{ranking_correlations[list(model_names).index(model)]:.3f}"
                ])
        
        table = ax4.table(cellText=summary_data,
                         colLabels=['Model', 'N', 'Mean pTM', 'Mean TM', 'Rank Corr'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax4.set_title("Performance Summary")
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "ranking_analysis.png", dpi=300, bbox_inches='tight')
        plt.savefig(self.plots_dir / "ranking_analysis.pdf", bbox_inches='tight')
        plt.close()

    def _save_results(self):
        """Save detailed results to files"""
        
        print("\nSaving results...")
        
        # Save detailed results as JSON
        results_for_json = {}
        for model_name, results in self.results.items():
            results_for_json[model_name] = {
                "config_name": results["config"]["name"],
                "ptm_scores": results["ptm_scores"],
                "tm_scores": results["tm_scores"],
                "analysis": self.analysis.get(model_name, {})
            }
        
        with open(self.analysis_dir / "detailed_results.json", 'w') as f:
            json.dump(results_for_json, f, indent=2)
        
        # Save summary as CSV
        summary_data = []
        all_chains = set()
        for results in self.results.values():
            all_chains.update(results["ptm_scores"].keys())
        
        for chain_id in sorted(all_chains):
            row = {"chain_id": chain_id}
            for model_name, results in self.results.items():
                model_short = model_name
                if chain_id in results["ptm_scores"]:
                    row[f"{model_short}_ptm"] = results["ptm_scores"][chain_id]
                    row[f"{model_short}_tm"] = results["tm_scores"][chain_id]
                else:
                    row[f"{model_short}_ptm"] = None
                    row[f"{model_short}_tm"] = None
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(self.analysis_dir / "summary_results.csv", index=False)
        
        # Save analysis summary
        with open(self.analysis_dir / "analysis_summary.txt", 'w') as f:
            f.write("OpenFold Model Architecture Evaluation Results\n")
            f.write("=" * 50 + "\n\n")
            
            for model_name, analysis in self.analysis.items():
                f.write(f"{self.results[model_name]['config']['name']}:\n")
                f.write(f"  Proteins evaluated: {analysis['n_proteins']}\n")
                f.write(f"  Mean pTM: {analysis['mean_ptm']:.3f} ± {analysis['std_ptm']:.3f}\n")
                f.write(f"  Mean TM-score: {analysis['mean_tm']:.3f} ± {analysis['std_tm']:.3f}\n")
                if 'pearson_r' in analysis:
                    f.write(f"  pTM-TM correlation (Pearson): {analysis['pearson_r']:.3f}\n")
                    f.write(f"  pTM-TM correlation (Spearman): {analysis['spearman_r']:.3f}\n")
                f.write("\n")
        
        print(f"  Detailed results: {self.analysis_dir / 'detailed_results.json'}")
        print(f"  Summary CSV: {self.analysis_dir / 'summary_results.csv'}")
        print(f"  Analysis summary: {self.analysis_dir / 'analysis_summary.txt'}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate OpenFold model architectures",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input data
    parser.add_argument(
        "--csv_path", type=str, required=True,
        help="Path to CSV file containing chain list (relative to home directory)"
    )
    parser.add_argument(
        "--pdb_dir", type=str, required=True,
        help="Directory containing PDB/mmCIF ground truth files (relative to home directory)"
    )
    
    # Model weights
    parser.add_argument(
        "--original_weights", type=str, required=True,
        help="Path to original pretrained weights (relative to home directory)"
    )
    parser.add_argument(
        "--finetuned_weights", type=str, default=None,
        help="Path to fine-tuned weights (relative to home directory)"
    )
    
    # Model architecture parameters
    parser.add_argument(
        "--replace_block_index", type=int, required=True,
        help="Index of Evoformer block that was replaced/removed (0-based)"
    )
    parser.add_argument(
        "--replacement_hidden_dim", type=int, default=256,
        help="Hidden dimension used in replacement block"
    )
    
    # Output
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for predictions and analysis (relative to home directory)"
    )
    
    # Evaluation parameters
    parser.add_argument(
        "--max_proteins", type=int, default=None,
        help="Maximum number of proteins to evaluate (None for all)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.replace_block_index <= 0 or args.replace_block_index >= 47:
        raise ValueError("replace_block_index must be between 1 and 46 (not first/last block)")
    
    # Run evaluation
    evaluator = ModelArchitectureEvaluator(args)
    evaluator.evaluate_models()
    
    print("\n=== Evaluation Complete! ===")
    print(f"Results saved to: {evaluator.output_dir}")
    print(f"Predictions: {evaluator.predictions_dir}")
    print(f"Analysis: {evaluator.analysis_dir}")
    print(f"Plots: {evaluator.plots_dir}")


if __name__ == "__main__":
    main()
