#!/usr/bin/env python3
"""
Simplified feature utilities for OpenFold model evaluation.

This module provides streamlined feature creation for protein structure prediction
evaluation, using the same approach as the OpenFold inference notebooks with
single sequence mode (no MSAs, no templates).
"""

import os
import torch
import numpy as np
from typing import Dict, Tuple, Any
from pathlib import Path

from openfold.data.mmcif_parsing import parse as parse_mmcif
from openfold.np import residue_constants


def extract_sequence_from_structure(file_path: str, chain_id: str = None) -> str:
    """Extract protein sequence from structure file"""
    
    file_path = Path(file_path)
    
    if file_path.suffix.lower() == '.cif':
        # Parse mmCIF file
        with open(file_path, 'r') as f:
            mmcif_string = f.read()
        
        parsing_result = parse_mmcif(file_id=file_path.stem, mmcif_string=mmcif_string)
        mmcif_object = parsing_result.mmcif_object
        
        # Get sequence for specified chain or first available
        if chain_id and chain_id in mmcif_object.chain_to_seqres:
            sequence = mmcif_object.chain_to_seqres[chain_id]
        else:
            # Get first available sequence
            sequences = list(mmcif_object.chain_to_seqres.values())
            if sequences:
                sequence = sequences[0]
            else:
                raise ValueError(f"No sequences found in {file_path}")
    
    elif file_path.suffix.lower() == '.pdb':
        # For PDB files, parse ATOM records
        sequence = parse_pdb_sequence(file_path, chain_id)
    
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    if not sequence:
        raise ValueError(f"Could not extract sequence from {file_path}")
    
    # Clean sequence (remove unknown residues)
    cleaned_sequence = ''.join([res for res in sequence if res in residue_constants.restype_order])
    
    if not cleaned_sequence:
        raise ValueError(f"No valid amino acids found in sequence from {file_path}")
    
    return cleaned_sequence


def parse_pdb_sequence(file_path: str, chain_id: str = None) -> str:
    """Parse sequence from PDB file (simplified)"""
    
    residues = []
    seen_residues = set()
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM') and line[12:16].strip() == 'CA':  # CA atoms only
                chain = line[21]
                if chain_id is None or chain == chain_id:
                    res_num = int(line[22:26])
                    res_type = line[17:20].strip()
                    
                    # Convert 3-letter to 1-letter code
                    if res_type in residue_constants.restype_3to1:
                        res_1letter = residue_constants.restype_3to1[res_type]
                        
                        # Avoid duplicates (same residue number)
                        if (chain, res_num) not in seen_residues:
                            residues.append((res_num, res_1letter))
                            seen_residues.add((chain, res_num))
    
    # Sort by residue number and extract sequence
    residues.sort()
    sequence = ''.join([res[1] for res in residues])
    
    return sequence


def create_single_sequence_features(sequence: str) -> Dict[str, torch.Tensor]:
    """Create features for single sequence mode (no MSAs, no templates)"""
    
    seq_len = len(sequence)
    
    # Convert sequence to aatype
    aatype = np.array([residue_constants.restype_order.get(res, 0) for res in sequence])
    
    # Create minimal feature set required for single sequence inference
    # Based on the OpenFold inference pipeline
    
    # Basic sequence features
    features = {
        # Target sequence
        "aatype": aatype,
        "residue_index": np.arange(seq_len),
        "seq_length": np.array([seq_len]),
        "seq_mask": np.ones(seq_len),
        "between_segment_residues": np.zeros(seq_len),
        
        # MSA features (single sequence MSA)
        "msa": aatype[None],  # Shape: (1, seq_len) - single sequence MSA
        "msa_feat": np.eye(25)[aatype[None]],  # One-hot encoded MSA features (25 MSA alphabet)
        "deletion_matrix": np.zeros((1, seq_len)),
        "msa_mask": np.ones((1, seq_len)),
        "msa_row_mask": np.ones(1),
        
        # Template features (empty)
        "template_aatype": np.zeros((1, seq_len)),
        "template_all_atom_positions": np.zeros((1, seq_len, 37, 3)),
        "template_all_atom_mask": np.zeros((1, seq_len, 37)),
        "template_sum_probs": np.zeros((1, seq_len)),
        "template_pseudo_beta": np.zeros((1, seq_len, 3)),
        "template_pseudo_beta_mask": np.zeros((1, seq_len)),
        
        # Pair features  
        "pair_mask": np.ones((seq_len, seq_len)),
        
        # Additional required features
        "no_recycling_iters": np.array([0]),
        "target_feat": np.eye(21)[aatype],  # One-hot encoding
    }
    
    # Convert to tensors and add proper dimensions
    tensor_features = {}
    
    for key, value in features.items():
        # Use appropriate dtype based on the feature type
        if key in ["aatype", "residue_index", "seq_length", "no_recycling_iters", 
                   "between_segment_residues", "msa", "seq_mask"]:
            # Integer or mask features
            tensor_value = torch.tensor(value, dtype=torch.long if key in ["aatype", "residue_index", "seq_length", "no_recycling_iters", "msa"] else torch.float32)
        else:
            # Float features - use float32 for model compatibility
            tensor_value = torch.tensor(value, dtype=torch.float32)
        
        # Add batch dimension
        if key == "seq_length":
            # seq_length stays as is
            tensor_features[key] = tensor_value
        elif key == "no_recycling_iters":
            # no_recycling_iters stays as is
            tensor_features[key] = tensor_value
        else:
            # Add batch dimension for all other features
            tensor_value = tensor_value.unsqueeze(0)
            tensor_features[key] = tensor_value
    
    return tensor_features


def features_from_chain_id(chain_id: str, pdb_dir: str) -> Dict[str, torch.Tensor]:
    """Create features for a protein chain given its ID and structure directory"""
    
    from openfold.block_replacement_scripts.enhanced_data_utils import EnhancedStructureFinder
    
    # Find structure file
    structure_finder = EnhancedStructureFinder(
        pdb_dir,
        [".cif", ".pdb", ".core"],
        None
    )
    
    file_path, pdb_id, chain, ext = structure_finder.find_structure_path(chain_id)
    
    # Extract sequence
    sequence = extract_sequence_from_structure(file_path, chain)
    
    # Create features using single sequence mode
    features = create_single_sequence_features(sequence)
    
    # Add metadata
    features["domain_name"] = chain_id
    features["sequence"] = sequence
    
    return features


def test_feature_creation():
    """Test the feature creation pipeline"""
    
    # Test with a simple sequence
    test_sequence = "MKLLISGLATLLLAHCEQGVEPSNHPWWRCCPFSYTARHNKDYWRNWEESRPWQRRLEQQLFLKGTFRYNRRAQRQTQPICSIISEHQTLQCAEPLRHQVFDTEELVDAAKRAGQTQHTDPKGCFICSLMLKKSPNRVEELRCYCPQTGQLGGCKTPAKPDIVKMGWLRKHNSYDWLNEQDNEQLMDQLKRQMEQQLNEQTTAKLEAAITEIEKSQARKFSQARLQNLLHELAALLREICGPQAQTPQHTDPKGNFICESEIYKSPNRVAELRCYCSQVSQLGGCKTPAKPDIRFMGWLRTQHNYGWLNEVQKKQLNGLVRDIEKQIEAQITEIEKSQARKFTQAQLQNTLLDLAALLEQICQSTQKESTKYDKICGFICDTDVYKTPNRVSRLRCQCPQTGQLGGCKTPAKPDIIKGVWLQRQHSYGWLNQVEIKQINNAIKDITEKEMEQSVARKMTKARLQKMLQELANLLEDICSTKHNQGTHADKIGSFICESIMYKSPNRVEELKCYCRPKGQLGGCKTPAKPDIRKDGWLKLRNYGWLNEVQMKQINEAIQDVEKAMEQTIEKLEKAKMQAKLLENLHQLQGALLEQICRSNHNQGTTADKIGNFICDSIIYKSPNRVEQLQCYCRPVGQLGGCKTPAKPDIIKDGWLLQRFGYGWLNEVDLKQINKQALSELQTANRQSIEKREALLEQICRKKHNQGMIADKIGAFICESIIYKSPNRVEELQCYCRTVGQLGGCKTPAKPDIVKDGWLIQRYGYGWLNEVGIQEINQRALKEMEQSVRQRTERARDQILKGIRQRINSLLKEIAELLCEILRQDQGQCADFCCQIWIWNCLLLLRFKEKNYDGTFQSYEEEPEPEPFRCYCKEEFRRLQNYAELRSYQSVDKSSKSIEYNEGDSCSCSWSMCNIDSVLRAIKCLPFDKSLPRWEKIAKQYFFNVVSRFTIDPSLRSLFKSYIKREPPSKPELFRNYYFNVSKRFILGQQDCDDFVPPMAQNLFISQYIKREPPSKDGIFRSFYYNLAQDTFWMLPQCDSSSEEPELISSCYIKDSPPSNPMIFRSLAFKNKAQKSSWLYPQCDDEDRAEEKFRCCFINKEPPSNPLVFRSLEYKSAQKTLWLWPQFEEEEPFQEHLLSYSRYMDAPPSHPFLQGSFLSVYKSQPPCSLKLKEHNPCNIFWNYTILKMPPSKPPLLYSYFNPILNKSSWKLSTQCDDGFQTNPGFICCYTRDNPPHNSVLYGSLGPSSKYNLKEPPPKKPLLFQSYFNTVLQNEQLNEQMDQQLQNGFLGRPFLGDVCQDLTAKVDQSLLKVTLQELQLKTQKSKPDEYYIYLLNRPWNVQDSFLKYAKDLKKMRPPKPEFFISNFLNLSQMDCLFKLLVPVCDEEKSICCFIMKHPYSNPALYVSLGSNEKYKLAEPPPNKPLLYTSYFNKEGQDDLLNEQAKQSQTDFIGRQSYGNVCDDLTTLLQEIKRNTTSKDDNFYISGLNRPWNMQESLLTFVNTTNDSKMKPLVMQRFVQSKSSMDLLKKACQLRDEKAMDKFLNQPELLHGNASFLKLLNEVCQNTTQSNLNQIGQAYFDSFGKEEMLGNFCDDLNTTNQSLIDQIKEGSLDLLNEGCQEYSDSNLQQYFFDRPGRTRVSDSKAQLSRKLNQPLLDQFGRQYFHNVCQDLNTTVQRLIDQTKQSCLDFMNSGTQAYISNNLGQKFFFDRPRRYRTAAQIQRYLKQPVLNQFGRRSYSNVCDDLGQNLKEQIMKSKFHYYTGMDNLGSQIGRYLSQTNGDNGNSFYYSYTFDDNQFCDDLTLTIIQNYIDQATYSLFSSSLKRYLPQMSIHQFGRAAFNNISDSLGQSRQGILDRRTLQQVFHSHTSNSNQIHNIIHQDGKYNFLHTYVGRNNNKQGMKQSEVCSYFCSQPQEGLQNYQYDRVNYEDVRTICDSLNYCPPQASQGRYHFLQSFYFLGKYFPYDNVAYVRRYGQYAAYRVLTPFFGCSFEENIHSNDSNEGTFLGKYFLHGKYYPDRVTGDVGRYFANLLGVAYFYDGGSKDGLGFRYSYFLHGKYFEGEVTYKVGTFCPSLLGVAYLCSGGNKNDLGFRIAFFLHGNYYGSNVAYDVGNYSDLVALKAYKQGGNKNRYVFKYSYFLHGKSFPFTMPKTKKLMKYTFPELEALIKQKQAIEFLRQGEIQPEFLVKLLIRQLHRDIFQFLRQGDFQDEFLRQLLIRQLHRDLQGFLRQGNIQDKLLNYLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGNIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDEFLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRDLQGFLRQGEIQDELLRQLLIRQLHRD"
    
    print(f"Test sequence length: {len(test_sequence)}")
    
    # Create features
    features = create_single_sequence_features(test_sequence)
    
    print("Created features:")
    for key, value in features.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape} {value.dtype}")
        else:
            print(f"  {key}: {type(value)}")
    
    print("\nFeature creation test completed successfully!")


if __name__ == "__main__":
    test_feature_creation()
