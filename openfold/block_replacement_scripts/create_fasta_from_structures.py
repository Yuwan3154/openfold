#!/usr/bin/env python3
"""
Utility script to create FASTA files from protein structure files.

This helps prepare data for representation distillation training by extracting
sequences from PDB/mmCIF files.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple
import pandas as pd

# Add openfold to path
sys.path.append(str(Path.home() / 'openfold'))

from openfold.data.mmcif_parsing import parse as parse_mmcif
from openfold.np import residue_constants

# Import enhanced data utils
sys.path.append(str(Path(__file__).parent))
from enhanced_data_utils import EnhancedStructureFinder


def extract_sequence_from_mmcif(mmcif_path: str, chain_id: str = None) -> Tuple[str, str]:
    """
    Extract sequence from mmCIF file
    
    Returns:
        (sequence, chain_id): Extracted sequence and chain ID used
    """
    with open(mmcif_path, 'r') as f:
        mmcif_string = f.read()
    
    parsing_result = parse_mmcif(file_id=Path(mmcif_path).stem, mmcif_string=mmcif_string)
    mmcif_object = parsing_result.mmcif_object
    
    # Get sequence for specified chain or first available
    if chain_id and chain_id in mmcif_object.chain_to_seqres:
        sequence = mmcif_object.chain_to_seqres[chain_id]
        used_chain = chain_id
    else:
        # Get first available sequence
        sequences = list(mmcif_object.chain_to_seqres.items())
        if sequences:
            used_chain, sequence = sequences[0]
        else:
            raise ValueError(f"No sequences found in {mmcif_path}")
    
    # Clean sequence (remove unknown residues)
    cleaned_sequence = ''.join([res for res in sequence if res in residue_constants.restype_order])
    
    if not cleaned_sequence:
        raise ValueError(f"No valid amino acids found in sequence from {mmcif_path}")
    
    return cleaned_sequence, used_chain


def extract_sequence_from_pdb(pdb_path: str, chain_id: str = None) -> Tuple[str, str]:
    """
    Extract sequence from PDB file
    
    Returns:
        (sequence, chain_id): Extracted sequence and chain ID used
    """
    residues = []
    seen_residues = set()
    used_chain = None
    
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM') and line[12:16].strip() == 'CA':  # CA atoms only
                chain = line[21]
                if chain_id is None or chain == chain_id:
                    if used_chain is None:
                        used_chain = chain
                    
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
    
    if not sequence:
        raise ValueError(f"No sequence extracted from {pdb_path}")
    
    return sequence, used_chain


def create_fasta_from_csv(
    csv_path: str,
    pdb_dir: str,
    output_fasta: str,
    max_length: int = None,
    min_length: int = None,
):
    """
    Create FASTA file from CSV file with chain list
    
    Args:
        csv_path: Path to CSV with 'natives_rcsb' column
        pdb_dir: Directory containing structure files
        output_fasta: Output FASTA file path
        max_length: Maximum sequence length to include
        min_length: Minimum sequence length to include
    
    Note: Train/val split should be done in the training script, not here.
    """
    # Read CSV
    df = pd.read_csv(csv_path)
    all_chains = df['natives_rcsb'].dropna().tolist()
    
    print(f"Found {len(all_chains)} chains in CSV")
    
    # Setup structure finder
    structure_finder = EnhancedStructureFinder(
        pdb_dir,
        [".cif", ".pdb", ".core"],
        None
    )
    
    # Extract sequences
    sequences = []
    failed = []
    
    for chain_spec in all_chains:
        try:
            # Find structure file
            structure_path, file_id, chain_id, ext = structure_finder.find_structure_path(chain_spec)
            
            # Extract sequence
            if ext in ['.cif', '.core']:
                sequence, used_chain = extract_sequence_from_mmcif(structure_path, chain_id)
            else:
                sequence, used_chain = extract_sequence_from_pdb(structure_path, chain_id)
            
            # Filter by length
            if max_length and len(sequence) > max_length:
                continue
            if min_length and len(sequence) < min_length:
                continue
            
            sequences.append({
                'id': chain_spec,
                'sequence': sequence,
                'length': len(sequence)
            })
            
        except Exception as e:
            failed.append((chain_spec, str(e)))
    
    print(f"Successfully extracted {len(sequences)} sequences")
    if failed:
        print(f"Failed to extract {len(failed)} sequences")
    
    # Write all sequences to single file
    _write_fasta(sequences, output_fasta)
    print(f"Wrote {len(sequences)} sequences to: {output_fasta}")
    
    # Print statistics
    lengths = [s['length'] for s in sequences]
    print(f"\nSequence length statistics:")
    print(f"  Min: {min(lengths)}")
    print(f"  Max: {max(lengths)}")
    print(f"  Mean: {sum(lengths) / len(lengths):.1f}")
    print(f"  Median: {sorted(lengths)[len(lengths)//2]}")


def _write_fasta(sequences: List[dict], output_path: str):
    """Write sequences to FASTA file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        for seq_dict in sequences:
            f.write(f">{seq_dict['id']}\n")
            # Write sequence in 80-character lines
            sequence = seq_dict['sequence']
            for i in range(0, len(sequence), 80):
                f.write(sequence[i:i+80] + '\n')


def main():
    parser = argparse.ArgumentParser(
        description="Create FASTA files from protein structure files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--csv_path', type=str, required=True,
                       help='Path to CSV file with chain list')
    parser.add_argument('--pdb_dir', type=str, required=True,
                       help='Directory containing structure files')
    parser.add_argument('--output_fasta', type=str, required=True,
                       help='Output FASTA file path')
    parser.add_argument('--max_length', type=int, default=None,
                       help='Maximum sequence length to include')
    parser.add_argument('--min_length', type=int, default=None,
                       help='Minimum sequence length to include')
    args = parser.parse_args()
    
    # Convert to absolute paths
    csv_path = str(Path(args.csv_path).expanduser().resolve())
    pdb_dir = str(Path(args.pdb_dir).expanduser().resolve())
    output_fasta = str(Path(args.output_fasta).expanduser().resolve())
    
    create_fasta_from_csv(
        csv_path=csv_path,
        pdb_dir=pdb_dir,
        output_fasta=output_fasta,
        max_length=args.max_length,
        min_length=args.min_length,
    )


if __name__ == '__main__':
    main()

