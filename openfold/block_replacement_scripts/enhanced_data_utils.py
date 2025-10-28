"""
Enhanced data utilities for recursive directory search and chain-specific training
"""
import os
import glob
from typing import Dict, List, Optional, Set

try:
    from pytorch_lightning.utilities.rank_zero import rank_zero_info
except ImportError:
    # Fallback if pytorch_lightning is not available
    def rank_zero_info(msg):
        print(msg)

def find_structure_file_recursive(data_dir: str, file_id: str, supported_exts: List[str]) -> Optional[str]:
    """
    Recursively search for a structure file in the data directory
    
    Args:
        data_dir: Root directory to search in
        file_id: File ID to search for (e.g., "1abc")
        supported_exts: List of supported extensions (e.g., [".cif", ".core", ".pdb"])
    
    Returns:
        Full path to the found file, or None if not found
    """
    for ext in supported_exts:
        # Use glob to search recursively for the file
        pattern = os.path.join(data_dir, "**", f"{file_id}{ext}")
        matches = glob.glob(pattern, recursive=True)
        
        if matches:
            # Return the first match
            return matches[0]
    
    return None


def load_chain_training_list(chain_list_path: str) -> Dict[str, str]:
    """
    Load a training list with chain specifications
    
    Args:
        chain_list_path: Path to text file containing entries like "1abc_A"
    
    Returns:
        Dictionary mapping chain names to their components
        e.g., {"1abc_A": {"file_id": "1abc", "chain_id": "A"}}
    """
    chain_mapping = {}
    
    with open(chain_list_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            # Handle both "1abc_A" and "1abc" formats
            if '_' in line:
                file_id, chain_id = line.rsplit('_', 1)
            else:
                file_id = line
                chain_id = None
            
            chain_mapping[line] = {
                "file_id": file_id,
                "chain_id": chain_id
            }
    
    return chain_mapping


def get_structure_files_recursive(data_dir: str, supported_exts: List[str]) -> Dict[str, str]:
    """
    Get all structure files in a directory recursively
    
    Args:
        data_dir: Root directory to search
        supported_exts: List of supported extensions
    
    Returns:
        Dictionary mapping file_id to full path
    """
    structure_files = {}
    
    for ext in supported_exts:
        pattern = os.path.join(data_dir, "**", f"*{ext}")
        matches = glob.glob(pattern, recursive=True)
        
        for file_path in matches:
            # Extract file ID from filename
            filename = os.path.basename(file_path)
            file_id = os.path.splitext(filename)[0]
            
            # Store the mapping (first occurrence wins if duplicates)
            if file_id not in structure_files:
                structure_files[file_id] = file_path
    
    return structure_files


def build_chain_ids_from_structures_and_list(
    data_dir: str, 
    supported_exts: List[str],
    chain_list_path: Optional[str] = None,
    alignment_dir: Optional[str] = None
) -> List[str]:
    """
    Build chain IDs either from a provided list or by discovering from structures
    
    Args:
        data_dir: Directory containing structure files
        supported_exts: List of supported file extensions
        chain_list_path: Optional path to chain list file
        alignment_dir: Optional alignment directory (for backward compatibility)
    
    Returns:
        List of chain IDs to use for training
    """
    if chain_list_path and os.path.exists(chain_list_path):
        # Load from provided chain list
        chain_mapping = load_chain_training_list(chain_list_path)
        chain_ids = list(chain_mapping.keys())
        rank_zero_info(f"Loaded {len(chain_ids)} chains from {chain_list_path}")
        return chain_ids
    
    elif alignment_dir and os.path.exists(alignment_dir):
        # Use existing alignment directory logic (backward compatibility)
        chain_ids = list(os.listdir(alignment_dir))
        rank_zero_info(f"Using {len(chain_ids)} chains from alignment directory")
        return chain_ids
    
    else:
        # Discover from structure files
        structure_files = get_structure_files_recursive(data_dir, supported_exts)
        chain_ids = []
        
        for file_id in structure_files.keys():
            # For auto-discovery, we assume single chain or use default chain naming
            # Could be enhanced to parse actual chains from structures
            chain_ids.append(file_id)
        
        rank_zero_info(f"Auto-discovered {len(chain_ids)} structures in {data_dir}")
        return chain_ids


class EnhancedStructureFinder:
    """
    Enhanced structure file finder with recursive search and chain list support
    """
    
    def __init__(self, data_dir: str, supported_exts: List[str], chain_list_path: Optional[str] = None):
        self.data_dir = data_dir
        self.supported_exts = supported_exts
        self.chain_list_path = chain_list_path
        
        # Build file mapping for faster lookups
        self.structure_files = get_structure_files_recursive(data_dir, supported_exts)
        # For backward compatibility, alias structure_map to structure_files
        self.structure_map = self.structure_files
        
        # Load chain mapping if provided
        self.chain_mapping = {}
        if chain_list_path and os.path.exists(chain_list_path):
            self.chain_mapping = load_chain_training_list(chain_list_path)
            rank_zero_info(f"Loaded chain mapping for {len(self.chain_mapping)} entries")
        
        # Load label->author chain ID mapping if available
        self.label_to_author_mapping = {}
        chain_id_mapping_path = os.path.join(os.path.dirname(data_dir), 'chain_id_mapping.json')
        if os.path.exists(chain_id_mapping_path):
            import json
            with open(chain_id_mapping_path, 'r') as f:
                self.label_to_author_mapping = json.load(f)
            rank_zero_info(f"Loaded label->author chain ID mapping for {len(self.label_to_author_mapping)} files")
    
    def find_structure_path(self, name: str) -> tuple:
        """
        Find structure path and extract chain information
        
        Args:
            name: Chain name (e.g., "1abc_A" or "1abc")
        
        Returns:
            Tuple of (file_path, file_id, chain_id, ext)
        """
        # Check if we have explicit chain mapping
        if name in self.chain_mapping:
            file_id = self.chain_mapping[name]["file_id"]
            chain_id = self.chain_mapping[name]["chain_id"]
        else:
            # Fall back to old logic
            spl = name.rsplit('_', 1)
            if len(spl) == 2:
                file_id, chain_id = spl
            else:
                file_id, = spl
                chain_id = None
        
        # Translate label chain ID to author chain ID if mapping exists
        # This handles cases where mmCIF uses standardized chain IDs (label_asym_id)
        # but OpenFold parser uses author chain IDs (auth_asym_id)
        if chain_id and file_id in self.label_to_author_mapping:
            if chain_id in self.label_to_author_mapping[file_id]:
                author_chain_id = self.label_to_author_mapping[file_id][chain_id]
                # Use author chain ID for OpenFold's parser
                chain_id = author_chain_id
        
        # Find the structure file
        if file_id in self.structure_files:
            file_path = self.structure_files[file_id]
            ext = os.path.splitext(file_path)[1]
            return file_path, file_id, chain_id, ext
        else:
            # Fall back to old non-recursive search
            path = os.path.join(self.data_dir, file_id)
            for ext in self.supported_exts:
                if os.path.exists(path + ext):
                    return path + ext, file_id, chain_id, ext
            
            raise ValueError(f"Could not find structure file for {file_id} in {self.data_dir}")



