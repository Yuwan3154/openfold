"""PDA (Protein Design Archive) single-sequence validation dataset.

Feeds real de novo protein designs (sequence + native structure) through OpenFold's OWN existing
feature-construction machinery -- make_mmcif_features (ground truth) + make_dummy_msa_feats
(single-sequence "MSA") + empty template features, via DataPipeline.process_mmcif(seqemb_mode=True,
alignment_dir=<nonexistent path>). Both _parse_template_hit_files and _process_seqemb_features in
openfold/data/data_pipeline.py explicitly handle a nonexistent alignment_dir ("case where the
alignment directory doesn't exist (e.g., single sequence mode)") by returning empty results -- this
is the existing, tested OpenFold code path for exactly this scenario, not a new mechanism. Produces
batches structurally identical to a normal training/eval example (same feature_pipeline, same
cropping/recycling config), just sourced from PDA instead of openproteinset_aln-backed PDB chains,
and without needing any precomputed alignment/template search (PDA entries have none).
"""
import json
import os

import torch

from openfold.data import data_pipeline, feature_pipeline, mmcif_parsing


class PDASingleSeqDataset(torch.utils.data.Dataset):
    """manifest_path: JSON list of {"pdb": str, "chain_id": str, "seq": str} (Foldseek-clustered
    representative set). cif_cache_dir: directory of already-fetched {pdb}.cif files. config: the
    SAME config.data object OpenFoldDataModule/OpenFoldSingleDataset use. mode: "eval" (matches
    OpenFoldSingleDataset's mode argument -- controls cropping/recycling-count sampling)."""

    def __init__(self, manifest_path, cif_cache_dir, config, mode="eval"):
        with open(manifest_path) as f:
            self.manifest = json.load(f)
        self.cif_cache_dir = cif_cache_dir
        self.config = config
        self.mode = mode
        # template_featurizer=None -> make_template_features always takes the empty-template
        # branch (data_pipeline.py: `if template_featurizer is None or (len(hits_cat)==0 ...)`),
        # so no template mmcif dir / release-dates cache is needed for this template-free path.
        self.data_pipeline = data_pipeline.DataPipeline(template_featurizer=None)
        self.feature_pipeline = feature_pipeline.FeaturePipeline(config)

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        entry = self.manifest[idx]
        pdbid, chain_id = entry["pdb"], entry["chain_id"]
        cif_path = os.path.join(self.cif_cache_dir, f"{pdbid}.cif")
        with open(cif_path) as f:
            mmcif_string = f.read()
        parsed = mmcif_parsing.parse(file_id=pdbid, mmcif_string=mmcif_string)
        mmcif_object = parsed.mmcif_object
        if mmcif_object is None or chain_id not in mmcif_object.chain_to_seqres:
            raise ValueError(f"{pdbid}_{chain_id}: mmcif parse failed or chain missing")

        data = self.data_pipeline.process_mmcif(
            mmcif=mmcif_object,
            alignment_dir=f"/nonexistent_pda_single_seq/{pdbid}_{chain_id}",
            chain_id=chain_id,
            alignment_index=None,
            seqemb_mode=True,
        )
        feats = self.feature_pipeline.process_features(data, self.mode)
        feats["batch_idx"] = torch.tensor(
            [idx for _ in range(feats["aatype"].shape[-1])], dtype=torch.int64)
        return feats
