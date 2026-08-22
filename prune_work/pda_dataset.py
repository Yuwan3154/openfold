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

    def __init__(self, manifest_path, cif_cache_dir, config, mode="eval", train_overlap_ids_path=None,
                 source_tag=0, index_offset=0, nonneural_ids_path=None):
        with open(manifest_path) as f:
            self.manifest = json.load(f)
        self.cif_cache_dir = cif_cache_dir
        self.config = config
        self.mode = mode
        # ⛔ `index_offset` exists because per_entry_val_history.csv is keyed on `batch_idx` ALONE.
        # Under a ConcatDataset each member would otherwise restart its indices at 0 and three
        # populations would collide on the same key, silently interleaving different chains in one
        # row group. The offset makes batch_idx globally unique across the combined validation set.
        # `source_tag` rides along per entry so val metrics can be split by population without the
        # analysis having to re-derive membership from index arithmetic.
        self.source_tag = int(source_tag)
        self.index_offset = int(index_offset)
        # Entries whose pdb_chain is verbatim present in the model's own training set (see
        # ESMFOLD2_RECYCLE_SCALING.md PDA investigation) -- kept IN the validation population as a
        # deliberate "has the model learned its own training data" marker, not filtered out. Empty
        # set (default) means every item reports as held-out, i.e. no behavior change when unset.
        self.train_overlap_ids = set()
        if train_overlap_ids_path is not None:
            with open(train_overlap_ids_path) as f:
                self.train_overlap_ids = {
                    f"{e['pdb']}_{e['chain_id']}" for e in json.load(f)
                }
        # Entries whose paper names NO neural structure predictor (AF2/ColabFold/RoseTTAFold/
        # ESMFold/trRosetta) anywhere in its design protocol -- the CIRCULARITY-FREE subset. The PDA
        # pass rate is inflated where AF2 was itself a design-acceptance gate (42% of passes vs 15%
        # of failures, p=1.0e-05), which biases the benchmark AGAINST us on exactly the modern
        # designs; on this subset the reference population was never pre-screened by the model we are
        # comparing against.
        # ⚠️ Expected to be HARDER, not easier: it is dominated by pre-DL and rational/manual designs.
        # A lower number here is the expected outcome, not a regression.
        self.nonneural_ids = set()
        if nonneural_ids_path is not None:
            with open(nonneural_ids_path) as f:
                self.nonneural_ids = {
                    f"{e['pdb']}_{e['chain_id']}" for e in json.load(f)
                }
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
            [idx + self.index_offset for _ in range(feats["aatype"].shape[-1])], dtype=torch.int64)
        is_overlap = int(f"{pdbid}_{chain_id}" in self.train_overlap_ids)
        feats["is_train_overlap"] = torch.full_like(feats["batch_idx"], is_overlap)
        feats["val_source"] = torch.full_like(feats["batch_idx"], self.source_tag)
        if self.nonneural_ids:
            feats["in_nonneural_subset"] = torch.full_like(
                feats["batch_idx"], int(f"{pdbid}_{chain_id}" in self.nonneural_ids))
        return feats
