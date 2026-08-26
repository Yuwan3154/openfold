# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from functools import partial
import weakref

import torch
import torch.nn as nn

from openfold.data import data_transforms_multimer
from openfold.utils.feats import (
    pseudo_beta_fn,
    build_extra_msa_feat,
    dgram_from_positions,
    atom14_to_atom37,
)
from openfold.utils.tensor_utils import masked_mean
from openfold.model.embedders import (
    InputEmbedder,
    InputEmbedderMultimer,
    RecyclingEmbedder,
    TemplateEmbedder,
    TemplateEmbedderMultimer,
    ExtraMSAEmbedder,
    PreembeddingEmbedder,
)
from openfold.model.evoformer import EvoformerStack, ExtraMSAStack
from openfold.model.heads import AuxiliaryHeads
from openfold.model.structure_module import StructureModule
from openfold.model.template import (
    TemplatePairStack,
    TemplatePointwiseAttention,
    embed_templates_average,
    embed_templates_offload,
)
import openfold.np.residue_constants as residue_constants
from openfold.utils.feats import (
    pseudo_beta_fn,
    build_extra_msa_feat,
    build_template_angle_feat,
    build_template_pair_feat,
    atom14_to_atom37,
)
from openfold.utils.loss import (
    compute_plddt,
)
from openfold.utils.tensor_utils import (
    add,
    dict_multimap,
    tensor_tree_map,
)


class AlphaFold(nn.Module):
    """
    Alphafold 2.

    Implements Algorithm 2 (but with training).
    """

    def __init__(self, config):
        """
        Args:
            config:
                A dict-like config object (like the one in config.py)
        """
        super(AlphaFold, self).__init__()

        self.globals = config.globals
        self.config = config.model
        self.template_config = self.config.template
        self.extra_msa_config = self.config.extra_msa
        self.seqemb_mode = config.globals.seqemb_mode_enabled

        # Main trunk + structure module
        if self.globals.is_multimer:
            self.input_embedder = InputEmbedderMultimer(
                **self.config["input_embedder"]
            )
        elif self.seqemb_mode:
            # If using seqemb mode, embed the sequence embeddings passed
            # to the model ("preembeddings") instead of embedding the sequence
            self.input_embedder = PreembeddingEmbedder(
                **self.config["preembedding_embedder"],
            )
        else:
            self.input_embedder = InputEmbedder(
                **self.config["input_embedder"],
            )

        self.recycling_embedder = RecyclingEmbedder(
            **self.config["recycling_embedder"],
        )

        if self.template_config.enabled:
            if self.globals.is_multimer:
                self.template_embedder = TemplateEmbedderMultimer(
                    self.template_config,
                )
            else:
                self.template_embedder = TemplateEmbedder(
                    self.template_config,
                )

        if self.extra_msa_config.enabled:
            self.extra_msa_embedder = ExtraMSAEmbedder(
                **self.extra_msa_config["extra_msa_embedder"],
            )
            self.extra_msa_stack = ExtraMSAStack(
                **self.extra_msa_config["extra_msa_stack"],
            )

        self.evoformer = EvoformerStack(
            **self.config["evoformer_stack"],
        )

        self.structure_module = StructureModule(
            is_multimer=self.globals.is_multimer,
            **self.config["structure_module"],
        )
        self.aux_heads = AuxiliaryHeads(
            self.config["heads"],
        )

    def embed_templates(self, batch, feats, z, pair_mask, templ_dim, inplace_safe):
        if self.globals.is_multimer:
            asym_id = feats["asym_id"]
            multichain_mask_2d = (
                asym_id[..., None] == asym_id[..., None, :]
            )
            template_embeds = self.template_embedder(
                batch,
                z,
                pair_mask.to(dtype=z.dtype),
                templ_dim,
                chunk_size=self.globals.chunk_size,
                multichain_mask_2d=multichain_mask_2d,
                use_deepspeed_evo_attention=self.globals.use_deepspeed_evo_attention,
                use_cuequivariance_attention=self.globals.use_cuequivariance_attention,
                use_cuequivariance_multiplicative_update=self.globals.use_cuequivariance_multiplicative_update,
                use_lma=self.globals.use_lma,
                inplace_safe=inplace_safe,
                _mask_trans=self.config._mask_trans
            )
            feats["template_torsion_angles_mask"] = (
                template_embeds["template_mask"]
            )
        else:
            if self.template_config.offload_templates:
                return embed_templates_offload(self,
                                               batch, z, pair_mask, templ_dim, inplace_safe=inplace_safe,
                                               )
            elif self.template_config.average_templates:
                return embed_templates_average(self,
                                               batch, z, pair_mask, templ_dim, inplace_safe=inplace_safe,
                                               )

            template_embeds = self.template_embedder(
                batch,
                z,
                pair_mask.to(dtype=z.dtype),
                templ_dim,
                chunk_size=self.globals.chunk_size,
                use_deepspeed_evo_attention=self.globals.use_deepspeed_evo_attention,
                use_cuequivariance_attention=self.globals.use_cuequivariance_attention,
                use_cuequivariance_multiplicative_update=self.globals.use_cuequivariance_multiplicative_update,
                use_lma=self.globals.use_lma,
                inplace_safe=inplace_safe,
                _mask_trans=self.config._mask_trans
            )

        return template_embeds

    def tolerance_reached(self, prev_pos, next_pos, mask, eps=1e-8) -> bool:
        """
        Early stopping criteria based on criteria used in
        AF2Complex: https://www.nature.com/articles/s41467-022-29394-2
        Args:
          prev_pos: Previous atom positions in atom37/14 representation
          next_pos: Current atom positions in atom37/14 representation
          mask: 1-D sequence mask
          eps: Epsilon used in square root calculation
        Returns:
          Whether to stop recycling early based on the desired tolerance.
        """

        def distances(points):
            """Compute all pairwise distances for a set of points."""
            d = points[..., None, :] - points[..., None, :, :]
            return torch.sqrt(torch.sum(d ** 2, dim=-1))

        if self.config.recycle_early_stop_tolerance < 0:
            return False

        ca_idx = residue_constants.atom_order['CA']
        sq_diff = (distances(prev_pos[..., ca_idx, :]) - distances(next_pos[..., ca_idx, :])) ** 2
        mask = mask[..., None] * mask[..., None, :]
        sq_diff = masked_mean(mask=mask, value=sq_diff, dim=list(range(len(mask.shape))))
        diff = torch.sqrt(sq_diff + eps).item()
        return diff <= self.config.recycle_early_stop_tolerance

    def iteration(self, feats, prevs, cycle_no, _recycle=True, outputs={}, return_representations=False):
        # This needs to be done manually for DeepSpeed's sake
        dtype = next(self.parameters()).dtype
        for k in feats:
            if feats[k].dtype == torch.float32:
                feats[k] = feats[k].to(dtype=dtype)

        # Grab some data about the input
        batch_dims = feats["target_feat"].shape[:-2]
        no_batch_dims = len(batch_dims)
        n = feats["target_feat"].shape[-2]
        n_seq = feats["msa_feat"].shape[-3]
        device = feats["target_feat"].device

        # Controls whether the model uses in-place operations throughout
        # The dual condition accounts for activation checkpoints
        inplace_safe = not (self.training or torch.is_grad_enabled())

        # Prep some features
        seq_mask = feats["seq_mask"]
        pair_mask = seq_mask[..., None] * seq_mask[..., None, :]
        msa_mask = feats["msa_mask"]

        if self.globals.is_multimer:
            # Initialize the MSA and pair representations
            # m: [*, S_c, N, C_m]
            # z: [*, N, N, C_z]
            m, z = self.input_embedder(feats)
        elif self.seqemb_mode:
            # Initialize the SingleSeq and pair representations
            # m: [*, 1, N, C_m]
            # z: [*, N, N, C_z]
            m, z = self.input_embedder(
                feats["target_feat"],
                feats["residue_index"],
                feats["seq_embedding"]
            )
        else:
            # Initialize the MSA and pair representations
            # m: [*, S_c, N, C_m]
            # z: [*, N, N, C_z]
            m, z = self.input_embedder(
                feats["target_feat"],
                feats["residue_index"],
                feats["msa_feat"],
                inplace_safe=inplace_safe,
            )

        # Unpack the recycling embeddings. Removing them from the list allows 
        # them to be freed further down in this function, saving memory
        m_1_prev, z_prev, x_prev = reversed([prevs.pop() for _ in range(3)])

        # per-residue coverage of an injected recycle seed; None on every other path, which is what
        # keeps the un-seeded behaviour bit-identical
        recycle_seed_mask = None

        # Initialize the recycling embeddings, if needs be
        if None in [m_1_prev, z_prev, x_prev]:
            # [*, N, C_m]
            m_1_prev = m.new_zeros(
                (*batch_dims, n, self.config.input_embedder.c_m),
                requires_grad=False,
            )

            # [*, N, N, C_z]
            if getattr(self.config.recycling_embedder, "use_gaussian_pair_init", False):
                # ESMFold2-inspired (Appendix A.2.5): independent, seed-varying initial
                # recurrent pair state instead of a deterministic all-zero start -- the source
                # of structural sampling diversity that doesn't depend on MSA masking.
                from openfold.model.contractive_recycling import sample_gaussian_pair_init
                z_prev = sample_gaussian_pair_init(
                    (*batch_dims, n, n, self.config.input_embedder.c_z),
                    d_pair=self.config.input_embedder.c_z,
                    device=z.device,
                    dtype=z.dtype,
                    scale=float(getattr(
                        self.config.recycling_embedder, "gaussian_pair_init_scale", 1.0)),
                )
            else:
                z_prev = z.new_zeros(
                    (*batch_dims, n, n, self.config.input_embedder.c_z),
                    requires_grad=False,
                )

            # [*, N, 3]
            x_prev = z.new_zeros(
                (*batch_dims, n, residue_constants.atom_type_num, 3),
                requires_grad=False,
            )
            # ⭐ RECYCLE-SEED: replace that all-zero start with a real structure (a synthetic
            # template or the model's own promoted prediction), so the recycling DISTOGRAM track
            # opens on a candidate fold instead of the degenerate "every pair at distance 0" bin.
            # This is the training-time analogue of the inference condition where a candidate
            # structure seeds the search. Cycle 0 ONLY -- afterwards the model's own prediction
            # takes over, exactly as before.
            # ⛔ Gated on the feature being PRESENT, so a run whose dataset does not emit it is
            # bit-identical to before. `recycle_seed_mask` is per-residue coverage; templates cover
            # only part of the chain and an uncovered residue left at the origin would fabricate
            # contacts with everything.
            if "recycle_seed_positions" in feats:
                _seed = feats["recycle_seed_positions"].to(dtype=x_prev.dtype)
                _seed = _seed[..., 0] if _seed.shape[-1] == 1 else _seed
                x_prev = _seed
                recycle_seed_mask = feats["recycle_seed_mask"]
                recycle_seed_mask = (recycle_seed_mask[..., 0]
                                     if recycle_seed_mask.shape[-1] == 1 else recycle_seed_mask)

        pseudo_beta_x_prev = pseudo_beta_fn(
            feats["aatype"], x_prev, None
        ).to(dtype=z.dtype)

        # The recycling embedder is memory-intensive, so we offload first
        if self.globals.offload_inference and inplace_safe:
            m = m.cpu()
            z = z.cpu()

        # m_1_prev_emb: [*, N, C_m]
        # z_prev_emb: [*, N, N, C_z]
        m_1_prev_emb, z_prev_emb = self.recycling_embedder(
            m_1_prev,
            z_prev,
            pseudo_beta_x_prev,
            inplace_safe=inplace_safe,
            x_mask=recycle_seed_mask,
        )

        del pseudo_beta_x_prev

        if self.globals.offload_inference and inplace_safe:
            m = m.to(m_1_prev_emb.device)
            z = z.to(z_prev.device)

        # [*, S_c, N, C_m]
        m[..., 0, :, :] += m_1_prev_emb

        # [*, N, N, C_z]
        if getattr(self.config.recycling_embedder, "use_contractive", False):
            # ESMFold2-inspired (Appendix A.2.5, arXiv:2604.12946): z_prev_emb here is the
            # distogram-derived signal ALONE (RecyclingEmbedder.forward's use_contractive
            # branch), not yet combined with z_prev. Combine raw z_prev with (z + that signal)
            # via the contractive recurrence instead of a plain additive residual -- this is
            # what keeps the recurrent state numerically bounded across many recycle iterations.
            u_t = add(z, z_prev_emb, inplace=inplace_safe)
            z = self.recycling_embedder.contractive_pair_update(z_prev, u_t)
        else:
            z = add(z, z_prev_emb, inplace=inplace_safe)

        # Deletions like these become significant for inference with large N,
        # where they free unused tensors and remove references to others such
        # that they can be offloaded later
        del m_1_prev, z_prev, m_1_prev_emb, z_prev_emb

        # Embed the templates + merge with MSA/pair embeddings
        if self.config.template.enabled:
            template_feats = {
                k: v for k, v in feats.items() if k.startswith("template_")
            }

            template_embeds = self.embed_templates(
                template_feats,
                feats,
                z,
                pair_mask.to(dtype=z.dtype),
                no_batch_dims,
                inplace_safe=inplace_safe,
            )

            # [*, N, N, C_z]
            z = add(z,
                    template_embeds.pop("template_pair_embedding"),
                    inplace_safe,
                    )

            if (
                "template_single_embedding" in template_embeds
            ):
                # [*, S = S_c + S_t, N, C_m]
                m = torch.cat(
                    [m, template_embeds["template_single_embedding"]],
                    dim=-3
                )

                # [*, S, N]
                if not self.globals.is_multimer:
                    torsion_angles_mask = feats["template_torsion_angles_mask"]
                    msa_mask = torch.cat(
                        [feats["msa_mask"], torsion_angles_mask[..., 2]],
                        dim=-2
                    )
                else:
                    msa_mask = torch.cat(
                        [feats["msa_mask"], template_embeds["template_mask"]],
                        dim=-2,
                    )

        # Embed extra MSA features + merge with pairwise embeddings
        if self.config.extra_msa.enabled:
            if self.globals.is_multimer:
                extra_msa_fn = data_transforms_multimer.build_extra_msa_feat
            else:
                extra_msa_fn = build_extra_msa_feat

            # [*, S_e, N, C_e]
            extra_msa_feat = extra_msa_fn(feats).to(dtype=z.dtype)
            a = self.extra_msa_embedder(extra_msa_feat)

            if self.globals.offload_inference:
                # To allow the extra MSA stack (and later the evoformer) to
                # offload its inputs, we remove all references to them here
                input_tensors = [a, z]
                del a, z

                # [*, N, N, C_z]
                z = self.extra_msa_stack._forward_offload(
                    input_tensors,
                    msa_mask=feats["extra_msa_mask"].to(dtype=m.dtype),
                    chunk_size=self.globals.chunk_size,
                    use_deepspeed_evo_attention=self.globals.use_deepspeed_evo_attention,
                    use_cuequivariance_attention=self.globals.use_cuequivariance_attention,
                    use_cuequivariance_multiplicative_update=self.globals.use_cuequivariance_multiplicative_update,
                    use_lma=self.globals.use_lma,
                    pair_mask=pair_mask.to(dtype=m.dtype),
                    _mask_trans=self.config._mask_trans,
                )

                del input_tensors
            else:
                # [*, N, N, C_z]
                z = self.extra_msa_stack(
                    a, z,
                    msa_mask=feats["extra_msa_mask"].to(dtype=m.dtype),
                    chunk_size=self.globals.chunk_size,
                    use_deepspeed_evo_attention=self.globals.use_deepspeed_evo_attention,
                    use_cuequivariance_attention=self.globals.use_cuequivariance_attention,
                    use_cuequivariance_multiplicative_update=self.globals.use_cuequivariance_multiplicative_update,
                    use_lma=self.globals.use_lma,
                    pair_mask=pair_mask.to(dtype=m.dtype),
                    inplace_safe=inplace_safe,
                    _mask_trans=self.config._mask_trans,
                )

        # Run MSA + pair embeddings through the trunk of the network
        # m: [*, S, N, C_m]
        # z: [*, N, N, C_z]
        # s: [*, N, C_s]          
        if self.globals.offload_inference:
            input_tensors = [m, z]
            del m, z
            m, z, s = self.evoformer._forward_offload(
                input_tensors,
                msa_mask=msa_mask.to(dtype=input_tensors[0].dtype),
                pair_mask=pair_mask.to(dtype=input_tensors[1].dtype),
                chunk_size=self.globals.chunk_size,
                use_deepspeed_evo_attention=self.globals.use_deepspeed_evo_attention,
                use_cuequivariance_attention=self.globals.use_cuequivariance_attention,
                use_cuequivariance_multiplicative_update=self.globals.use_cuequivariance_multiplicative_update,
                use_lma=self.globals.use_lma,
                _mask_trans=self.config._mask_trans,
            )

            del input_tensors
        else:
            m, z, s = self.evoformer(
                m,
                z,
                msa_mask=msa_mask.to(dtype=m.dtype),
                pair_mask=pair_mask.to(dtype=z.dtype),
                outputs=outputs,
                cycle_no=cycle_no,
                chunk_size=self.globals.chunk_size,
                use_deepspeed_evo_attention=self.globals.use_deepspeed_evo_attention,
                use_cuequivariance_attention=self.globals.use_cuequivariance_attention,
                use_cuequivariance_multiplicative_update=self.globals.use_cuequivariance_multiplicative_update,
                use_lma=self.globals.use_lma,
                use_flash=self.globals.use_flash,
                inplace_safe=inplace_safe,
                _mask_trans=self.config._mask_trans,
            )

        outputs["msa"] = m[..., :n_seq, :, :]
        outputs["pair"] = z
        outputs["single"] = s

        # Early return for representation-only mode (skip structure module)
        if return_representations:
            # For representation mode, we don't need structure predictions
            # Return dummy values for recycling (they won't be used with 0 recycles)
            m_1_prev = None
            z_prev = None
            x_prev = None
            early_stop = True  # Signal to stop recycling
            return outputs, m_1_prev, z_prev, x_prev, early_stop

        del z

        # Predict 3D structure
        outputs["sm"] = self.structure_module(
            outputs,
            feats["aatype"],
            mask=feats["seq_mask"].to(dtype=s.dtype),
            inplace_safe=inplace_safe,
            _offload_inference=self.globals.offload_inference,
        )
        outputs["final_atom_positions"] = atom14_to_atom37(
            outputs["sm"]["positions"][-1], feats
        )
        outputs["final_atom_mask"] = feats["atom37_atom_exists"]
        outputs["final_affine_tensor"] = outputs["sm"]["frames"][-1]

        # Save embeddings for use during the next recycling iteration

        # [*, N, C_m]
        m_1_prev = m[..., 0, :, :]

        # [*, N, N, C_z]
        z_prev = outputs["pair"]

        early_stop = False
        if self.globals.is_multimer:
            early_stop = self.tolerance_reached(x_prev, outputs["final_atom_positions"], seq_mask)

        del x_prev

        # [*, N, 3]
        x_prev = outputs["final_atom_positions"]

        return outputs, m_1_prev, z_prev, x_prev, early_stop

    def _disable_activation_checkpointing(self):
        self.template_embedder.template_pair_stack.blocks_per_ckpt = None
        self.evoformer.blocks_per_ckpt = None

        for b in self.extra_msa_stack.blocks:
            b.ckpt = False

    def _enable_activation_checkpointing(self):
        self.template_embedder.template_pair_stack.blocks_per_ckpt = (
            self.config.template.template_pair_stack.blocks_per_ckpt
        )
        self.evoformer.blocks_per_ckpt = (
            self.config.evoformer_stack.blocks_per_ckpt
        )

        for b in self.extra_msa_stack.blocks:
            b.ckpt = self.config.extra_msa.extra_msa_stack.ckpt

    def forward(self, batch):
        """
        Args:
            batch:
                Dictionary of arguments outlined in Algorithm 2. Keys must
                include the official names of the features in the
                supplement subsection 1.2.9.

                The final dimension of each input must have length equal to
                the number of recycling iterations.

                Features (without the recycling dimension):

                    "aatype" ([*, N_res]):
                        Contrary to the supplement, this tensor of residue
                        indices is not one-hot.
                    "target_feat" ([*, N_res, C_tf])
                        One-hot encoding of the target sequence. C_tf is
                        config.model.input_embedder.tf_dim.
                    "residue_index" ([*, N_res])
                        Tensor whose final dimension consists of
                        consecutive indices from 0 to N_res.
                    "msa_feat" ([*, N_seq, N_res, C_msa])
                        MSA features, constructed as in the supplement.
                        C_msa is config.model.input_embedder.msa_dim.
                    "seq_mask" ([*, N_res])
                        1-D sequence mask
                    "msa_mask" ([*, N_seq, N_res])
                        MSA mask
                    "pair_mask" ([*, N_res, N_res])
                        2-D pair mask
                    "extra_msa_mask" ([*, N_extra, N_res])
                        Extra MSA mask
                    "template_mask" ([*, N_templ])
                        Template mask (on the level of templates, not
                        residues)
                    "template_aatype" ([*, N_templ, N_res])
                        Tensor of template residue indices (indices greater
                        than 19 are clamped to 20 (Unknown))
                    "template_all_atom_positions"
                        ([*, N_templ, N_res, 37, 3])
                        Template atom coordinates in atom37 format
                    "template_all_atom_mask" ([*, N_templ, N_res, 37])
                        Template atom coordinate mask
                    "template_pseudo_beta" ([*, N_templ, N_res, 3])
                        Positions of template carbon "pseudo-beta" atoms
                        (i.e. C_beta for all residues but glycine, for
                        for which C_alpha is used instead)
                    "template_pseudo_beta_mask" ([*, N_templ, N_res])
                        Pseudo-beta mask
        """
        # Initialize recycling embeddings
        m_1_prev, z_prev, x_prev = None, None, None
        prevs = [m_1_prev, z_prev, x_prev]

        is_grad_enabled = torch.is_grad_enabled()

        # Main recycling loop
        num_iters = batch["aatype"].shape[-1]
        early_stop = False
        num_recycles = 0
        outputs = {}
        
        # Extract non-tensor metadata before recycling loop
        return_representations = batch.pop("return_representations", False)
        
        for cycle_no in range(num_iters):
            # Select the features for the current recycling cycle
            fetch_cur_batch = lambda t: t[..., cycle_no]
            feats = tensor_tree_map(fetch_cur_batch, batch)

            # Enable grad iff we're training and it's the final recycling layer
            is_final_iter = cycle_no == (num_iters - 1) or early_stop
            with torch.set_grad_enabled(is_grad_enabled and is_final_iter):
                if is_final_iter:
                    # Sidestep AMP bug (PyTorch issue #65766)
                    if torch.is_autocast_enabled():
                        torch.clear_autocast_cache()

                # Run the next iteration of the model
                outputs, m_1_prev, z_prev, x_prev, early_stop = self.iteration(
                    feats,
                    prevs,
                    cycle_no=cycle_no,
                    _recycle=(num_iters > 1),
                    outputs=outputs,
                    return_representations=return_representations,
                )

                num_recycles += 1

                if not is_final_iter:
                    prevs = [m_1_prev, z_prev, x_prev]
                    del m_1_prev, z_prev, x_prev
                else:
                    break

        outputs["num_recycles"] = torch.tensor(num_recycles, device=feats["aatype"].device)

        if "asym_id" in batch:
            outputs["asym_id"] = feats["asym_id"]

        # Run auxiliary heads only if we ran structure module
        # (skip for representation-only mode)
        if not return_representations:
            outputs.update(self.aux_heads(outputs))

        return outputs

    # ------------------------------------------------------------------
    # Compile-friendly inference-only forward path
    # ------------------------------------------------------------------
    #
    # forward_inference()/iteration_inference() are a slimmed-down inference
    # path designed to be wrapped with torch.compile(dynamic=True).  They
    # differ from forward()/iteration() in that:
    #
    #   * Always inference (no torch.set_grad_enabled, no autocast cache flush,
    #     no activation checkpointing toggling).
    #   * Recycling state (m_1_prev/z_prev/x_prev) is pre-allocated as zeros
    #     before the loop, so no `None in [...]` branch inside iteration_*.
    #   * `inplace_safe=False`, `_force_no_inplace=True` everywhere
    #     (skip attn_core_inplace_cuda extension in IPA — not compile-traceable).
    #   * `_offload_inference=False`, no chunking, no deepspeed/cueq/lma/flash
    #     kernels — single compile-friendly attention path (SDPA by default,
    #     or vanilla matmul as a numerics-fidelity fallback).
    #   * No tolerance-based early stopping (contains a .item() call).
    #   * No `return_representations` mode.
    #   * Multimer is NOT supported here (AF2Rank is monomer-only).
    #
    # The outer recycling loop is kept in eager Python so torch.compile only
    # has to specialise on iteration_inference().

    def iteration_inference(
        self,
        feats,
        m_1_prev,
        z_prev,
        x_prev,
        use_torch_sdpa: bool = True,
        use_torch_vanilla: bool = False,
        use_torch_cueq: bool = False,
    ):
        # cycle_no is unused on the inference path (no activation
        # checkpointing, no per-cycle output saving), so we pass a constant
        # 0 to evoformer to avoid Dynamo re-specialising the compiled graph
        # per cycle.
        cycle_no = 0
        if self.globals.is_multimer or self.seqemb_mode:
            raise NotImplementedError(
                "forward_inference/iteration_inference only support monomer "
                "single-sequence mode (AF2Rank)."
            )
        if getattr(self.config.recycling_embedder, "use_contractive", False):
            # This compiled, AF2Rank-specific fast path has its own separate `z = z + z_prev_emb`
            # combination step (below) that was never updated for the contractive recurrence --
            # doing so silently here would combine z_prev_emb (which means something DIFFERENT
            # when use_contractive=True -- see RecyclingEmbedder.forward) incorrectly. Not yet
            # supported; use the regular forward()/iteration() path instead.
            raise NotImplementedError(
                "use_contractive is not yet supported on the iteration_inference/AF2Rank fast "
                "path -- use the regular forward()/iteration() path instead."
            )
        if getattr(self.config.recycling_embedder, "use_gaussian_pair_init", False):
            # z_prev is pre-allocated as zeros by the caller (forward_inference) before this
            # method ever runs -- use_gaussian_pair_init's sampling logic lives in iteration()'s
            # `None in [...]` branch only, so it would be silently ignored here. Guard instead
            # of letting the flag silently do nothing on this path.
            raise NotImplementedError(
                "use_gaussian_pair_init is not yet supported on the iteration_inference/AF2Rank "
                "fast path (z_prev is pre-zeroed by the caller) -- use the regular "
                "forward()/iteration() path instead."
            )

        # Convert input float32 feats to the model's parameter dtype.  This
        # mirrors the preamble of iteration() and is compile-friendly because
        # the dict iteration order is stable.
        dtype = next(self.parameters()).dtype
        for k in list(feats.keys()):
            if feats[k].dtype == torch.float32:
                feats[k] = feats[k].to(dtype=dtype)

        n_seq = feats["msa_feat"].shape[-3]

        seq_mask = feats["seq_mask"]
        pair_mask = seq_mask[..., None] * seq_mask[..., None, :]
        msa_mask = feats["msa_mask"]

        # Initialize MSA and pair representations.
        # m: [*, S_c, N, C_m] ; z: [*, N, N, C_z]
        m, z = self.input_embedder(
            feats["target_feat"],
            feats["residue_index"],
            feats["msa_feat"],
            inplace_safe=False,
        )

        # Recycling embedder
        pseudo_beta_x_prev = pseudo_beta_fn(
            feats["aatype"], x_prev, None
        ).to(dtype=z.dtype)

        m_1_prev_emb, z_prev_emb = self.recycling_embedder(
            m_1_prev,
            z_prev,
            pseudo_beta_x_prev,
            inplace_safe=False,
        )

        # The original forward uses `m[..., 0, :, :] += m_1_prev_emb` which is
        # an in-place op that compile dislikes when m is a graph input.
        # Build a fresh m with the recycling update applied to row 0.
        m = torch.cat(
            [
                (m[..., 0, :, :] + m_1_prev_emb).unsqueeze(-3),
                m[..., 1:, :, :],
            ],
            dim=-3,
        )
        z = z + z_prev_emb

        # Hint the L (residue) axis on m, z so the per-block compile
        # contexts treat L as symbolic from the FIRST call.  Use the SOFT
        # form (`maybe_mark_dynamic`) — the strict form raises
        # ConstraintViolationError when Dynamo's symbolic engine infers
        # the dim must be constant during tracing, which IS the case at
        # several spots inside AlphaFold blocks (mixed shape arithmetic
        # between marked and non-marked tensors).
        # Skipped when iteration_inference itself is being traced by
        # Dynamo (whole-graph strategy) — mark_dynamic / maybe_mark_dynamic
        # are forbidden callables inside a compiled function and would
        # raise.
        if not torch.compiler.is_compiling():
            _mfn = getattr(
                torch._dynamo, "maybe_mark_dynamic", torch._dynamo.mark_dynamic,
            )
            try:
                _mfn(m, m.dim() - 2)   # N_res axis in m
                _mfn(z, z.dim() - 2)   # one of z's L axes
            except Exception:
                pass

        # Chunking configuration for the inference path.
        #   * Main Evoformer trunk: chunk_size=None (single-MSA-seq, the
        #     attention intermediates are <10 MB even at L=512, no need to
        #     chunk).
        #   * TemplatePairStack: chunk_size=None as well (small N_templ).
        #   * ExtraMSAStack: chunked along N_extra (5120) using a fixed
        #     chunk_size that divides cleanly so the chunk loop count is
        #     L-independent. Defaults to 64 → 80 chunks; users may set
        #     `model._inference_aux_chunk_size` to override.
        _aux_chunk = getattr(self, "_inference_aux_chunk_size", 64)
        _use_cueq_mul = getattr(self, "_inference_use_cueq_mul_update", False)

        # Template embedder (single-sequence + distogram or full template).
        if self.config.template.enabled:
            template_feats = {
                k: v for k, v in feats.items() if k.startswith("template_")
            }
            template_embeds = self.template_embedder(
                template_feats,
                z,
                pair_mask.to(dtype=z.dtype),
                # templ_dim is the leading template dim of a single-sample
                # batch.  iteration() passes len(target_feat.shape[:-2]).
                len(feats["target_feat"].shape[:-2]),
                chunk_size=None,
                use_deepspeed_evo_attention=False,
                use_cuequivariance_attention=use_torch_cueq,
                use_cuequivariance_multiplicative_update=_use_cueq_mul,
                use_lma=False,
                inplace_safe=False,
                _mask_trans=self.config._mask_trans,
                use_torch_sdpa=use_torch_sdpa,
                use_torch_vanilla=use_torch_vanilla,
            )

            z = z + template_embeds.pop("template_pair_embedding")

            if "template_single_embedding" in template_embeds:
                m = torch.cat(
                    [m, template_embeds["template_single_embedding"]], dim=-3,
                )
                torsion_angles_mask = feats["template_torsion_angles_mask"]
                msa_mask = torch.cat(
                    [feats["msa_mask"], torsion_angles_mask[..., 2]],
                    dim=-2,
                )

        # Extra MSA stack (still optional in monomer config).
        if self.config.extra_msa.enabled:
            extra_msa_feat = build_extra_msa_feat(feats).to(dtype=z.dtype)
            a = self.extra_msa_embedder(extra_msa_feat)
            # ExtraMSABlock.forward sees `a` as positional `m` — hint
            # its L axis so the per-block compile sees a symbolic dim.
            # Use soft (maybe_mark_dynamic) — see m/z note above for why.
            # Skipped under whole-graph compile (see note above).
            if not torch.compiler.is_compiling():
                try:
                    _mfn = getattr(
                        torch._dynamo, "maybe_mark_dynamic",
                        torch._dynamo.mark_dynamic,
                    )
                    _mfn(a, a.dim() - 2)
                except Exception:
                    pass
            z = self.extra_msa_stack(
                a, z,
                msa_mask=feats["extra_msa_mask"].to(dtype=m.dtype),
                chunk_size=_aux_chunk,
                use_deepspeed_evo_attention=False,
                use_cuequivariance_attention=use_torch_cueq,
                use_cuequivariance_multiplicative_update=_use_cueq_mul,
                use_lma=False,
                pair_mask=pair_mask.to(dtype=m.dtype),
                inplace_safe=False,
                _mask_trans=self.config._mask_trans,
                use_torch_sdpa=use_torch_sdpa,
                use_torch_vanilla=use_torch_vanilla,
            )

        # Evoformer trunk
        outputs = {}
        m, z, s = self.evoformer(
            m,
            z,
            msa_mask=msa_mask.to(dtype=m.dtype),
            pair_mask=pair_mask.to(dtype=z.dtype),
            outputs=outputs,
            cycle_no=cycle_no,
            chunk_size=None,
            use_deepspeed_evo_attention=False,
            use_cuequivariance_attention=use_torch_cueq,
            use_cuequivariance_multiplicative_update=_use_cueq_mul,
            use_lma=False,
            use_flash=False,
            inplace_safe=False,
            _mask_trans=self.config._mask_trans,
            use_torch_sdpa=use_torch_sdpa,
            use_torch_vanilla=use_torch_vanilla,
        )

        outputs["msa"] = m[..., :n_seq, :, :]
        outputs["pair"] = z
        outputs["single"] = s

        # Structure module
        outputs["sm"] = self.structure_module(
            outputs,
            feats["aatype"],
            mask=feats["seq_mask"].to(dtype=s.dtype),
            inplace_safe=False,
            _offload_inference=False,
            _force_no_inplace=True,
        )
        outputs["final_atom_positions"] = atom14_to_atom37(
            outputs["sm"]["positions"][-1], feats
        )
        outputs["final_atom_mask"] = feats["atom37_atom_exists"]
        outputs["final_affine_tensor"] = outputs["sm"]["frames"][-1]

        # Recycling state for next iteration.
        m_1_prev_next = m[..., 0, :, :]
        z_prev_next = outputs["pair"]
        x_prev_next = outputs["final_atom_positions"]

        return outputs, m_1_prev_next, z_prev_next, x_prev_next

    def _mark_inputs_dynamic(self, batch: dict) -> None:
        """Mark the residue (L) dimension of per-residue feats as dynamic.

        torch._dynamo.mark_dynamic is idempotent and silent in eager.
        For the input batch tensors (pre per-cycle slice), the trailing dim
        is the recycling dim R, and the NUM_RES (L) axis sits one or more
        positions before the end. Slicing along the recycling dim
        preserves dynamic-shape annotations.

        Multi-axis features (e.g. `template_dgram_probs` shaped
        `(N_templ, L, L, 39, R)`) need every L axis marked, hence the
        per-key axis lists.
        """
        # Per-key list of negative-axis indices into the pre-slice tensor
        # (trailing axis is the recycling dim).
        # Mirrors `data.common.feat` in openfold/config.py, expanded to
        # include every NUM_RES axis explicitly. Keys mapping to () are
        # scalar/no-residue features and skipped.
        res_axes = {
            "aatype": (-2,),
            "target_feat": (-3,),
            "residue_index": (-2,),
            "seq_mask": (-2,),
            "msa_mask": (-2,),
            "msa_feat": (-3,),
            "extra_msa": (-2,),
            "extra_msa_mask": (-2,),
            "extra_has_deletion": (-2,),
            "extra_deletion_value": (-2,),
            "template_aatype": (-2,),
            "template_all_atom_mask": (-3,),
            "template_all_atom_positions": (-4,),
            "template_pseudo_beta_mask": (-2,),
            "template_pseudo_beta": (-3,),
            "template_torsion_angles_sin_cos": (-4,),
            "template_alt_torsion_angles_sin_cos": (-4,),
            "template_torsion_angles_mask": (-3,),
            "atom14_atom_exists": (-3,),
            "atom37_atom_exists": (-3,),
            "residx_atom14_to_atom37": (-3,),
            "residx_atom37_to_atom14": (-3,),
            "atom14_alt_gt_positions": (-4,),
            "atom14_alt_gt_exists": (-3,),
            "atom14_atom_is_ambiguous": (-3,),
            "all_atom_positions": (-4,),
            "all_atom_mask": (-3,),
            "asym_id": (-2,),
            "sym_id": (-2,),
            "entity_id": (-2,),
            "deletion_mean": (-2,),
            "pseudo_beta": (-3,),
            "pseudo_beta_mask": (-2,),
            "bert_mask": (-2,),
            "true_msa": (-2,),
            # AF2Rank-specific distogram tensor: (N_templ, L, L, 39, R).
            "template_dgram_probs": (-4, -3),
            # Pair features explicitly carrying two N_res axes.
            "msa_row_mask": (),       # NUM_MSA_SEQ only
            "template_mask": (),
            "template_sum_probs": (),
            "extra_msa_row_mask": (),
            "is_distillation": (),
            "no_recycling_iters": (),
            "use_clamped_fape": (),
            "seq_length": (),
            "resolution": (),
        }
        for k, v in batch.items():
            axes = res_axes.get(k, None)
            if axes is None:
                # Unknown key — best-effort: mark every axis except the
                # trailing (recycling) one and the leading batch/sequence
                # dims of size <=4 (heuristic, may over-mark).
                if not isinstance(v, torch.Tensor) or v.dim() == 0:
                    continue
                axes = tuple(range(-v.dim() + 1, -1))  # all but trailing
            if not isinstance(v, torch.Tensor):
                continue
            for ax in axes:
                if v.dim() < abs(ax):
                    continue
                try:
                    torch._dynamo.mark_dynamic(v, v.dim() + ax)
                except Exception:
                    pass

    def forward_inference(self, batch):
        """Compile-friendly inference forward.

        Strips the training-only branches (gradient toggling, autocast cache
        flush, activation checkpointing) from forward()/iteration() and
        routes attention through torch.nn.functional.scaled_dot_product_attention
        (or vanilla matmul softmax) and the IPA non-inplace branch.

        The outer recycling loop runs in eager Python; iteration_inference()
        is what should be wrapped with torch.compile.
        """
        if self.globals.is_multimer or self.seqemb_mode:
            raise NotImplementedError(
                "forward_inference only supports monomer single-sequence mode."
            )

        # Recycling-loop trip count is encoded in the recycling dim of every
        # tensor in `batch`.  This is fixed at config build time (typically 4).
        num_iters = batch["aatype"].shape[-1]

        # Pull (and ignore) representation-only flag so the rest of the path
        # is purely tensor-only.
        batch.pop("return_representations", False)

        # Slice cycle 0 features eagerly so we can allocate recycling state
        # at the right shapes.
        feats0 = tensor_tree_map(lambda t: t[..., 0], batch)

        # Mark the sequence-length dimension of every per-residue feature
        # as dynamic so the compiled blocks reuse a single graph across
        # different L.  Without this, Dynamo specialises on first-call
        # shape and triggers a full recompile on the second L it sees.
        # This is a no-op in eager (mark_dynamic is harmless without compile).
        self._mark_inputs_dynamic(batch)

        device = feats0["target_feat"].device
        dtype = next(self.parameters()).dtype
        n = feats0["target_feat"].shape[-2]
        batch_dims = feats0["target_feat"].shape[:-2]

        m_1_prev = torch.zeros(
            (*batch_dims, n, self.config.input_embedder.c_m),
            device=device, dtype=dtype,
        )
        z_prev = torch.zeros(
            (*batch_dims, n, n, self.config.input_embedder.c_z),
            device=device, dtype=dtype,
        )
        x_prev = torch.zeros(
            (*batch_dims, n, residue_constants.atom_type_num, 3),
            device=device, dtype=dtype,
        )

        # Hint the L axis on the freshly allocated recycling state
        # tensors as dynamic too.  Without this the compiled
        # iteration_inference body would re-specialise the second time
        # it sees a new L, because torch.zeros((..., n, ...)) where n
        # came from a symint still produces a concrete tensor whose
        # shape Dynamo will guard on by default.  Soft hint
        # (maybe_mark_dynamic) — strict mark_dynamic would crash with
        # ConstraintViolationError because internal ops force the second
        # L of z_prev to specialise to the first.
        # forward_inference always runs eager (outer recycling loop).
        try:
            _mfn = getattr(
                torch._dynamo, "maybe_mark_dynamic", torch._dynamo.mark_dynamic,
            )
            _mfn(m_1_prev, m_1_prev.dim() - 2)
            # Only hint ONE L axis of z_prev (the two L dims share a
            # symbolic value through the ops in iteration_inference).
            _mfn(z_prev, z_prev.dim() - 2)
            _mfn(x_prev, x_prev.dim() - 3)
        except Exception:
            pass

        # Attention kernel selection.  Default to SDPA; the inference attribute
        # can be set on the model to switch to the vanilla fallback or to
        # cuEquivariance triangle attention.
        use_torch_sdpa = getattr(self, "_inference_use_torch_sdpa", True)
        use_torch_vanilla = getattr(self, "_inference_use_torch_vanilla", False)
        use_torch_cueq = getattr(self, "_inference_use_torch_cueq", False)
        # Mutually exclusive: cueq > vanilla > sdpa.
        if use_torch_cueq:
            use_torch_sdpa = False
            use_torch_vanilla = False
        elif use_torch_vanilla:
            use_torch_sdpa = False

        # The wrapper may have wrapped individual block forwards with
        # torch.compile (per-block compilation strategy).  iteration_inference
        # itself stays eager so the recycling loop can call the (possibly
        # compiled) sub-modules.
        outputs = {}
        for cycle_no in range(num_iters):
            feats = tensor_tree_map(lambda t: t[..., cycle_no], batch)
            outputs, m_1_prev, z_prev, x_prev = self.iteration_inference(
                feats,
                m_1_prev,
                z_prev,
                x_prev,
                use_torch_sdpa,
                use_torch_vanilla,
                use_torch_cueq,
            )

        outputs["num_recycles"] = torch.tensor(num_iters, device=device)

        # Auxiliary heads (pTM/pLDDT/pAE/distogram) run eagerly after the loop.
        outputs.update(self.aux_heads(outputs))

        return outputs
