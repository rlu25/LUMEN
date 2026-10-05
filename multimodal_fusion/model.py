"""Masked cross-modal Transformer for shared-representation pretraining."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .data import MAX_GENETIC_TOKENS, MODALITY_DIMS, MODALITY_IDX, VISIT_IDX
except ImportError:
    from data import MAX_GENETIC_TOKENS, MODALITY_DIMS, MODALITY_IDX, VISIT_IDX


class MultimodalFusionMAE(nn.Module):
    def __init__(self, d_model=256, d_decoder=128, encoder_depth=8,
                 decoder_depth=4, n_heads=8, mask_ratio=0.60, dropout=0.1,
                 masking_mode="token", pooling_mode="token_mean",
                 fusion_mode="token"):
        super().__init__()
        if masking_mode not in {"token", "modality"}:
            raise ValueError("masking_mode must be 'token' or 'modality'")
        if pooling_mode not in {"token_mean", "modality_mean", "fusion_token"}:
            raise ValueError(
                "pooling_mode must be 'token_mean', 'modality_mean', or "
                "'fusion_token'"
            )
        if fusion_mode not in {"token", "modality_summary"}:
            raise ValueError("fusion_mode must be 'token' or 'modality_summary'")
        if pooling_mode == "fusion_token" and fusion_mode != "modality_summary":
            raise ValueError(
                "pooling_mode='fusion_token' requires "
                "fusion_mode='modality_summary'"
            )
        self.d_model = d_model
        self.d_decoder = d_decoder
        self.mask_ratio = mask_ratio
        self.masking_mode = masking_mode
        self.pooling_mode = pooling_mode
        self.fusion_mode = fusion_mode
        self.projectors = nn.ModuleDict({
            name: nn.Linear(dim, d_model) for name, dim in MODALITY_DIMS.items()
        })
        self.modality_embedding = nn.Embedding(len(MODALITY_IDX), d_model)
        self.visit_embedding = nn.Embedding(len(VISIT_IDX), d_model)
        self.genetic_slot_embedding = nn.Embedding(MAX_GENETIC_TOKENS, d_model)
        self.mask_token = nn.Parameter(torch.randn(d_model) * 0.02)
        if fusion_mode == "modality_summary":
            self.modality_queries = nn.Embedding(len(MODALITY_IDX), d_model)
            self.within_modality_attention = nn.MultiheadAttention(
                d_model, n_heads, dropout=dropout, batch_first=True,
            )
            if pooling_mode == "fusion_token":
                # This is the single, task-independent foundation bottleneck.
                # It attends jointly with the modality summaries in the encoder.
                self.fusion_token = nn.Parameter(torch.empty(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_model * 4, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, encoder_depth)
        self.encoder_norm = nn.LayerNorm(d_model)
        self.decoder_projection = nn.Linear(d_model, d_decoder)
        decoder_layer = nn.TransformerEncoderLayer(
            d_decoder, max(1, n_heads // 2), d_decoder * 4, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, decoder_depth)
        self.decoder_norm = nn.LayerNorm(d_decoder)
        self.reconstruction_heads = nn.ModuleDict({
            name: nn.Linear(d_decoder, dim) for name, dim in MODALITY_DIMS.items()
        })
        self.apply(self._initialize)
        if hasattr(self, "fusion_token"):
            nn.init.trunc_normal_(self.fusion_token, std=0.02)

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=0.02)

    def project_tokens(self, raw, mod_ids, padding):
        batch, n_tokens, _ = raw.shape
        projected = raw.new_zeros((batch, n_tokens, self.d_model))
        for modality, mod_idx in MODALITY_IDX.items():
            selected = (mod_ids == mod_idx) & ~padding
            if selected.any():
                dim = MODALITY_DIMS[modality]
                projected[selected] = self.projectors[modality](raw[selected][:, :dim])
        return projected

    def _summarize_modalities(self, tokens, mod_ids, padding):
        """Learn one summary from all available tokens inside each modality."""
        batch_size = len(tokens)
        summaries = tokens.new_zeros((batch_size, len(MODALITY_IDX), self.d_model))
        summary_padding = torch.ones(
            batch_size, len(MODALITY_IDX), dtype=torch.bool, device=tokens.device
        )
        for mod_idx in MODALITY_IDX.values():
            selected = (mod_ids == mod_idx) & ~padding
            present = selected.any(dim=1)
            if not present.any():
                continue
            query = self.modality_queries.weight[mod_idx].view(1, 1, -1)
            query = query.expand(int(present.sum()), -1, -1)
            key_padding = ~selected[present]
            summary, _ = self.within_modality_attention(
                query,
                tokens[present],
                tokens[present],
                key_padding_mask=key_padding,
                need_weights=False,
            )
            summaries[present, mod_idx] = summary[:, 0]
            summary_padding[present, mod_idx] = False
        return summaries, summary_padding

    def _positions(self, mod_ids, visit_ids, slot_ids):
        return (
            self.modality_embedding(mod_ids)
            + self.visit_embedding(visit_ids)
            + self.genetic_slot_embedding(slot_ids)
        )

    def _encode_modality_summaries(self, raw, mod_ids, visit_ids, slot_ids, padding):
        tokens = self.project_tokens(raw, mod_ids, padding)
        tokens = tokens + self._positions(mod_ids, visit_ids, slot_ids)
        summaries, summary_padding = self._summarize_modalities(tokens, mod_ids, padding)
        if self.pooling_mode == "fusion_token":
            fusion = self.fusion_token.expand(len(summaries), -1, -1)
            encoder_input = torch.cat([fusion, summaries], dim=1)
            fusion_padding = torch.zeros(
                len(summaries), 1, dtype=torch.bool, device=summaries.device
            )
            encoder_padding = torch.cat([fusion_padding, summary_padding], dim=1)
            encoded = self.encoder(
                encoder_input, src_key_padding_mask=encoder_padding
            )
            encoded = self.encoder_norm(encoded)
            subject = encoded[:, 0]
            modalities = encoded[:, 1:].masked_fill(
                summary_padding.unsqueeze(-1), 0.0
            )
            return modalities, summary_padding, subject

        encoded = self.encoder(summaries, src_key_padding_mask=summary_padding)
        encoded = self.encoder_norm(encoded)
        encoded = encoded.masked_fill(summary_padding.unsqueeze(-1), 0.0)
        return encoded, summary_padding, None

    def _token_random_mask(self, real, generator=None):
        scores = torch.rand(real.shape, device=real.device, generator=generator)
        scores = scores.masked_fill(~real, float("inf"))
        n_real = real.sum(dim=1)
        n_mask = torch.floor(n_real.float() * self.mask_ratio).long().clamp(min=1)
        n_mask = torch.minimum(n_mask, (n_real - 1).clamp(min=1))
        ranks = scores.argsort(dim=1).argsort(dim=1)
        return (ranks < n_mask.unsqueeze(1)) & real

    def _modality_random_mask(self, real, mod_ids, generator=None):
        """Give each represented modality the same per-token mask probability."""
        mask = torch.zeros_like(real)
        for batch_index in range(len(real)):
            for mod_idx in MODALITY_IDX.values():
                indices = (real[batch_index] & (mod_ids[batch_index] == mod_idx)).nonzero(
                    as_tuple=False
                ).flatten()
                count = len(indices)
                if count == 0:
                    continue
                expected = count * self.mask_ratio
                n_mask = int(expected)
                if torch.rand((), device=real.device, generator=generator) < expected - n_mask:
                    n_mask += 1
                n_mask = min(n_mask, count)
                if n_mask:
                    order = torch.randperm(count, device=real.device, generator=generator)
                    mask[batch_index, indices[order[:n_mask]]] = True

            subject_indices = real[batch_index].nonzero(as_tuple=False).flatten()
            if len(subject_indices) and not mask[batch_index].any():
                choice = torch.randint(
                    len(subject_indices), (), device=real.device, generator=generator
                )
                mask[batch_index, subject_indices[choice]] = True
            if len(subject_indices) > 1 and mask[batch_index, subject_indices].all():
                masked_indices = subject_indices[mask[batch_index, subject_indices]]
                choice = torch.randint(
                    len(masked_indices), (), device=real.device, generator=generator
                )
                mask[batch_index, masked_indices[choice]] = False
        return mask

    def random_mask(self, real, generator=None, mod_ids=None):
        if self.masking_mode == "token":
            return self._token_random_mask(real, generator)
        if mod_ids is None:
            raise ValueError("mod_ids are required for modality-balanced masking")
        return self._modality_random_mask(real, mod_ids, generator)

    def forward(self, raw, mod_ids, visit_ids, slot_ids, padding, mask=None, generator=None):
        position = self._positions(mod_ids, visit_ids, slot_ids)
        tokens = self.project_tokens(raw, mod_ids, padding)
        tokens = tokens + position
        if mask is None:
            mask = self.random_mask(~padding, generator=generator, mod_ids=mod_ids)
        # Preserve modality and visit identity at masked positions.
        tokens = torch.where(
            mask.unsqueeze(-1),
            self.mask_token.view(1, 1, -1) + position,
            tokens,
        )
        if self.fusion_mode == "modality_summary":
            summaries, summary_padding = self._summarize_modalities(tokens, mod_ids, padding)
            if self.pooling_mode == "fusion_token":
                fusion = self.fusion_token.expand(len(summaries), -1, -1)
                encoder_input = torch.cat([fusion, summaries], dim=1)
                fusion_padding = torch.zeros(
                    len(summaries), 1, dtype=torch.bool, device=summaries.device
                )
                encoder_padding = torch.cat(
                    [fusion_padding, summary_padding], dim=1
                )
                encoded = self.encoder(
                    encoder_input, src_key_padding_mask=encoder_padding
                )
                encoded = self.encoder_norm(encoded)
                # Every reconstruction must pass through the same shared
                # foundation representation. Position identifies the target.
                token_context = encoded[:, :1].expand(-1, raw.shape[1], -1)
            else:
                encoded = self.encoder(summaries, src_key_padding_mask=summary_padding)
                encoded = self.encoder_norm(encoded)
                token_context = encoded.gather(
                    1, mod_ids.unsqueeze(-1).expand(-1, -1, self.d_model)
                )
            decoded = self.decoder_projection(token_context + position)
        else:
            encoded = self.encoder(tokens, src_key_padding_mask=padding)
            encoded = self.encoder_norm(encoded)
            decoded = self.decoder_projection(encoded)
        decoded = self.decoder(decoded, src_key_padding_mask=padding)
        decoded = self.decoder_norm(decoded)

        modality_losses = []
        for modality, mod_idx in MODALITY_IDX.items():
            selected = mask & (mod_ids == mod_idx)
            if not selected.any():
                continue
            dim = MODALITY_DIMS[modality]
            prediction = self.reconstruction_heads[modality](decoded[selected])
            target = raw[selected][:, :dim]
            modality_losses.append(F.smooth_l1_loss(prediction, target))
        if not modality_losses:
            return raw.sum() * 0.0
        # Equal weight per represented modality prevents high-token-count or
        # high-dimensional modalities from dominating the objective.
        return torch.stack(modality_losses).mean()

    def encode(self, raw, mod_ids, visit_ids, slot_ids, padding):
        if self.fusion_mode == "modality_summary":
            modalities, _, subject = self._encode_modality_summaries(
                raw, mod_ids, visit_ids, slot_ids, padding
            )
            if subject is not None:
                return subject.unsqueeze(1)
            return modalities
        tokens = self.project_tokens(raw, mod_ids, padding)
        tokens = tokens + self._positions(mod_ids, visit_ids, slot_ids)
        encoded = self.encoder(tokens, src_key_padding_mask=padding)
        encoded = self.encoder_norm(encoded)
        return encoded.masked_fill(padding.unsqueeze(-1), 0.0)

    def encode_modalities(self, raw, mod_ids, visit_ids, slot_ids, padding):
        """Return contextual modality tokens and their availability mask.

        The output has one fixed slot per entry in ``MODALITY_IDX``.  Available
        slots contain contextualized diagnostic states; unavailable slots are
        zero and marked ``False`` in ``present``. For fusion-token checkpoints,
        these states are not the shared foundation representation used by the
        primary downstream path.
        """
        if self.fusion_mode != "modality_summary":
            raise ValueError(
                "encode_modalities requires a checkpoint with "
                "fusion_mode='modality_summary'"
            )
        encoded, summary_padding, _ = self._encode_modality_summaries(
            raw, mod_ids, visit_ids, slot_ids, padding
        )
        return encoded, ~summary_padding

    def embed_subject(self, raw, mod_ids, visit_ids, slot_ids, padding):
        if self.fusion_mode == "modality_summary":
            encoded, summary_padding, subject = self._encode_modality_summaries(
                raw, mod_ids, visit_ids, slot_ids, padding
            )
            if subject is not None:
                return subject
            present = ~summary_padding
            weights = present.unsqueeze(-1).to(encoded.dtype)
            return (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)

        encoded = self.encode(raw, mod_ids, visit_ids, slot_ids, padding)
        real = (~padding).unsqueeze(-1).to(encoded.dtype)
        if self.pooling_mode == "token_mean":
            return (encoded * real).sum(dim=1) / real.sum(dim=1).clamp(min=1)

        modality_vectors = []
        modality_present = []
        for mod_idx in MODALITY_IDX.values():
            selected = ((mod_ids == mod_idx) & ~padding).unsqueeze(-1).to(encoded.dtype)
            modality_vectors.append(
                (encoded * selected).sum(dim=1) / selected.sum(dim=1).clamp(min=1)
            )
            modality_present.append(selected.any(dim=1))
        modality_vectors = torch.stack(modality_vectors, dim=1)
        modality_present = torch.stack(modality_present, dim=1).to(encoded.dtype)
        return (
            modality_vectors * modality_present
        ).sum(dim=1) / modality_present.sum(dim=1).clamp(min=1)
