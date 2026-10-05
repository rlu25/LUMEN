"""Perceiver-style masked regional genetic autoencoder."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneticRegionMAE(nn.Module):
    def __init__(self, n_regions: int, input_dim=64, d_model=256, n_latents=32,
                 n_heads=8, latent_depth=4, mask_ratio=0.40, dropout=0.1):
        super().__init__()
        self.n_regions = n_regions
        self.input_dim = input_dim
        self.d_model = d_model
        self.n_latents = n_latents
        self.mask_ratio = mask_ratio
        self.input_projection = nn.Linear(input_dim, d_model)
        self.region_embedding = nn.Embedding(n_regions, d_model)
        self.latents = nn.Parameter(torch.randn(1, n_latents, d_model) * 0.02)
        self.encoder_cross_attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.encoder_cross_norm = nn.LayerNorm(d_model)
        latent_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_model * 4, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.latent_encoder = nn.TransformerEncoder(latent_layer, latent_depth)
        self.latent_norm = nn.LayerNorm(d_model)
        self.decoder_cross_attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.decoder_norm = nn.LayerNorm(d_model)
        self.reconstruction_head = nn.Linear(d_model, input_dim)
        self.apply(self._initialize)
        nn.init.trunc_normal_(self.latents, std=0.02)

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=0.02)

    def random_mask(self, batch_size, device, generator=None):
        n_mask = max(1, min(self.n_regions - 1, int(self.n_regions * self.mask_ratio)))
        scores = torch.rand(batch_size, self.n_regions, device=device, generator=generator)
        return scores.argsort(dim=1).argsort(dim=1) < n_mask

    def encode(self, regions, hidden_mask=None):
        batch = len(regions)
        region_ids = torch.arange(self.n_regions, device=regions.device)
        tokens = self.input_projection(regions) + self.region_embedding(region_ids).unsqueeze(0)
        latents = self.latents.expand(batch, -1, -1)
        update, _ = self.encoder_cross_attention(
            latents, tokens, tokens, key_padding_mask=hidden_mask, need_weights=False,
        )
        latents = self.encoder_cross_norm(latents + update)
        latents = self.latent_encoder(latents)
        return self.latent_norm(latents)

    def forward(self, regions, mask=None, generator=None):
        if mask is None:
            mask = self.random_mask(len(regions), regions.device, generator)
        latents = self.encode(regions, hidden_mask=mask)
        batch_indices, region_indices = mask.nonzero(as_tuple=True)
        queries = self.region_embedding(region_indices).view(len(regions), -1, self.d_model)
        decoded, _ = self.decoder_cross_attention(
            queries, latents, latents, need_weights=False,
        )
        decoded = self.decoder_norm(queries + decoded)
        prediction = self.reconstruction_head(decoded).reshape(-1, self.input_dim)
        target = regions[batch_indices, region_indices]
        return F.smooth_l1_loss(prediction, target)
