"""Small task head for frozen shared subject representations."""

from __future__ import annotations

import torch
import torch.nn as nn


class DownstreamHead(nn.Module):
    def __init__(self, stage2_dim: int = 256, task_genetic_dim: int = 0,
                 hidden_dims: tuple[int, ...] = (128, 64), dropout: float = 0.30,
                 projection_dim: int = 128, use_layer_norm: bool = True,
                 residual_linear: bool = False):
        super().__init__()
        self.stage2_dim = stage2_dim
        self.task_genetic_dim = task_genetic_dim
        self.residual_linear = residual_linear
        stage2_layers: list[nn.Module] = []
        if use_layer_norm:
            stage2_layers.append(nn.LayerNorm(stage2_dim))
        stage2_layers.extend([
            nn.Linear(stage2_dim, projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        ])
        self.stage2_branch = nn.Sequential(*stage2_layers)
        if task_genetic_dim:
            genetic_layers: list[nn.Module] = []
            if use_layer_norm:
                genetic_layers.append(nn.LayerNorm(task_genetic_dim))
            genetic_layers.extend([
                nn.Linear(task_genetic_dim, projection_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            self.task_genetic_branch = nn.Sequential(*genetic_layers)
            width = 2 * projection_dim
        else:
            self.task_genetic_branch = None
            width = projection_dim
        layers: list[nn.Module] = []
        for hidden in hidden_dims:
            layers.extend([nn.Linear(width, hidden), nn.GELU(), nn.Dropout(dropout)])
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.head = nn.Sequential(*layers)
        self.linear_skip = (
            nn.Linear(stage2_dim + task_genetic_dim, 1)
            if residual_linear else None
        )

    def forward(self, stage2, task_genetic=None):
        value = self.stage2_branch(stage2)
        raw = stage2
        if self.task_genetic_branch is not None:
            if task_genetic is None:
                raise ValueError("task_genetic input is required by this checkpoint")
            value = torch.cat([value, self.task_genetic_branch(task_genetic)], dim=-1)
            raw = torch.cat([raw, task_genetic], dim=-1)
        output = self.head(value)
        if self.linear_skip is not None:
            output = output + self.linear_skip(raw)
        return output.squeeze(-1)
