"""Temporal GRU classifier over normalized hand-landmark sequences."""

from __future__ import annotations

import torch
from torch import nn


class GestureGRU(nn.Module):
    def __init__(
        self,
        num_classes: int,
        input_size: int = 63,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        if sequences.ndim != 3:
            raise ValueError("input must have shape [batch, time, features]")
        _, hidden = self.gru(sequences)
        return self.head(hidden[-1])
