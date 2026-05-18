from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import nn


class DirectLSTMNet(nn.Module):
    def __init__(
        self,
        *,
        sequence_feature_size: int,
        future_feature_size: int,
        static_feature_size: int,
        output_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        mlp_hidden_size: int,
    ) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=sequence_feature_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size + future_feature_size + static_feature_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_size, output_size),
        )

    def forward(self, sequence: torch.Tensor, future: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(sequence)
        encoded = hidden[-1]
        return self.head(torch.cat([encoded, future, static], dim=1))


class LSTMDirectForecaster(nn.Module):
    def __init__(
        self,
        *,
        sequence_feature_size: int,
        future_feature_size: int,
        static_feature_size: int,
        output_size: int,
        model_config: dict[str, Any],
        target_mean: Sequence[float],
        target_scale: Sequence[float],
    ) -> None:
        super().__init__()
        self.model = DirectLSTMNet(
            sequence_feature_size=sequence_feature_size,
            future_feature_size=future_feature_size,
            static_feature_size=static_feature_size,
            output_size=output_size,
            hidden_size=int(model_config["hidden_size"]),
            num_layers=int(model_config["num_layers"]),
            dropout=float(model_config["dropout"]),
            mlp_hidden_size=int(model_config["mlp_hidden_size"]),
        )
        self.register_buffer("target_mean", torch.tensor([float(v) for v in target_mean], dtype=torch.float32))
        self.register_buffer("target_scale", torch.tensor([float(v) for v in target_scale], dtype=torch.float32))

    def forward(self, sequence: torch.Tensor, future: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        return self.model(sequence, future, static)

    def predict_final(self, sequence: torch.Tensor, future: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        prediction_scaled = self(sequence, future, static)
        return prediction_scaled * self.target_scale + self.target_mean

