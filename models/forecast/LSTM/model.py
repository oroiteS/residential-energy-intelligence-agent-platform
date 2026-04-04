"""Seq2Seq LSTM 预测模型定义。"""

from __future__ import annotations

import random

import torch
from torch import nn

from forecast.LSTM.constants import TARGET_LENGTH


class Seq2SeqLSTMForecaster(nn.Module):
    """基于编码器-解码器结构的多步负荷预测模型。"""

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        target_length: int = TARGET_LENGTH,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.target_length = target_length
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.decoder = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        features: torch.Tensor,
        decoder_targets: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(
                f"输入维度应为 [batch, seq_len, channels]，实际为 {tuple(features.shape)}"
            )
        if features.size(-1) != self.input_size:
            raise ValueError(
                f"输入通道数应为 {self.input_size}，实际为 {features.size(-1)}"
            )
        if decoder_targets is not None and decoder_targets.ndim != 2:
            raise ValueError(
                "decoder_targets 维度应为 [batch, target_length]"
            )

        batch_size = features.size(0)
        _, encoder_state = self.encoder(features)

        decoder_input = torch.zeros(
            batch_size,
            1,
            1,
            device=features.device,
            dtype=features.dtype,
        )
        hidden, cell = encoder_state

        predictions: list[torch.Tensor] = []
        for step_index in range(self.target_length):
            decoder_output, (hidden, cell) = self.decoder(
                decoder_input, (hidden, cell)
            )
            current_prediction = self.output_layer(decoder_output[:, -1, :])
            predictions.append(current_prediction)

            use_teacher_forcing = (
                self.training
                and decoder_targets is not None
                and teacher_forcing_ratio > 0.0
                and random.random() < teacher_forcing_ratio
            )
            if use_teacher_forcing:
                decoder_input = decoder_targets[:, step_index].view(
                    batch_size, 1, 1
                )
            else:
                decoder_input = current_prediction.unsqueeze(1)

        return torch.cat(predictions, dim=1)
