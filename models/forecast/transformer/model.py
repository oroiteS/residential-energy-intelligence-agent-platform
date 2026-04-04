"""Patch-based direct multi-step Transformer 预测模型定义。

实现思路参考 PatchTST:
Yuqi Nie et al., "A Time Series is Worth 64 Words: Long-term Forecasting with
Transformers", ICLR 2023.
这里结合当前项目的 288 -> 96 负荷预测任务，采用 patch token 编码历史序列，
并通过直接预测头一次性输出未来 96 个点，避免自回归误差累积。
"""

from __future__ import annotations

import torch
from torch import nn

from forecast.transformer.constants import INPUT_LENGTH, TARGET_LENGTH


class PatchDirectTransformerForecaster(nn.Module):
    """面向 288->96 任务的 patch-based direct multi-step Transformer。"""

    def __init__(
        self,
        input_size: int = 1,
        d_model: int = 256,
        num_layers: int = 6,
        num_heads: int = 4,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        target_length: int = TARGET_LENGTH,
        patch_length: int = 16,
        patch_stride: int = 8,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.history_length = INPUT_LENGTH
        self.target_length = target_length
        self.patch_length = patch_length
        self.patch_stride = patch_stride
        self.num_patches = self._compute_num_patches()

        self.patch_projection = nn.Linear(input_size * patch_length, d_model)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.num_patches, d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.output_layer = nn.Sequential(
            nn.LayerNorm(self.num_patches * d_model),
            nn.Linear(self.num_patches * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, target_length),
        )
        self._reset_parameters()

    def _compute_num_patches(self) -> int:
        if self.patch_length <= 0:
            raise ValueError("patch_length 必须大于 0")
        if self.patch_stride <= 0:
            raise ValueError("patch_stride 必须大于 0")
        if self.patch_length > self.history_length:
            raise ValueError("patch_length 不能大于输入历史长度")
        return 1 + (self.history_length - self.patch_length) // self.patch_stride

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    def _patchify(self, features: torch.Tensor) -> torch.Tensor:
        patches = features.transpose(1, 2).unfold(
            dimension=2,
            size=self.patch_length,
            step=self.patch_stride,
        )
        patches = patches.permute(0, 2, 1, 3).contiguous()
        batch_size, patch_count, channel_count, patch_length = patches.shape
        if patch_count != self.num_patches:
            raise RuntimeError(
                f"patch 数量异常，期望 {self.num_patches}，实际 {patch_count}"
            )
        return patches.view(batch_size, patch_count, channel_count * patch_length)

    def _encode(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        encoded = self.transformer(
            patch_tokens + self.position_embedding[:, : patch_tokens.size(1), :]
        )
        if encoded.size(1) != self.num_patches:
            raise RuntimeError("Transformer 编码输出 patch 维度异常")
        return encoded

    def _validate_inputs(self, features: torch.Tensor) -> None:
        if features.ndim != 3:
            raise ValueError(
                f"输入维度应为 [batch, seq_len, channels]，实际为 {tuple(features.shape)}"
            )
        if features.size(1) != self.history_length:
            raise ValueError(
                f"输入序列长度应为 {self.history_length}，实际为 {features.size(1)}"
            )
        if features.size(-1) != self.input_size:
            raise ValueError(
                f"输入通道数应为 {self.input_size}，实际为 {features.size(-1)}"
            )

    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_inputs(features)
        patch_tokens = self.patch_projection(self._patchify(features))
        encoded = self._encode(patch_tokens)
        flattened = encoded.reshape(encoded.size(0), -1)
        return self.output_layer(flattened)
