"""TCN 分类模型定义。"""

from __future__ import annotations

import torch
from torch import nn

from classification.TCN.constants import INPUT_CHANNELS


class Chomp1d(nn.Module):
    """裁掉因因果卷积补齐带来的尾部长度。"""

    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return tensor
        return tensor[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """带残差连接的 TCN 基础模块。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
        self.final_relu = nn.ReLU()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        out = self.conv1(tensor)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        residual = tensor if self.downsample is None else self.downsample(tensor)
        return self.final_relu(out + residual)


class TCNClassifier(nn.Module):
    """面向 96x3 日级样本的 TCN 四分类器。"""

    def __init__(
        self,
        input_channels: int = INPUT_CHANNELS,
        num_classes: int = 4,
        channel_sizes: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if channel_sizes is None:
            channel_sizes = [32, 64, 128]

        layers: list[nn.Module] = []
        in_channels = input_channels
        for block_index, out_channels in enumerate(channel_sizes):
            layers.append(
                TemporalBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=2 ** block_index,
                    dropout=dropout,
                )
            )
            in_channels = out_channels

        self.network = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channel_sizes[-1], channel_sizes[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(channel_sizes[-1] // 2, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(f"输入维度应为 [batch, seq_len, channels]，实际为 {tuple(features.shape)}")

        temporal_features = features.transpose(1, 2)
        encoded = self.network(temporal_features)
        pooled = self.pool(encoded)
        return self.classifier(pooled)
