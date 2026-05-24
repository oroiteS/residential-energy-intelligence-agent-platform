from __future__ import annotations

from typing import Any, Sequence, cast

import torch
from torch import nn


torch_any = cast(Any, torch)


class DirectLSTMNet(nn.Module):
    """LSTM Direct 预测网络。

    sequence 输入经过 LSTM 编码为历史表示；
    future 日历特征和 static 历史统计特征会与历史表示拼接，
    最后通过 MLP 一次性输出未来 7 天的多个目标。
    """

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

        # PyTorch 的多层 LSTM 才会真正使用 dropout。
        # 单层时设为 0，避免参数看似启用但实际无效。
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=sequence_feature_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        # head 的输入由三部分拼接：
        # LSTM 历史编码 + 未来日历特征 + 历史静态统计特征。
        self.head = nn.Sequential(
            nn.Linear(hidden_size + future_feature_size + static_feature_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_size, output_size),
        )

    def forward(self, sequence: torch.Tensor, future: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        """执行前向推理，返回标准化空间中的预测值。"""

        _, (hidden, _) = self.lstm(sequence)
        encoded = hidden[-1]
        return self.head(torch_any.cat([encoded, future, static], dim=1))


class LSTMDirectForecaster(nn.Module):
    """带目标反标准化的预测封装器。

    训练时目标值经过标准化；服务端推理时需要把模型输出还原到真实 kWh 尺度。
    """

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

        # target_mean/target_scale 注册为 buffer。
        # 它们会随模型保存和加载，但不参与梯度训练。
        self.register_buffer("target_mean", torch_any.tensor([float(v) for v in target_mean], dtype=torch_any.float32))
        self.register_buffer("target_scale", torch_any.tensor([float(v) for v in target_scale], dtype=torch_any.float32))

    def forward(self, sequence: torch.Tensor, future: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        """返回标准化目标空间中的预测值。"""

        return self.model(sequence, future, static)

    def predict_final(self, sequence: torch.Tensor, future: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        """返回反标准化后的最终预测值。"""

        prediction_scaled = self(sequence, future, static)
        return prediction_scaled * self.target_scale + self.target_mean
