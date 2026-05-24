"""PyTorch Lightning LSTM 直接预测模型。"""

from __future__ import annotations

from typing import Any, Sequence

import torch
from lightning import LightningModule
from torch import nn


class DirectLSTMNet(nn.Module):
    """LSTM Direct 网络。

    历史 30 天序列先通过 LSTM 编码，随后与未来日历特征和历史静态统计特征拼接，
    最后一次性预测未来 7 天的总量、峰时和谷时电量。
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

        # 单层 LSTM 中 PyTorch 不会使用 dropout，因此这里显式置为 0。
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=sequence_feature_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        # head 的输入 = LSTM 历史编码 + 未来日历特征 + 静态统计特征。
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


class LSTMDirectForecaster(LightningModule):
    """PyTorch Lightning 封装的 LSTM 直接预测器。

    LightningModule 负责训练步骤、验证步骤、测试步骤、优化器配置和日志记录。
    target_mean/target_scale 用于把标准化目标还原为真实 kWh。
    """

    def __init__(
        self,
        *,
        sequence_feature_size: int,
        future_feature_size: int,
        static_feature_size: int,
        output_size: int,
        model_config: dict[str, Any],
        learning_rate: float,
        weight_decay: float,
        target_mean: Sequence[float],
        target_scale: Sequence[float],
    ) -> None:
        super().__init__()
        target_mean_list = [float(value) for value in target_mean]
        target_scale_list = [float(value) for value in target_scale]
        self.save_hyperparameters(
            "sequence_feature_size",
            "future_feature_size",
            "static_feature_size",
            "output_size",
            "model_config",
            "learning_rate",
            "weight_decay",
            "target_mean",
            "target_scale",
        )
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
        self.loss_fn = nn.HuberLoss(delta=1.0)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # target_mean/target_scale 是标准化参数，不参与训练，但需要随 checkpoint 保存。
        self.register_buffer("target_mean", torch.tensor(target_mean_list, dtype=torch.float32))
        self.register_buffer("target_scale", torch.tensor(target_scale_list, dtype=torch.float32))
        self.target_mean: torch.Tensor
        self.target_scale: torch.Tensor

    def forward(self, sequence: torch.Tensor, future: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        return self.model(sequence, future, static)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        prediction = self(batch["sequence"], batch["future"], batch["static"])
        loss = self.loss_fn(prediction, batch["target"])
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        self._shared_eval_step(batch, prefix="valid")

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        self._shared_eval_step(batch, prefix="test")

    def predict_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self.predict_final(batch)

    def predict_final(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """返回反标准化后的 21 维预测结果。"""

        prediction_scaled = self(batch["sequence"], batch["future"], batch["static"])
        return self.inverse_target(prediction_scaled)

    def inverse_target(self, prediction_scaled: torch.Tensor) -> torch.Tensor:
        return prediction_scaled * self.target_scale + self.target_mean

    def _shared_eval_step(self, batch: dict[str, torch.Tensor], *, prefix: str) -> None:
        prediction_scaled = self(batch["sequence"], batch["future"], batch["static"])
        scaled_loss = self.loss_fn(prediction_scaled, batch["target"])
        final_prediction = self.inverse_target(prediction_scaled)
        error = final_prediction - batch["actual"]
        mse = torch.mean(error**2)
        rmse = torch.sqrt(mse)
        mae = torch.mean(torch.abs(error))
        self.log(f"{prefix}_scaled_loss", scaled_loss, on_step=False, on_epoch=True)
        self.log(f"{prefix}_mse", mse, on_step=False, on_epoch=True)
        self.log(f"{prefix}_rmse", rmse, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}_mae", mae, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
