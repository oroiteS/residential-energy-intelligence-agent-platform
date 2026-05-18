"""PyTorch Lightning LSTM 残差模型。"""

from __future__ import annotations

from typing import Any, Sequence

import torch
from lightning import LightningModule
from torch import nn


class ResidualLSTMNet(nn.Module):
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


class LSTMResidualForecaster(LightningModule):
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
        self.model = ResidualLSTMNet(
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
        self.register_buffer("target_mean", torch.tensor(target_mean_list, dtype=torch.float32))
        self.register_buffer("target_scale", torch.tensor(target_scale_list, dtype=torch.float32))

    def forward(self, sequence: torch.Tensor, future: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        return self.model(sequence, future, static)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        prediction = self(batch["sequence"], batch["future"], batch["static"])
        loss = self.loss_fn(prediction, batch["residual_target"])
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        self._shared_eval_step(batch, prefix="valid")

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        self._shared_eval_step(batch, prefix="test")

    def predict_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self.predict_final(batch)

    def predict_final(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        residual_scaled = self(batch["sequence"], batch["future"], batch["static"])
        residual = self.inverse_residual(residual_scaled)
        return batch["baseline"] + residual

    def inverse_residual(self, residual_scaled: torch.Tensor) -> torch.Tensor:
        return residual_scaled * self.target_scale + self.target_mean

    def _shared_eval_step(self, batch: dict[str, torch.Tensor], *, prefix: str) -> None:
        residual_scaled = self(batch["sequence"], batch["future"], batch["static"])
        residual_loss = self.loss_fn(residual_scaled, batch["residual_target"])
        final_prediction = batch["baseline"] + self.inverse_residual(residual_scaled)
        error = final_prediction - batch["actual"]
        mse = torch.mean(error**2)
        rmse = torch.sqrt(mse)
        mae = torch.mean(torch.abs(error))
        self.log(f"{prefix}_residual_loss", residual_loss, on_step=False, on_epoch=True)
        self.log(f"{prefix}_mse", mse, on_step=False, on_epoch=True)
        self.log(f"{prefix}_rmse", rmse, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}_mae", mae, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
