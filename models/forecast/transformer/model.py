"""PyTorch Lightning Transformer 直接预测模型。"""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch
from lightning import LightningModule
from torch import nn
from torch.optim.lr_scheduler import LambdaLR


class SinusoidalPositionalEncoding(nn.Module):
    """固定正弦位置编码，用于标识 30 天历史序列中的相对位置。"""

    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(max_len, d_model, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)
        self.encoding: torch.Tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.encoding[:, : x.size(1), :]


class DirectTransformerNet(nn.Module):
    """Transformer Direct 网络。

    历史 30 天序列先投影到 d_model 维，再加位置编码进入 Transformer Encoder；
    编码结果与未来日历特征、静态统计特征拼接后，一次性输出 21 个预测目标。
    """

    def __init__(
        self,
        *,
        sequence_feature_size: int,
        future_feature_size: int,
        static_feature_size: int,
        output_size: int,
        input_days: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        mlp_hidden_size: int,
        pooling: str,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model 必须能被 nhead 整除")
        if pooling not in {"mean", "last"}:
            raise ValueError("pooling 只能是 mean 或 last")

        self.pooling = pooling

        # 将每日原始特征映射到 Transformer 使用的 d_model 维空间。
        self.input_projection = nn.Linear(sequence_feature_size, d_model)
        self.position_encoding = SinusoidalPositionalEncoding(d_model=d_model, max_len=input_days)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model + future_feature_size + static_feature_size),
            nn.Linear(d_model + future_feature_size + static_feature_size, mlp_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_size, output_size),
        )

    def forward(self, sequence: torch.Tensor, future: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        encoded_sequence = self.input_projection(sequence)
        encoded_sequence = self.position_encoding(encoded_sequence)
        encoded_sequence = self.encoder(encoded_sequence)

        # pooling 决定如何把 30 天编码序列压缩为一个历史表示。
        # mean 使用全窗口平均，last 使用最后一天的编码。
        if self.pooling == "last":
            encoded = encoded_sequence[:, -1, :]
        else:
            encoded = encoded_sequence.mean(dim=1)
        return self.head(torch.cat([encoded, future, static], dim=1))


class TransformerDirectForecaster(LightningModule):
    def __init__(
        self,
        *,
        sequence_feature_size: int,
        future_feature_size: int,
        static_feature_size: int,
        output_size: int,
        input_days: int,
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
            "input_days",
            "model_config",
            "learning_rate",
            "weight_decay",
            "target_mean",
            "target_scale",
        )
        self.model = DirectTransformerNet(
            sequence_feature_size=sequence_feature_size,
            future_feature_size=future_feature_size,
            static_feature_size=static_feature_size,
            output_size=output_size,
            input_days=input_days,
            d_model=int(model_config["d_model"]),
            nhead=int(model_config["nhead"]),
            num_layers=int(model_config["num_layers"]),
            dim_feedforward=int(model_config["dim_feedforward"]),
            dropout=float(model_config["dropout"]),
            mlp_hidden_size=int(model_config["mlp_hidden_size"]),
            pooling=str(model_config["pooling"]),
        )
        self.loss_fn = nn.HuberLoss(delta=1.0)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
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

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # 前 10% steps 线性 warmup，之后 cosine decay 到 lr * 0.1。
        # Transformer 对学习率更敏感，warmup 可以降低训练初期不稳定。
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = max(1, int(total_steps * 0.1))

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return current_step / warmup_steps
            progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
