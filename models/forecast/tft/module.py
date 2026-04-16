"""LightningModule 封装。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.nn import functional as F

try:
    import lightning as L
except ModuleNotFoundError:  # pragma: no cover - 兼容旧包名
    import pytorch_lightning as L  # type: ignore[no-redef]

from forecast.tft.model import TemporalFusionTransformer
from forecast.tft.stat_tests import compare_model_vs_baseline
from forecast.tft.visualization import (
    plot_prediction_examples,
    plot_test_household_metrics,
)


class TftForecastModule(L.LightningModule):
    """基于 baseline-aware residual 的 TFT v2 训练模块。"""

    def __init__(
        self,
        data_config: dict[str, Any],
        model_config: dict[str, Any],
        loss_config: dict[str, Any],
        feature_spec: dict[str, Any],
        learning_rate: float,
        weight_decay: float,
        test_output_dir: str | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        encoder_dim = int(feature_spec["feature_dim"])
        decoder_dim = 5
        target_length = int(feature_spec["target_length"])
        profile_dim = int(feature_spec["profile_dim"])
        baseline_stat_dim = int(feature_spec["baseline_stat_dim"])

        self.loss_diff_weight = float(loss_config["diff_weight"])
        self.huber_delta = float(loss_config["huber_delta"])
        self.peak_weight = float(loss_config["peak_weight"])
        self.peak_quantile = float(loss_config["peak_quantile"])
        self.ramp_weight = float(loss_config["ramp_weight"])
        self.guard_weight = float(loss_config["guard_weight"])
        self.guard_margin = float(loss_config["guard_margin"])
        self.gate_regularization = float(loss_config["gate_regularization"])
        self.regularization_warmup_epochs = int(loss_config["regularization_warmup_epochs"])
        self.regularization_ramp_epochs = int(loss_config["regularization_ramp_epochs"])
        self.test_output_dir = Path(test_output_dir) if test_output_dir else None

        self.model = TemporalFusionTransformer(
            encoder_input_dim=encoder_dim,
            decoder_input_dim=decoder_dim,
            hidden_size=int(model_config["hidden_size"]),
            lstm_layers=int(model_config["lstm_layers"]),
            attention_heads=int(model_config["attention_heads"]),
            dropout=float(model_config["dropout"]),
            prediction_length=target_length,
            profile_dim=profile_dim,
            baseline_stat_dim=baseline_stat_dim,
            router_prior_weight=float(model_config.get("router_prior_weight", 1.0)),
            global_gate_bias_init=float(model_config.get("global_gate_bias_init", -1.0)),
            local_gate_bias_init=float(model_config.get("local_gate_bias_init", -1.0)),
            return_attention_weights=bool(model_config.get("return_attention_weights", False)),
        )
        self._test_records: list[dict[str, Any]] = []
        self._prediction_examples: list[dict[str, Any]] = []
        self._max_prediction_examples = 12

    def forward(
        self,
        encoder_cont: torch.Tensor,
        decoder_known: torch.Tensor,
        profile_prior: torch.Tensor,
        baseline_stats: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return self.model(
            encoder_cont=encoder_cont,
            decoder_known=decoder_known,
            profile_prior=profile_prior,
            baseline_stats=baseline_stats,
        )

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        outputs = self._shared_step(batch, stage="train")
        return outputs["loss"]

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        self._shared_step(batch, stage="val")

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        outputs = self._shared_step(batch, stage="test")
        sample_mae = self._to_numpy(outputs["sample_mae"])
        baseline_mae = self._to_numpy(outputs["sample_baseline_mae"])
        sample_rmse = self._to_numpy(outputs["sample_rmse"])
        baseline_rmse = self._to_numpy(outputs["sample_baseline_rmse"])
        prediction_array = self._to_numpy(outputs["prediction_denorm"])
        target_array = self._to_numpy(outputs["target_denorm"])
        baseline_array = self._to_numpy(outputs["baseline_denorm"])
        global_gate = self._to_numpy(outputs["global_gate"])
        local_gate_mean = self._to_numpy(outputs["local_gate"]).mean(axis=-1)
        expert_weights = self._to_numpy(outputs["expert_weights"])
        profile_prior = self._to_numpy(outputs["profile_prior"])
        for index, sample_id in enumerate(batch["sample_id"]):
            top_expert_index = int(expert_weights[index].argmax())
            self._test_records.append(
                {
                    "sample_id": str(sample_id),
                    "house_id": str(batch["house_id"][index]),
                    "target_start": str(batch["target_start"][index]),
                    "model_mae": float(sample_mae[index]),
                    "baseline_mae": float(baseline_mae[index]),
                    "model_rmse": float(sample_rmse[index]),
                    "baseline_rmse": float(baseline_rmse[index]),
                    "global_gate": float(global_gate[index]),
                    "local_gate_mean": float(local_gate_mean[index]),
                    "top_expert_index": top_expert_index,
                    "top_expert_weight": float(expert_weights[index, top_expert_index]),
                }
            )
            if len(self._prediction_examples) < self._max_prediction_examples:
                self._prediction_examples.append(
                    {
                        "sample_id": str(sample_id),
                        "house_id": str(batch["house_id"][index]),
                        "target_start": str(batch["target_start"][index]),
                        "model_mae": float(sample_mae[index]),
                        "baseline_mae": float(baseline_mae[index]),
                        "global_gate": float(global_gate[index]),
                        "local_gate_mean": float(local_gate_mean[index]),
                        "expert_weights": expert_weights[index].astype(float).tolist(),
                        "profile_prior": profile_prior[index].astype(float).tolist(),
                        "prediction": prediction_array[index].astype(float).tolist(),
                        "target": target_array[index].astype(float).tolist(),
                        "baseline": baseline_array[index].astype(float).tolist(),
                    }
                )

    def on_test_epoch_start(self) -> None:
        self._test_records = []
        self._prediction_examples = []

    def on_test_epoch_end(self) -> None:
        if not self._test_records or self.test_output_dir is None:
            return

        self.test_output_dir.mkdir(parents=True, exist_ok=True)
        sample_df = pd.DataFrame(self._test_records)
        sample_path = self.test_output_dir / "test_sample_metrics.csv"
        sample_df.to_csv(sample_path, index=False)

        per_house_df = (
            sample_df.groupby("house_id", as_index=False)
            .agg(
                model_mae=("model_mae", "mean"),
                baseline_mae=("baseline_mae", "mean"),
                model_rmse=("model_rmse", "mean"),
                baseline_rmse=("baseline_rmse", "mean"),
                sample_count=("sample_id", "size"),
            )
            .sort_values("house_id")
            .reset_index(drop=True)
        )
        per_house_df["mae_improvement"] = per_house_df["baseline_mae"] - per_house_df["model_mae"]
        per_house_df["rmse_improvement"] = per_house_df["baseline_rmse"] - per_house_df["model_rmse"]
        house_path = self.test_output_dir / "test_household_metrics.csv"
        per_house_df.to_csv(house_path, index=False)
        plot_test_household_metrics(per_house_df=per_house_df, output_dir=self.test_output_dir)

        stats_summary = compare_model_vs_baseline(per_house_df)
        summary_path = self.test_output_dir / "test_statistical_summary.json"
        summary_path.write_text(
            json.dumps(stats_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        plot_prediction_examples(
            prediction_examples=self._prediction_examples,
            output_dir=self.test_output_dir,
        )

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(self.hparams["learning_rate"]),
            weight_decay=float(self.hparams["weight_decay"]),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_mae",
            },
        }

    def _shared_step(
        self,
        batch: dict[str, Any],
        stage: str,
    ) -> dict[str, torch.Tensor]:
        outputs = self.forward(
            encoder_cont=batch["encoder_cont"].float(),
            decoder_known=batch["decoder_known"].float(),
            profile_prior=batch["profile_prior"].float(),
            baseline_stats=batch["baseline_stats"].float(),
        )
        baseline = batch["baseline"].float()
        target = batch["target"].float()
        prediction = baseline + outputs["residual_prediction"]

        point_weights = self._build_point_weights(target)
        huber_map = F.huber_loss(
            prediction,
            target,
            delta=self.huber_delta,
            reduction="none",
        )
        huber_loss = (huber_map * point_weights).mean()

        prediction_diff = torch.diff(prediction, dim=-1)
        target_diff = torch.diff(target, dim=-1)
        diff_loss = F.l1_loss(prediction_diff, target_diff, reduction="mean")

        baseline_error = torch.abs(baseline - target)
        model_error = torch.abs(prediction - target)
        guard_loss = F.relu(model_error - baseline_error - self.guard_margin).mean()

        global_gate_mean = outputs["global_gate"].mean()
        local_gate_mean = outputs["local_gate"].mean()
        gate_loss = global_gate_mean + local_gate_mean
        regularization_scale = self._resolve_regularization_scale(stage=stage)
        effective_guard_weight = self.guard_weight * regularization_scale
        effective_gate_weight = self.gate_regularization * regularization_scale

        loss = (
            huber_loss
            + self.loss_diff_weight * diff_loss
            + effective_guard_weight * guard_loss
            + effective_gate_weight * gate_loss
        )

        scale = batch["scale"].float().unsqueeze(-1)
        prediction_denorm = prediction * scale
        target_denorm = target * scale
        baseline_denorm = baseline * scale

        absolute_error = torch.abs(prediction_denorm - target_denorm)
        baseline_absolute_error = torch.abs(baseline_denorm - target_denorm)
        squared_error = torch.square(prediction_denorm - target_denorm)
        baseline_squared_error = torch.square(baseline_denorm - target_denorm)
        diff_absolute_error = torch.abs(
            torch.diff(prediction_denorm, dim=-1) - torch.diff(target_denorm, dim=-1)
        )
        router_entropy = -torch.sum(
            outputs["expert_weights"] * torch.log(outputs["expert_weights"].clamp_min(1e-6)),
            dim=-1,
        ).mean()
        metrics = {
            f"{stage}_loss": loss,
            f"{stage}_huber_loss": huber_loss.detach(),
            f"{stage}_diff_loss": diff_loss.detach(),
            f"{stage}_guard_loss": guard_loss.detach(),
            f"{stage}_gate_loss": gate_loss.detach(),
            f"{stage}_mae": absolute_error.mean(),
            f"{stage}_rmse": torch.sqrt(squared_error.mean()),
            f"{stage}_diff_mae": diff_absolute_error.mean(),
            f"{stage}_baseline_mae": baseline_absolute_error.mean(),
            f"{stage}_baseline_rmse": torch.sqrt(baseline_squared_error.mean()),
            f"{stage}_global_gate_mean": global_gate_mean.detach(),
            f"{stage}_local_gate_mean": local_gate_mean.detach(),
            f"{stage}_router_entropy": router_entropy.detach(),
            f"{stage}_regularization_scale": prediction.new_tensor(regularization_scale),
            f"{stage}_effective_guard_weight": prediction.new_tensor(effective_guard_weight),
            f"{stage}_effective_gate_weight": prediction.new_tensor(effective_gate_weight),
        }
        for name, value in metrics.items():
            self.log(
                name,
                value,
                prog_bar=name in {f"{stage}_loss", f"{stage}_mae"},
                on_step=stage == "train",
                on_epoch=True,
                batch_size=prediction.size(0),
            )

        return {
            "loss": loss,
            "sample_mae": absolute_error.mean(dim=-1),
            "sample_baseline_mae": baseline_absolute_error.mean(dim=-1),
            "sample_rmse": torch.sqrt(squared_error.mean(dim=-1)),
            "sample_baseline_rmse": torch.sqrt(baseline_squared_error.mean(dim=-1)),
            "prediction_denorm": prediction_denorm,
            "target_denorm": target_denorm,
            "baseline_denorm": baseline_denorm,
            "global_gate": outputs["global_gate"],
            "local_gate": outputs["local_gate"],
            "expert_weights": outputs["expert_weights"],
            "profile_prior": outputs["profile_prior"],
        }

    @staticmethod
    def _to_numpy(tensor: torch.Tensor):
        return tensor.detach().float().cpu().numpy()

    def _build_point_weights(self, target: torch.Tensor) -> torch.Tensor:
        peak_threshold = torch.quantile(
            target.detach(),
            q=self.peak_quantile,
            dim=-1,
            keepdim=True,
        )
        peak_mask = (target >= peak_threshold).to(target.dtype)
        ramp = torch.zeros_like(target)
        ramp[..., 1:] = torch.abs(target[..., 1:] - target[..., :-1])
        ramp_scale = ramp / (ramp.mean(dim=-1, keepdim=True) + 1e-6)
        ramp_scale = torch.clamp(ramp_scale, max=3.0)
        return 1.0 + self.peak_weight * peak_mask + self.ramp_weight * ramp_scale

    def _resolve_regularization_scale(self, stage: str) -> float:
        if stage == "test":
            return 1.0
        current_epoch_index = int(self.current_epoch) + 1
        if current_epoch_index <= self.regularization_warmup_epochs:
            return 0.0
        if self.regularization_ramp_epochs == 0:
            return 1.0
        ramp_progress = (
            current_epoch_index - self.regularization_warmup_epochs
        ) / float(self.regularization_ramp_epochs)
        return float(max(0.0, min(1.0, ramp_progress)))
