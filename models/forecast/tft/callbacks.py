"""TFT 训练过程的控制台回调。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from lightning.pytorch.callbacks import Callback
except ModuleNotFoundError:  # pragma: no cover - 兼容旧包名
    from pytorch_lightning.callbacks import Callback  # type: ignore[no-redef]


def _to_float(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "item"):
        value = value.item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class CompactTrainingCallback(Callback):
    """输出简洁、可读的训练状态信息。"""

    def __init__(
        self,
        runtime_summary: dict[str, Any],
        split_summary: dict[str, int],
        output_dir: Path,
    ) -> None:
        super().__init__()
        self.runtime_summary = runtime_summary
        self.split_summary = split_summary
        self.output_dir = output_dir

    def on_fit_start(self, trainer, pl_module) -> None:  # type: ignore[no-untyped-def]
        print("=" * 72)
        print("TFT 训练开始")
        print(
            "设备: "
            f"{self.runtime_summary.get('cuda_device_name') or 'cpu/mps'} | "
            f"precision={self.runtime_summary.get('precision')} | "
            f"batch_size={self.runtime_summary.get('batch_size')} | "
            f"num_workers={self.runtime_summary.get('num_workers')}"
        )
        print(
            "数据集: "
            f"train={self.split_summary.get('train_samples', 0)} | "
            f"val={self.split_summary.get('val_samples', 0)} | "
            f"test={self.split_summary.get('test_samples', 0)}"
        )
        print(f"输出目录: {self.output_dir}")
        print("=" * 72)

    def on_train_epoch_start(self, trainer, pl_module) -> None:  # type: ignore[no-untyped-def]
        print(f"[Epoch {trainer.current_epoch + 1}/{trainer.max_epochs}] 开始训练")

    def on_validation_epoch_end(self, trainer, pl_module) -> None:  # type: ignore[no-untyped-def]
        if trainer.sanity_checking:
            return
        metrics = trainer.callback_metrics
        train_loss = _to_float(metrics, "train_loss_epoch")
        train_mae = _to_float(metrics, "train_mae_epoch")
        val_loss = _to_float(metrics, "val_loss")
        val_mae = _to_float(metrics, "val_mae")
        val_baseline_mae = _to_float(metrics, "val_baseline_mae")
        effective_guard_weight = _to_float(metrics, "train_effective_guard_weight")
        effective_gate_weight = _to_float(metrics, "train_effective_gate_weight")
        parts = [f"[Epoch {trainer.current_epoch + 1}/{trainer.max_epochs}] 结束"]
        if train_loss is not None:
            parts.append(f"train_loss={train_loss:.4f}")
        if train_mae is not None:
            parts.append(f"train_mae={train_mae:.2f}")
        if val_loss is not None:
            parts.append(f"val_loss={val_loss:.4f}")
        if val_mae is not None:
            parts.append(f"val_mae={val_mae:.2f}")
        if val_baseline_mae is not None:
            parts.append(f"baseline_mae={val_baseline_mae:.2f}")
        if val_mae is not None and val_baseline_mae is not None:
            improvement = val_baseline_mae - val_mae
            parts.append(f"improve={improvement:.2f}")
        if effective_guard_weight is not None:
            parts.append(f"guard_w={effective_guard_weight:.4f}")
        if effective_gate_weight is not None:
            parts.append(f"gate_w={effective_gate_weight:.4f}")
        optimizer = trainer.optimizers[0] if trainer.optimizers else None
        if optimizer is not None and optimizer.param_groups:
            parts.append(f"lr={optimizer.param_groups[0]['lr']:.6g}")
        print(" | ".join(parts))

    def on_fit_end(self, trainer, pl_module) -> None:  # type: ignore[no-untyped-def]
        checkpoint_callback = trainer.checkpoint_callback
        best_path = getattr(checkpoint_callback, "best_model_path", "") if checkpoint_callback else ""
        best_score = getattr(checkpoint_callback, "best_model_score", None) if checkpoint_callback else None
        best_score_text = (
            f"{float(best_score.item()):.4f}" if best_score is not None and hasattr(best_score, "item") else "None"
        )
        print("=" * 72)
        print(f"TFT 训练结束 | best_val_mae={best_score_text} | best_model={best_path or 'None'}")
        print("=" * 72)
