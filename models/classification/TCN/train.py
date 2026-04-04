"""TCN 分类训练入口。"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from classification.TCN.config import DEFAULT_CONFIG_PATH, detect_device, load_experiment_config
from classification.TCN.engine import (
    build_model,
    create_data_loaders,
    create_split_datasets,
    evaluate,
    save_checkpoint,
    save_json_summary,
    set_seed,
    train_one_epoch,
)


def build_optimizer(train_config, model: nn.Module) -> torch.optim.Optimizer:
    optimizer_name = train_config.optimizer.lower()
    if optimizer_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
        )
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
        )
    raise ValueError(f"不支持的优化器: {train_config.optimizer}")


def build_scheduler(train_config, optimizer: torch.optim.Optimizer):
    scheduler_name = train_config.scheduler.lower()
    if scheduler_name == "none":
        return None
    if scheduler_name == "cosine_annealing_warm_restarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=train_config.scheduler_t0,
            T_mult=train_config.scheduler_t_mult,
            eta_min=train_config.scheduler_eta_min,
        )
    raise ValueError(f"不支持的学习率调度器: {train_config.scheduler}")


def plot_training_history(
    history: list[dict[str, float]],
    output_dir: Path,
    best_epoch: int,
) -> dict[str, str]:
    if not history:
        raise ValueError("history 为空，无法绘制训练曲线")

    epochs = [int(item["epoch"]) for item in history]
    train_loss = [float(item["train_loss"]) for item in history]
    val_loss = [float(item["val_loss"]) for item in history]
    train_accuracy = [float(item["train_accuracy"]) for item in history]
    val_accuracy = [float(item["val_accuracy"]) for item in history]
    train_macro_f1 = [float(item["train_macro_f1"]) for item in history]
    val_macro_f1 = [float(item["val_macro_f1"]) for item in history]
    learning_rates = [float(item["learning_rate"]) for item in history]

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "result.png"

    figure, axes = plt.subplots(4, 1, figsize=(12, 20), sharex=True)
    metric_specs = [
        ("Loss", train_loss, val_loss),
        ("Accuracy", train_accuracy, val_accuracy),
        ("Macro-F1", train_macro_f1, val_macro_f1),
    ]
    for axis, (metric_name, train_values, val_values) in zip(axes, metric_specs):
        axis.plot(epochs, train_values, label=f"train_{metric_name.lower()}", linewidth=2)
        axis.plot(epochs, val_values, label=f"val_{metric_name.lower()}", linewidth=2)
        axis.axvline(best_epoch, color="red", linestyle="--", linewidth=1.5, label=f"best_epoch={best_epoch}")
        axis.set_ylabel(metric_name)
        axis.grid(True, linestyle="--", alpha=0.4)
        axis.legend()

    lr_axis = axes[3]
    lr_axis.plot(epochs, learning_rates, label="learning_rate", linewidth=2, color="green")
    lr_axis.axvline(best_epoch, color="red", linestyle="--", linewidth=1.5, label=f"best_epoch={best_epoch}")
    lr_axis.set_ylabel("LR")
    lr_axis.set_xlabel("Epoch")
    lr_axis.grid(True, linestyle="--", alpha=0.4)
    lr_axis.legend()

    figure.suptitle("TCN Training Curves", fontsize=16)
    figure.tight_layout(rect=[0, 0, 1, 0.98])
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return {"result": str(output_path)}


def run_training(experiment_config) -> dict[str, object]:
    train_config = experiment_config.train
    model_config = experiment_config.model
    set_seed(train_config.seed)

    device_name = detect_device()
    device = torch.device(device_name)
    train_dataset, val_dataset, _ = create_split_datasets(
        data_config=experiment_config.data,
        seed=train_config.seed,
    )
    train_loader, val_loader, _ = create_data_loaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=val_dataset,
        batch_size=train_config.batch_size,
    )

    model = build_model(model_config).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=train_config.label_smoothing)
    optimizer = build_optimizer(train_config, model)
    scheduler = build_scheduler(train_config, optimizer)

    best_val_f1 = -1.0
    best_metrics: dict[str, float] = {}
    best_state_dict = None
    history: list[dict[str, float]] = []
    epochs_without_improvement = 0
    stopped_early = False
    stop_epoch: int | None = None

    start_time = time.time()
    for epoch in range(1, train_config.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        current_learning_rate = float(optimizer.param_groups[0]["lr"])

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "learning_rate": current_learning_rate,
        }
        history.append(epoch_record)

        print(
            f"epoch={epoch:02d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"train_f1={train_metrics['macro_f1']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['macro_f1']:.4f} "
            f"lr={current_learning_rate:.6f}"
        )

        if val_metrics["macro_f1"] > best_val_f1 + train_config.early_stopping_min_delta:
            best_val_f1 = val_metrics["macro_f1"]
            epochs_without_improvement = 0
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_metrics = {
                "best_epoch": epoch,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
            }
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= train_config.early_stopping_patience:
            stopped_early = True
            stop_epoch = epoch
            print(
                "触发 early stopping，"
                f"epoch={epoch} "
                f"patience={train_config.early_stopping_patience} "
                f"min_delta={train_config.early_stopping_min_delta}"
            )
            break

        if scheduler is not None:
            scheduler.step()

    if best_state_dict is None:
        raise RuntimeError("训练未产生有效模型参数")

    model.load_state_dict(best_state_dict)
    best_metrics.update(
        {
            "elapsed_seconds": time.time() - start_time,
            "stopped_early": stopped_early,
            "stop_epoch": stop_epoch,
        }
    )

    curve_paths = plot_training_history(
        history=history,
        output_dir=train_config.output_dir,
        best_epoch=int(best_metrics["best_epoch"]),
    )

    checkpoint_path = train_config.output_dir / "best_model.pt"
    save_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        train_dataset=train_dataset,
        experiment_config=experiment_config,
        metrics=best_metrics,
    )
    save_json_summary(
        train_config.output_dir / "training_summary.json",
        {
            "best_metrics": best_metrics,
            "history": history,
            "config": experiment_config.to_dict(),
            "runtime": {
                "device": device_name,
                "loss": "CrossEntropyLoss",
                "optimizer": train_config.optimizer,
                "scheduler": train_config.scheduler,
            },
            "early_stopping": {
                "enabled": True,
                "patience": train_config.early_stopping_patience,
                "min_delta": train_config.early_stopping_min_delta,
                "stopped_early": stopped_early,
                "stop_epoch": stop_epoch,
            },
            "artifacts": {
                "checkpoint": str(checkpoint_path),
                "curves": curve_paths,
            },
        },
    )

    print(
        "训练完成，"
        f"best_epoch={best_metrics['best_epoch']} "
        f"val_acc={best_metrics['val_accuracy']:.4f} "
        f"val_f1={best_metrics['val_macro_f1']:.4f} "
        f"device={device_name}"
    )
    return {
        "best_metrics": best_metrics,
        "history": history,
        "checkpoint_path": checkpoint_path,
        "device": device_name,
    }


def main(config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    experiment_config = load_experiment_config(config_path=config_path)
    run_training(experiment_config)


if __name__ == "__main__":
    main()
