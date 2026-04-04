"""LSTM 预测训练入口。"""

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

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from forecast.LSTM.config import DEFAULT_CONFIG_PATH, detect_device, load_experiment_config
from forecast.LSTM.engine import (
    build_model,
    build_criterion,
    count_parameters,
    create_data_loaders,
    create_split_datasets,
    describe_loss,
    evaluate,
    save_checkpoint,
    save_json_summary,
    set_seed,
    train_one_epoch,
)


def plot_training_history(
    history: list[dict[str, float]],
    output_dir: Path,
    experiment_config,
    best_metrics: dict[str, float],
    device_name: str,
    parameter_count: int,
) -> dict[str, str]:
    if not history:
        raise ValueError("history 为空，无法绘制训练结果")

    epochs = [int(item["epoch"]) for item in history]
    train_loss = [float(item["train_loss"]) for item in history]
    val_loss = [float(item["val_loss"]) for item in history]
    train_rmse = [float(item["train_rmse"]) for item in history]
    val_rmse = [float(item["val_rmse"]) for item in history]
    train_mae = [float(item["train_mae"]) for item in history]
    val_mae = [float(item["val_mae"]) for item in history]
    train_smape = [float(item["train_smape"]) for item in history]
    val_smape = [float(item["val_smape"]) for item in history]
    train_wape = [float(item["train_wape"]) for item in history]
    val_wape = [float(item["val_wape"]) for item in history]

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "result.png"

    figure, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    axes[0].axis("off")
    text_lines = [
        "LSTM Experiment Config",
        f"data_path: {experiment_config.data.data_path}",
        f"split_mode: {experiment_config.data.split_mode}",
        f"train_ratio: {experiment_config.data.train_ratio}",
        f"val_ratio: {experiment_config.data.val_ratio}",
        f"feature_names: {experiment_config.data.feature_names}",
        f"aggregate_normalization: {experiment_config.data.aggregate_normalization}",
        f"hidden_size: {experiment_config.model.hidden_size}",
        f"num_layers: {experiment_config.model.num_layers}",
        f"dropout: {experiment_config.model.dropout}",
        f"teacher_forcing_ratio: {experiment_config.model.teacher_forcing_ratio}",
        f"batch_size: {experiment_config.train.batch_size}",
        f"epochs: {experiment_config.train.epochs}",
        f"learning_rate: {experiment_config.train.learning_rate}",
        f"weight_decay: {experiment_config.train.weight_decay}",
        f"gradient_clip_norm: {experiment_config.train.gradient_clip_norm}",
        f"loss: {describe_loss(experiment_config.train)}",
        f"seed: {experiment_config.train.seed}",
        f"device: {device_name}",
        f"parameter_count: {parameter_count}",
        "",
        "Best Metrics",
        f"best_epoch: {best_metrics['best_epoch']}",
        f"val_loss: {best_metrics['val_loss']:.4f}",
        f"val_mae: {best_metrics['val_mae']:.4f}",
        f"val_rmse: {best_metrics['val_rmse']:.4f}",
        f"val_smape: {best_metrics['val_smape']:.4f}",
        f"val_wape: {best_metrics['val_wape']:.4f}",
        f"elapsed_seconds: {best_metrics['elapsed_seconds']:.2f}",
    ]
    axes[0].text(
        0.01,
        0.99,
        "\n".join(text_lines),
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
    )

    axes[1].plot(epochs, train_loss, label="train_loss", linewidth=2)
    axes[1].plot(epochs, val_loss, label="val_loss", linewidth=2)
    axes[1].set_title("Loss")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend()

    axes[2].plot(epochs, train_rmse, label="train_rmse", linewidth=2)
    axes[2].plot(epochs, val_rmse, label="val_rmse", linewidth=2)
    axes[2].plot(epochs, train_mae, label="train_mae", linewidth=1.5)
    axes[2].plot(epochs, val_mae, label="val_mae", linewidth=1.5)
    axes[2].set_title("RMSE / MAE")
    axes[2].grid(True, linestyle="--", alpha=0.4)
    axes[2].legend()

    axes[3].plot(epochs, train_smape, label="train_smape", linewidth=2)
    axes[3].plot(epochs, val_smape, label="val_smape", linewidth=2)
    axes[3].plot(epochs, train_wape, label="train_wape", linewidth=1.5)
    axes[3].plot(epochs, val_wape, label="val_wape", linewidth=1.5)
    axes[3].set_title("sMAPE / WAPE")
    axes[3].grid(True, linestyle="--", alpha=0.4)
    axes[3].legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return {"result": str(output_path)}


def run_training(experiment_config) -> dict[str, object]:
    train_config = experiment_config.train
    model_config = experiment_config.model
    set_seed(train_config.seed)

    device_name = detect_device()
    device = torch.device(device_name)

    train_dataset, val_dataset, test_dataset = create_split_datasets(
        data_config=experiment_config.data,
        seed=train_config.seed,
    )
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=train_config.batch_size,
    )

    model = build_model(model_config).to(device)
    criterion = build_criterion(train_config)
    loss_name = describe_loss(train_config)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    best_val_rmse = float("inf")
    best_metrics: dict[str, float] = {}
    best_state_dict = None
    history: list[dict[str, float]] = []
    epochs_without_improvement = 0
    stopped_early = False
    stop_epoch: int | None = None
    start_time = time.time()

    for epoch in range(1, train_config.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            teacher_forcing_ratio=model_config.teacher_forcing_ratio,
            gradient_clip_norm=train_config.gradient_clip_norm,
        )
        val_metrics = evaluate(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device,
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_mae": train_metrics["mae"],
            "train_rmse": train_metrics["rmse"],
            "train_smape": train_metrics["smape"],
            "train_wape": train_metrics["wape"],
            "val_loss": val_metrics["loss"],
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "val_smape": val_metrics["smape"],
            "val_wape": val_metrics["wape"],
        }
        history.append(epoch_record)

        print(
            f"epoch={epoch:02d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_mae={train_metrics['mae']:.4f} "
            f"train_rmse={train_metrics['rmse']:.4f} "
            f"train_smape={train_metrics['smape']:.4f} "
            f"train_wape={train_metrics['wape']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_mae={val_metrics['mae']:.4f} "
            f"val_rmse={val_metrics['rmse']:.4f} "
            f"val_smape={val_metrics['smape']:.4f} "
            f"val_wape={val_metrics['wape']:.4f}"
        )

        if val_metrics["rmse"] < best_val_rmse - train_config.early_stopping_min_delta:
            best_val_rmse = val_metrics["rmse"]
            epochs_without_improvement = 0
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_metrics = {
                "best_epoch": epoch,
                "val_loss": val_metrics["loss"],
                "val_mae": val_metrics["mae"],
                "val_rmse": val_metrics["rmse"],
                "val_smape": val_metrics["smape"],
                "val_wape": val_metrics["wape"],
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

    if best_state_dict is None:
        raise RuntimeError("训练未产生有效模型参数")

    model.load_state_dict(best_state_dict)
    test_metrics = evaluate(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
    )
    best_metrics.update(
        {
            "test_loss": test_metrics["loss"],
            "test_mae": test_metrics["mae"],
            "test_rmse": test_metrics["rmse"],
            "test_smape": test_metrics["smape"],
            "test_wape": test_metrics["wape"],
            "elapsed_seconds": time.time() - start_time,
            "stopped_early": stopped_early,
            "stop_epoch": stop_epoch,
        }
    )

    checkpoint_path = train_config.output_dir / "best_model.pt"
    save_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        train_dataset=train_dataset,
        experiment_config=experiment_config,
        metrics=best_metrics,
    )

    curve_paths = plot_training_history(
        history=history,
        output_dir=train_config.output_dir,
        experiment_config=experiment_config,
        best_metrics=best_metrics,
        device_name=device_name,
        parameter_count=count_parameters(model),
    )

    save_json_summary(
        train_config.output_dir / "training_summary.json",
        {
            "best_metrics": best_metrics,
            "history": history,
            "config": experiment_config.to_dict(),
            "runtime": {
                "device": device_name,
                "loss": loss_name,
                "parameter_count": count_parameters(model),
                "aggregate_normalization": experiment_config.data.aggregate_normalization,
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
        f"val_rmse={best_metrics['val_rmse']:.4f} "
        f"test_rmse={best_metrics['test_rmse']:.4f} "
        f"loss={loss_name} "
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
