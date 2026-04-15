"""Patch encoder-decoder direct Transformer 预测训练入口。"""

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
from tqdm.auto import tqdm

if __package__ is None or __package__ == "":
    current_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(current_dir))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from config import DEFAULT_CONFIG_PATH, detect_device, load_experiment_config
    from engine import (
        build_criterion,
        build_model,
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
else:
    from .config import DEFAULT_CONFIG_PATH, detect_device, load_experiment_config
    from .engine import (
        build_criterion,
        build_model,
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
        "Patch Encoder-Decoder Direct Transformer Config",
        f"data_path: {experiment_config.data.data_path}",
        f"split_mode: {experiment_config.data.split_mode}",
        f"train_ratio: {experiment_config.data.train_ratio}",
        f"val_ratio: {experiment_config.data.val_ratio}",
        f"feature_names: {experiment_config.data.feature_names}",
        f"aggregate_normalization: {experiment_config.data.aggregate_normalization}",
        f"d_model: {experiment_config.model.d_model}",
        f"num_layers: {experiment_config.model.num_layers}",
        f"num_heads: {experiment_config.model.num_heads}",
        f"ffn_dim: {experiment_config.model.ffn_dim}",
        f"dropout: {experiment_config.model.dropout}",
        f"patch_length: {experiment_config.model.patch_length}",
        f"patch_stride: {experiment_config.model.patch_stride}",
        f"batch_size: {experiment_config.train.batch_size}",
        f"epochs: {experiment_config.train.epochs}",
        f"learning_rate: {experiment_config.train.learning_rate}",
        f"scheduler_mode: {experiment_config.train.scheduler_mode}",
        f"scheduler_factor: {experiment_config.train.scheduler_factor}",
        f"scheduler_patience: {experiment_config.train.scheduler_patience}",
        f"scheduler_min_lr: {experiment_config.train.scheduler_min_lr}",
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
    log = tqdm.write

    device_name = detect_device()
    device = torch.device(device_name)

    log("[阶段] 开始准备 Transformer 训练")
    log(
        f"[配置] data_path={experiment_config.data.data_path} "
        f"split_mode={experiment_config.data.split_mode} "
        f"batch_size={train_config.batch_size} "
        f"device={device_name}"
    )

    log("[阶段] 加载并切分预测样本")
    train_dataset, val_dataset, test_dataset = create_split_datasets(
        data_config=experiment_config.data,
        seed=train_config.seed,
    )
    log(
        f"[数据] train={len(train_dataset)} val={len(val_dataset)} test={len(test_dataset)}"
    )

    log("[阶段] 构建 DataLoader")
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=train_config.batch_size,
    )
    log(
        f"[DataLoader] train_batches={len(train_loader)} "
        f"val_batches={len(val_loader)} test_batches={len(test_loader)}"
    )

    log("[阶段] 初始化模型、损失函数与优化器")
    model = build_model(model_config).to(device)
    parameter_count = count_parameters(model)
    criterion = build_criterion(train_config)
    loss_name = describe_loss(train_config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    scheduler = None
    if train_config.scheduler_mode == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=train_config.scheduler_factor,
            patience=train_config.scheduler_patience,
            min_lr=train_config.scheduler_min_lr,
        )
    log(
        f"[模型] parameter_count={parameter_count} loss={loss_name} "
        f"learning_rate={train_config.learning_rate}"
    )

    best_val_rmse = float("inf")
    best_metrics: dict[str, float] = {}
    best_state_dict = None
    history: list[dict[str, float]] = []
    epochs_without_improvement = 0
    stopped_early = False
    stop_epoch: int | None = None
    start_time = time.time()

    epoch_progress = tqdm(
        range(1, train_config.epochs + 1),
        desc="训练 Epoch",
        dynamic_ncols=True,
        mininterval=1.0,
    )
    for epoch in epoch_progress:
        train_metrics = train_one_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            gradient_clip_norm=train_config.gradient_clip_norm,
            stage_label=f"Epoch {epoch}/{train_config.epochs} 训练",
        )
        val_metrics = evaluate(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device,
            stage_label=f"Epoch {epoch}/{train_config.epochs} 验证",
        )

        epoch_record = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
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

        if scheduler is not None:
            scheduler.step(val_metrics["rmse"])

        epoch_progress.set_postfix(
            lr=f"{epoch_record['learning_rate']:.6f}",
            val_rmse=f"{val_metrics['rmse']:.2f}",
            best_rmse=f"{min(best_val_rmse, val_metrics['rmse']):.2f}",
        )
        log(
            f"epoch={epoch:02d} "
            f"lr={epoch_record['learning_rate']:.6f} "
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
            log(
                "触发 early stopping，"
                f"epoch={epoch} "
                f"patience={train_config.early_stopping_patience} "
                f"min_delta={train_config.early_stopping_min_delta}"
            )
            break

    if best_state_dict is None:
        raise RuntimeError("训练未产生有效模型参数")

    model.load_state_dict(best_state_dict)
    log("[阶段] 使用最佳权重进行测试评估")
    test_metrics = evaluate(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
        stage_label="测试",
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
            "stop_epoch": stop_epoch or train_config.epochs,
        }
    )

    checkpoint_path = train_config.output_dir / "best_model.pt"
    parameter_count = count_parameters(model)
    save_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        train_dataset=train_dataset,
        experiment_config=experiment_config,
        metrics=best_metrics,
    )
    artifacts = plot_training_history(
        history=history,
        output_dir=train_config.output_dir,
        experiment_config=experiment_config,
        best_metrics=best_metrics,
        device_name=device_name,
        parameter_count=parameter_count,
    )
    summary = {
        "best_metrics": best_metrics,
        "history": history,
        "config": experiment_config.to_dict(),
        "runtime": {
            "device": device_name,
            "loss": loss_name,
            "parameter_count": parameter_count,
            "aggregate_normalization": train_dataset.normalization.aggregate_mode,
            "scheduler_mode": train_config.scheduler_mode,
        },
        "early_stopping": {
            "enabled": True,
            "patience": train_config.early_stopping_patience,
            "min_delta": train_config.early_stopping_min_delta,
            "stopped_early": stopped_early,
            "stop_epoch": stop_epoch or train_config.epochs,
        },
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "curves": artifacts,
        },
    }
    save_json_summary(train_config.output_dir / "training_summary.json", summary)

    log(
        "训练完成，"
        f"best_epoch={best_metrics['best_epoch']} "
        f"val_rmse={best_metrics['val_rmse']:.4f} "
        f"test_rmse={best_metrics['test_rmse']:.4f} "
        f"loss={loss_name} "
        f"device={device_name}"
    )
    return summary


def main(config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    experiment_config = load_experiment_config(config_path=config_path)
    run_training(experiment_config)


if __name__ == "__main__":
    main()
