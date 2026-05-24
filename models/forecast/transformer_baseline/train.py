#!/usr/bin/env python3
"""PyTorch Lightning Transformer 训练入口。"""

from __future__ import annotations

import argparse
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

try:
    from .config import load_config, resolve_path
    from .dataloader import TransformerForecastDataModule
    from .model import TransformerResidualForecaster
except ImportError:  # 兼容 `python forecast/transformer/train.py`
    from config import load_config, resolve_path
    from dataloader import TransformerForecastDataModule
    from model import TransformerResidualForecaster


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 Transformer 残差预测模型")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config" / "default.yaml",
        help="训练配置 YAML 路径",
    )
    parser.add_argument("--no-resume", action="store_true", help="忽略 last.ckpt，从头训练")
    return parser.parse_args()


def train(config_path: Path, *, no_resume: bool = False) -> None:
    config = load_config(config_path)
    L.seed_everything(int(config["training"]["random_seed"]), workers=True)

    output_dir = resolve_path(config["output"]["output_dir"])
    checkpoint_dir = output_dir / config["output"]["checkpoint_dir"]
    log_dir = output_dir / config["output"]["log_dir"]
    feature_file = output_dir / config["output"]["feature_file"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    datamodule = TransformerForecastDataModule(config)
    datamodule.setup("fit")
    datamodule.write_feature_file(feature_file)
    target_mean, target_scale = datamodule.target_scaler_state()
    target_mean = [float(value) for value in target_mean]
    target_scale = [float(value) for value in target_scale]

    model = TransformerResidualForecaster(
        sequence_feature_size=datamodule.sequence_feature_size,
        future_feature_size=datamodule.future_feature_size,
        static_feature_size=datamodule.static_feature_size,
        output_size=datamodule.output_size,
        input_days=int(config["features"]["input_days"]),
        model_config=config["model"],
        learning_rate=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
        target_mean=target_mean,
        target_scale=target_scale,
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="best-{epoch:03d}-{valid_rmse:.4f}",
            monitor="valid_rmse",
            mode="min",
            save_top_k=1,
            save_last=True,
        ),
        EarlyStopping(
            monitor="valid_rmse",
            mode="min",
            patience=int(config["training"]["early_stopping_patience"]),
            min_delta=float(config["training"]["min_delta"]),
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    logger = CSVLogger(save_dir=log_dir, name="transformer_baseline_residual")

    trainer = L.Trainer(
        max_epochs=int(config["training"]["max_epochs"]),
        accelerator=config["trainer"]["accelerator"],
        devices=config["trainer"]["devices"],
        precision=config["trainer"]["precision"],
        deterministic=bool(config["trainer"]["deterministic"]),
        gradient_clip_val=float(config["training"]["gradient_clip_val"]),
        log_every_n_steps=int(config["trainer"]["log_every_n_steps"]),
        callbacks=callbacks,
        logger=logger,
    )

    ckpt_path = None
    last_ckpt = checkpoint_dir / "last.ckpt"
    if bool(config["training"]["resume"]) and not no_resume and last_ckpt.exists():
        ckpt_path = str(last_ckpt)

    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)


def main() -> None:
    args = parse_args()
    train(args.config, no_resume=args.no_resume)


if __name__ == "__main__":
    main()
