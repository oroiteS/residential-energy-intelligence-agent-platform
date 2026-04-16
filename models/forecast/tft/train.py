"""TFT 训练入口。"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import warnings

if __package__ is None or __package__ == "":
    script_dir = str(Path(__file__).resolve().parent)
    project_root = str(Path(__file__).resolve().parents[2])
    sys.path[:] = [path for path in sys.path if path != script_dir]
    sys.path.insert(0, project_root)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ["TRITON_PRINT_AUTOTUNING"] = "0"

try:
    import lightning as L
    from lightning.pytorch.callbacks import (
        EarlyStopping,
        LearningRateMonitor,
        ModelCheckpoint,
        TQDMProgressBar,
    )
    from lightning.pytorch.loggers import CSVLogger
except ModuleNotFoundError:  # pragma: no cover - 兼容旧包名
    import pytorch_lightning as L  # type: ignore[no-redef]
    from pytorch_lightning.callbacks import (
        EarlyStopping,
        LearningRateMonitor,
        ModelCheckpoint,
        TQDMProgressBar,
    )
    from pytorch_lightning.loggers import CSVLogger

from forecast.tft.callbacks import CompactTrainingCallback
from forecast.tft.config import DEFAULT_CONFIG_PATH, load_experiment_config
from forecast.tft.datamodule import ForecastTftDataModule
from forecast.tft.module import TftForecastModule
from forecast.tft.runtime import (
    collect_runtime_summary,
    maybe_compile_model,
    resolve_runtime_settings,
)
from forecast.tft.visualization import (
    plot_parameter_distributions,
    plot_training_history,
    summarize_model_parameters,
)


def run_train(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, object]:
    experiment_config = load_experiment_config(config_path=config_path)
    warnings.filterwarnings(
        "ignore",
        message=".*does not have many workers.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=".*LeafSpec.*deprecated.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=".*pin_memory.*no accelerator is found.*",
    )
    L.seed_everything(experiment_config.train.seed, workers=True)

    datamodule = ForecastTftDataModule(
        config=experiment_config.data,
        seed=experiment_config.train.seed,
    )
    datamodule.setup()
    if datamodule.feature_spec is None:
        raise RuntimeError("DataModule 未能加载 feature_spec")

    output_dir = experiment_config.train.output_dir
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_settings = resolve_runtime_settings(experiment_config.train)
    runtime_summary = collect_runtime_summary(
        data_config=experiment_config.data,
        train_config=experiment_config.train,
        runtime_settings=runtime_settings,
    )
    runtime_summary_path = output_dir / "runtime_summary.json"
    runtime_summary_path.write_text(
        json.dumps(runtime_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    model = TftForecastModule(
        data_config=experiment_config.to_dict()["data"],
        model_config=experiment_config.to_dict()["model"],
        loss_config=experiment_config.to_dict()["loss"],
        feature_spec={
            "feature_dim": datamodule.feature_spec.feature_dim,
            "target_length": datamodule.feature_spec.target_length,
            "profile_dim": len(datamodule.feature_spec.profile_feature_indices),
            "baseline_stat_dim": datamodule.feature_spec.baseline_stat_dim,
        },
        learning_rate=experiment_config.train.learning_rate,
        weight_decay=experiment_config.train.weight_decay,
        test_output_dir=str(experiment_config.test.output_dir),
    )
    model.model = maybe_compile_model(model.model, runtime_settings)
    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="best",
            monitor="val_mae",
            mode="min",
            save_top_k=1,
            save_last=True,
            auto_insert_metric_name=False,
        ),
        EarlyStopping(
            monitor="val_mae",
            mode="min",
            patience=experiment_config.train.early_stopping_patience,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        CompactTrainingCallback(
            runtime_summary=runtime_summary,
            split_summary=datamodule.split_summary,
            output_dir=output_dir,
        ),
    ]
    if experiment_config.train.enable_progress_bar:
        callbacks.append(
            TQDMProgressBar(refresh_rate=experiment_config.train.progress_bar_refresh_rate)
        )
    logger = CSVLogger(
        save_dir=str(output_dir / "logs"),
        name="tft",
    )
    trainer = L.Trainer(
        default_root_dir=str(output_dir),
        accelerator=experiment_config.train.accelerator,
        devices=experiment_config.train.devices,
        max_epochs=experiment_config.train.max_epochs,
        gradient_clip_val=experiment_config.train.gradient_clip_val,
        precision=runtime_settings["precision"],
        deterministic=experiment_config.train.deterministic,
        benchmark=runtime_settings["benchmark"],
        log_every_n_steps=experiment_config.train.log_every_n_steps,
        enable_progress_bar=experiment_config.train.enable_progress_bar,
        enable_model_summary=experiment_config.train.enable_model_summary,
        num_sanity_val_steps=experiment_config.train.num_sanity_val_steps,
        logger=logger,
        callbacks=callbacks,
    )
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule, ckpt_path="best")

    metrics_csv_path = Path(logger.log_dir) / "metrics.csv"
    history_paths = plot_training_history(
        metrics_csv_path=metrics_csv_path,
        output_dir=output_dir,
    )
    parameter_summary = summarize_model_parameters(
        model=model,
        output_dir=output_dir,
    )
    parameter_plot_paths = plot_parameter_distributions(
        model=model,
        output_dir=output_dir,
    )
    config_snapshot_path = output_dir / "config_snapshot.json"
    config_snapshot_path.write_text(
        json.dumps(experiment_config.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "config": experiment_config.to_dict(),
        "split_summary": datamodule.split_summary,
        "best_model_path": callbacks[0].best_model_path,
        "last_model_path": str(checkpoint_dir / "last.ckpt"),
        "metrics_csv_path": str(metrics_csv_path),
        "best_score": (
            float(callbacks[0].best_model_score.item())
            if callbacks[0].best_model_score is not None
            else None
        ),
        "history_plot_paths": [str(path) for path in history_paths],
        "parameter_plot_paths": [str(path) for path in parameter_plot_paths],
        "parameter_summary_path": str(output_dir / "parameter_summary.json"),
        "config_snapshot_path": str(config_snapshot_path),
        "runtime_summary_path": str(runtime_summary_path),
        "test_output_dir": str(experiment_config.test.output_dir),
        "trainable_params": parameter_summary["trainable_params"],
        "total_params": parameter_summary["total_params"],
    }
    summary_path = output_dir / "training_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main(config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    summary = run_train(config_path=config_path)
    print(
        "TFT 训练完成，"
        f"best_model_path={summary['best_model_path']}，"
        f"best_score={summary['best_score']}，"
        f"output_dir={summary['config']['train']['output_dir']}"
    )
    print(
        "训练产物："
        f"metrics_csv={summary['metrics_csv_path']}，"
        f"history_plots={len(summary['history_plot_paths'])}，"
        f"parameter_plots={len(summary['parameter_plot_paths'])}，"
        f"test_output_dir={summary['test_output_dir']}"
    )


if __name__ == "__main__":
    main()
