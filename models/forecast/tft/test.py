"""TFT 测试入口。"""

from __future__ import annotations

import os
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    script_dir = str(Path(__file__).resolve().parent)
    project_root = str(Path(__file__).resolve().parents[2])
    sys.path[:] = [path for path in sys.path if path != script_dir]
    sys.path.insert(0, project_root)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ["TRITON_PRINT_AUTOTUNING"] = "0"

try:
    import lightning as L
except ModuleNotFoundError:  # pragma: no cover - 兼容旧包名
    import pytorch_lightning as L  # type: ignore[no-redef]

from forecast.tft.config import DEFAULT_CONFIG_PATH, load_experiment_config
from forecast.tft.datamodule import ForecastTftDataModule
from forecast.tft.module import TftForecastModule
from forecast.tft.runtime import maybe_compile_model, resolve_runtime_settings


def run_test(config_path: Path = DEFAULT_CONFIG_PATH) -> Path:
    experiment_config = load_experiment_config(config_path=config_path)
    datamodule = ForecastTftDataModule(
        config=experiment_config.data,
        seed=experiment_config.train.seed,
    )
    datamodule.setup()
    if datamodule.feature_spec is None:
        raise RuntimeError("DataModule 未能加载 feature_spec")

    checkpoint_path = experiment_config.test.checkpoint_path or (
        experiment_config.train.output_dir / "checkpoints" / "best.ckpt"
    )
    runtime_settings = resolve_runtime_settings(experiment_config.train)
    model = TftForecastModule.load_from_checkpoint(
        str(checkpoint_path),
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
    trainer = L.Trainer(
        default_root_dir=str(experiment_config.train.output_dir),
        accelerator=experiment_config.train.accelerator,
        devices=experiment_config.train.devices,
        precision=runtime_settings["precision"],
        benchmark=runtime_settings["benchmark"],
        logger=False,
    )
    trainer.test(model=model, datamodule=datamodule)
    return experiment_config.test.output_dir


def main(config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    output_dir = run_test(config_path=config_path)
    print(
        "TFT 测试完成，"
        f"输出目录: {output_dir}，"
        "已写出 sample/household 指标、统计摘要和预测曲线样例"
    )


if __name__ == "__main__":
    main()
