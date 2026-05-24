"""XGBoost 预测模型配置读取与路径解析。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ForecastXGBoostPaths:
    """预测 XGBoost 训练流程使用的输入输出路径。"""

    dataset_path: Path
    output_dir: Path
    model_dir: Path
    checkpoint_dir: Path
    metrics_path: Path
    feature_path: Path
    baseline_path: Path


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(config_path: Path) -> dict[str, Any]:
    """读取 XGBoost 预测任务 YAML 配置。"""

    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_paths(config: dict[str, Any]) -> ForecastXGBoostPaths:
    """把配置中的输入输出位置转换为路径对象。"""

    output_dir = resolve_path(config["output"]["output_dir"])
    return ForecastXGBoostPaths(
        dataset_path=resolve_path(config["data"]["dataset_path"]),
        output_dir=output_dir,
        model_dir=output_dir / config["output"]["model_dir"],
        checkpoint_dir=output_dir / config["output"]["checkpoint_dir"],
        metrics_path=output_dir / config["output"]["metrics_file"],
        feature_path=output_dir / config["output"]["feature_file"],
        baseline_path=output_dir / "peak_valley_baseline.json",
    )
