"""XGBoost 分类模型配置读取与路径解析。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_COLUMNS = [
    # 与 KMeans 聚类阶段保持完全一致的 16 维行为特征。
    # XGBoost 学习的是“特征 -> 聚类标签”的映射。
    "avg_energy",
    "std_energy",
    "max_energy",
    "min_energy",
    "avg_peak",
    "avg_valley",
    "peak_valley_ratio",
    "peak_ratio",
    "valley_ratio",
    "load_factor",
    "workday_avg",
    "weekend_avg",
    "weekend_workday_ratio",
    "trend_rel",
    "volatility",
    "med_mean_ratio",
]


@dataclass(frozen=True)
class XGBoostClassifierPaths:
    """XGBoost 分类训练使用的输入输出路径。"""

    features_path: Path
    labels_path: Path
    output_dir: Path
    model_path: Path
    label_encoder_path: Path
    report_path: Path


def resolve_path(path_value: str | Path) -> Path:
    """将配置路径解析为 models 工程根目录下的绝对路径。"""

    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(config_path: Path) -> dict[str, Any]:
    """读取 XGBoost 分类 YAML 配置。"""

    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_paths(config: dict[str, Any]) -> XGBoostClassifierPaths:
    """根据配置构造输入输出路径。"""

    output_dir = resolve_path(config["output"]["output_dir"])
    return XGBoostClassifierPaths(
        features_path=resolve_path(config["data"]["features_path"]),
        labels_path=resolve_path(config["data"]["labels_path"]),
        output_dir=output_dir,
        model_path=output_dir / "xgboost_model.json",
        label_encoder_path=output_dir / "label_encoder.pkl",
        report_path=output_dir / "classifier_report.json",
    )
