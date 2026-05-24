"""Isolation Forest 异常检测配置读取与路径解析。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_COLUMNS = [
    # 与分类任务共用的 7 天窗口特征。
    # 统一特征口径后，异常结果可以和行为分类结果一起解释。
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
class IsolationForestPaths:
    """训练流程中使用的输入输出路径。

    该数据类把 YAML 中的相对路径统一解析为绝对路径，避免训练、评估、
    保存模型等阶段重复拼接路径。
    """

    features_path: Path
    output_dir: Path
    anomaly_scores_path: Path
    anomaly_samples_path: Path
    anomaly_summary_path: Path
    model_path: Path


def resolve_path(path_value: str | Path) -> Path:
    """将配置中的路径解析为模型工程根目录下的绝对路径。"""

    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(config_path: Path) -> dict[str, Any]:
    """读取 YAML 配置文件。

    配置文件负责声明数据输入、模型超参数和输出文件名，是整个
    Isolation Forest 训练流程的唯一外部参数入口。
    """

    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_paths(config: dict[str, Any]) -> IsolationForestPaths:
    """根据配置构造训练流程需要的全部路径。"""

    output_dir = resolve_path(config["output"]["output_dir"])
    return IsolationForestPaths(
        features_path=resolve_path(config["data"]["features_path"]),
        output_dir=output_dir,
        anomaly_scores_path=output_dir / config["output"]["anomaly_scores_file"],
        anomaly_samples_path=output_dir / config["output"]["anomaly_samples_file"],
        anomaly_summary_path=output_dir / config["output"]["anomaly_summary_file"],
        model_path=output_dir / config["output"]["model_file"],
    )
