"""KMeans 聚类配置读取与路径解析。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_COLUMNS = [
    # 与 data/classification/preprocess_classification.py 输出保持一致的 16 维特征。
    # 这些列共同描述 7 天窗口的总量、峰谷结构、作息差异、趋势和波动。
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
class KMeansPaths:
    """KMeans 聚类训练使用的输入输出路径。"""

    features_path: Path
    output_dir: Path
    cluster_result_path: Path
    cluster_centers_path: Path
    cluster_centers_standardized_path: Path
    cluster_summary_path: Path
    scaler_path: Path
    model_path: Path


def resolve_path(path_value: str | Path) -> Path:
    """将配置路径解析为 models 工程根目录下的绝对路径。"""

    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(config_path: Path) -> dict[str, Any]:
    """读取 KMeans 聚类 YAML 配置。"""

    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_paths(config: dict[str, Any]) -> KMeansPaths:
    """把配置中的目录和固定输出文件名转换为路径对象。"""

    output_dir = resolve_path(config["output"]["output_dir"])
    return KMeansPaths(
        features_path=resolve_path(config["data"]["features_path"]),
        output_dir=output_dir,
        cluster_result_path=output_dir / "cluster_result.csv",
        cluster_centers_path=output_dir / "cluster_centers.csv",
        cluster_centers_standardized_path=output_dir / "cluster_centers_standardized.csv",
        cluster_summary_path=output_dir / "cluster_summary.json",
        scaler_path=output_dir / "scaler.pkl",
        model_path=output_dir / "kmeans_model.pkl",
    )
