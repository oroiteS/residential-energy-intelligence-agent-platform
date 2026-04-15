"""KMeans 聚类分析配置加载。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from common.config_validation import validate_config_schema


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "config.yaml"
TOP_LEVEL_KEYS = {"data", "clustering", "output", "relabel"}
SECTION_KEYS = {
    "data": {"data_path"},
    "clustering": {
        "n_clusters",
        "normalization_mode",
        "random_seed",
        "n_init",
        "max_iter",
        "tol",
    },
    "output": {"output_dir", "representatives_per_cluster"},
    "relabel": {
        "mapping_path",
        "labeled_features_filename",
        "labeled_labels_filename",
        "canonical_labeled_path",
    },
}


@dataclass(slots=True)
class DataConfig:
    data_path: Path


@dataclass(slots=True)
class ClusteringConfig:
    n_clusters: int
    normalization_mode: str
    random_seed: int
    n_init: int
    max_iter: int
    tol: float


@dataclass(slots=True)
class OutputConfig:
    output_dir: Path
    representatives_per_cluster: int


@dataclass(slots=True)
class RelabelConfig:
    mapping_path: Path
    labeled_features_filename: str
    labeled_labels_filename: str
    canonical_labeled_path: Path


@dataclass(slots=True)
class ExperimentConfig:
    data: DataConfig
    clustering: ClusteringConfig
    output: OutputConfig
    relabel: RelabelConfig

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["data"]["data_path"] = str(self.data.data_path)
        payload["output"]["output_dir"] = str(self.output.output_dir)
        payload["relabel"]["mapping_path"] = str(self.relabel.mapping_path)
        payload["relabel"]["canonical_labeled_path"] = str(
            self.relabel.canonical_labeled_path
        )
        return payload


def _resolve_path(path_value: str | None, base_dir: Path) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def load_experiment_config(config_path: Path = DEFAULT_CONFIG_PATH) -> ExperimentConfig:
    config_path = config_path.resolve()
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base_dir = Path(__file__).resolve().parents[3]
    sections = validate_config_schema(
        raw_config,
        config_path=config_path,
        allowed_top_level_keys=TOP_LEVEL_KEYS,
        allowed_section_keys=SECTION_KEYS,
    )
    data_raw = sections["data"]
    clustering_raw = sections["clustering"]
    output_raw = sections["output"]
    relabel_raw = sections["relabel"]

    data_config = DataConfig(
        data_path=_resolve_path(data_raw["data_path"], base_dir),
    )
    clustering_config = ClusteringConfig(
        n_clusters=int(clustering_raw.get("n_clusters", 4)),
        normalization_mode=str(clustering_raw.get("normalization_mode", "sample_zscore")),
        random_seed=int(clustering_raw.get("random_seed", 42)),
        n_init=int(clustering_raw.get("n_init", 5)),
        max_iter=int(clustering_raw.get("max_iter", 100)),
        tol=float(clustering_raw.get("tol", 1e-4)),
    )
    output_config = OutputConfig(
        output_dir=_resolve_path(output_raw["output_dir"], base_dir),
        representatives_per_cluster=int(output_raw.get("representatives_per_cluster", 5)),
    )
    relabel_config = RelabelConfig(
        mapping_path=_resolve_path(relabel_raw["mapping_path"], base_dir),
        labeled_features_filename=str(
            relabel_raw.get("labeled_features_filename", "cluster_labeled_day_features.csv")
        ),
        labeled_labels_filename=str(
            relabel_raw.get("labeled_labels_filename", "cluster_labeled_day_labels.csv")
        ),
        canonical_labeled_path=_resolve_path(
            relabel_raw.get(
                "canonical_labeled_path",
                "data/processed/classification/classification_day_labels.csv",
            ),
            base_dir,
        ),
    )

    if clustering_config.n_clusters < 2:
        raise ValueError("配置错误：clustering.n_clusters 必须至少为 2")
    if clustering_config.n_init <= 0:
        raise ValueError("配置错误：clustering.n_init 必须大于 0")
    if clustering_config.max_iter <= 0:
        raise ValueError("配置错误：clustering.max_iter 必须大于 0")
    if clustering_config.normalization_mode not in {"none", "sample_zscore", "sample_minmax", "day_mean"}:
        raise ValueError(
            "配置错误：clustering.normalization_mode 只支持 none / sample_zscore / sample_minmax / day_mean"
        )
    return ExperimentConfig(
        data=data_config,
        clustering=clustering_config,
        output=output_config,
        relabel=relabel_config,
    )
