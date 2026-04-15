"""KMeans 聚类结果人工映射打标入口。"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import yaml
from tqdm.auto import tqdm

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from data.process.classification.config import DEFAULT_CONFIG_PATH, load_experiment_config
from data.process.classification.dataset import load_day_feature_frame
from data.process.classification.engine import save_json_summary


def _load_mapping(mapping_path: Path) -> dict[int, str]:
    raw_payload = yaml.safe_load(mapping_path.read_text(encoding="utf-8"))
    if not isinstance(raw_payload, dict) or "cluster_to_label" not in raw_payload:
        raise ValueError("映射文件格式错误：需要包含 cluster_to_label 段")
    raw_mapping = raw_payload["cluster_to_label"]
    if not isinstance(raw_mapping, dict):
        raise ValueError("映射文件格式错误：cluster_to_label 必须是键值映射")

    mapping: dict[int, str] = {}
    for cluster_key, label_name in raw_mapping.items():
        mapping[int(cluster_key)] = str(label_name).strip()
    return mapping


def run_relabel(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, object]:
    experiment_config = load_experiment_config(config_path=config_path)
    output_dir = experiment_config.output.output_dir
    assignments_path = output_dir / "cluster_assignments.csv"
    mapping_path = experiment_config.relabel.mapping_path
    canonical_labeled_path = experiment_config.relabel.canonical_labeled_path
    log = tqdm.write

    log("[阶段] 开始根据人工映射重新打标签")
    if not assignments_path.exists():
        raise FileNotFoundError(
            f"未找到聚类分配文件：{assignments_path}，请先运行 fit 模式。"
        )
    if not mapping_path.exists():
        raise FileNotFoundError(
            f"未找到映射文件：{mapping_path}，请先编辑 fit 生成的映射模板。"
        )

    source_frame = load_day_feature_frame(experiment_config.data.data_path)
    assignments_frame = pd.read_csv(assignments_path)
    mapping = _load_mapping(mapping_path)
    observed_clusters = sorted(assignments_frame["cluster_id"].astype(int).unique().tolist())
    missing_clusters = [cluster_id for cluster_id in observed_clusters if cluster_id not in mapping]
    if missing_clusters:
        raise ValueError(f"映射文件缺少这些 cluster_id: {missing_clusters}")

    merged = source_frame.merge(
        assignments_frame[["sample_id", "cluster_id"]],
        on="sample_id",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(source_frame):
        raise ValueError("聚类分配与原始样本无法一一对应，请重新运行 fit。")

    if "label_name" in merged.columns:
        merged = merged.rename(columns={"label_name": "original_label_name"})
    merged["cluster_id"] = merged["cluster_id"].astype(int)
    merged["label_name"] = merged["cluster_id"].map(mapping)
    if merged["label_name"].isna().any():
        raise ValueError("存在 cluster_id 未映射到标签，请检查 mapping 文件。")

    labeled_features_path = output_dir / experiment_config.relabel.labeled_features_filename
    labeled_labels_path = output_dir / experiment_config.relabel.labeled_labels_filename
    labels_only = merged.loc[:, ["sample_id", "house_id", "date", "cluster_id", "label_name"]].copy()

    canonical_labeled_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(labeled_features_path, index=False)
    labels_only.to_csv(labeled_labels_path, index=False)
    merged.to_csv(canonical_labeled_path, index=False)
    save_json_summary(
        output_dir / "relabel_summary.json",
        {
            "mapping_path": str(mapping_path),
            "labeled_features_path": str(labeled_features_path),
            "labeled_labels_path": str(labeled_labels_path),
            "canonical_labeled_path": str(canonical_labeled_path),
            "label_distribution": {
                str(label_name): int(count)
                for label_name, count in labels_only["label_name"].value_counts().sort_index().items()
            },
            "cluster_to_label": {str(cluster_id): label_name for cluster_id, label_name in mapping.items()},
            "config": experiment_config.to_dict(),
        },
    )
    log(f"[产物] labeled_features={labeled_features_path}")
    log(f"[产物] labeled_labels={labeled_labels_path}")
    log(f"[产物] canonical_labeled={canonical_labeled_path}")
    return {
        "labeled_features_path": str(labeled_features_path),
        "labeled_labels_path": str(labeled_labels_path),
        "canonical_labeled_path": str(canonical_labeled_path),
    }


def main(config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    result = run_relabel(config_path=config_path)
    tqdm.write(
        "重新打标完成，"
        f"features={result['labeled_features_path']} "
        f"labels={result['labeled_labels_path']} "
        f"canonical={result['canonical_labeled_path']}"
    )


if __name__ == "__main__":
    main()
