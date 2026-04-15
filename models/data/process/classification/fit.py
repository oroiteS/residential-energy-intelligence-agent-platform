"""KMeans 聚类分析入口。"""

from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from tqdm.auto import tqdm

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from data.process.classification.config import DEFAULT_CONFIG_PATH, load_experiment_config
from data.process.classification.constants import (
    NORMALIZED_MEAN_COLUMNS,
    RAW_MEAN_COLUMNS,
    SEQUENCE_LENGTH,
)
from data.process.classification.dataset import (
    extract_raw_curves,
    load_day_feature_frame,
    normalize_curves,
)
from data.process.classification.engine import fit_kmeans, save_json_summary


def _build_profiles(
    data_frame: pd.DataFrame,
    raw_curves: np.ndarray,
    normalized_curves: np.ndarray,
    labels: np.ndarray,
    distances: np.ndarray,
    representatives_per_cluster: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    total_samples = len(data_frame)
    profile_rows: list[dict[str, object]] = []
    representative_rows: list[dict[str, object]] = []
    cluster_summaries: list[dict[str, object]] = []

    for cluster_id in sorted(np.unique(labels).tolist()):
        cluster_mask = labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        cluster_raw = raw_curves[cluster_mask]
        cluster_normalized = normalized_curves[cluster_mask]
        cluster_distances = distances[cluster_mask]
        raw_mean = cluster_raw.mean(axis=0)
        normalized_mean = cluster_normalized.mean(axis=0)
        sample_count = int(cluster_mask.sum())
        share = float(sample_count / total_samples)

        profile_row: dict[str, object] = {
            "cluster_id": int(cluster_id),
            "sample_count": sample_count,
            "sample_share": share,
        }
        for index, value in enumerate(raw_mean):
            profile_row[RAW_MEAN_COLUMNS[index]] = float(value)
        for index, value in enumerate(normalized_mean):
            profile_row[NORMALIZED_MEAN_COLUMNS[index]] = float(value)
        profile_rows.append(profile_row)

        order = np.argsort(cluster_distances)[:representatives_per_cluster]
        for rank, local_index in enumerate(order, start=1):
            source_index = int(cluster_indices[local_index])
            representative_rows.append(
                {
                    "cluster_id": int(cluster_id),
                    "rank": rank,
                    "distance_to_centroid": float(cluster_distances[local_index]),
                    "sample_id": str(data_frame.iloc[source_index]["sample_id"]),
                    "house_id": str(data_frame.iloc[source_index]["house_id"]),
                    "date": str(data_frame.iloc[source_index]["date"]),
                }
            )

        cluster_summaries.append(
            {
                "cluster_id": int(cluster_id),
                "sample_count": sample_count,
                "sample_share": share,
                "mean_distance": float(cluster_distances.mean()),
                "median_distance": float(np.median(cluster_distances)),
            }
        )

    return (
        pd.DataFrame(profile_rows),
        pd.DataFrame(representative_rows),
        cluster_summaries,
    )


def _plot_cluster_curves(profile_frame: pd.DataFrame, output_path: Path) -> Path:
    profile_frame = profile_frame.sort_values("cluster_id").reset_index(drop=True)
    x_axis = np.arange(SEQUENCE_LENGTH)
    figure, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    for row in profile_frame.itertuples(index=False):
        raw_curve = np.asarray([getattr(row, column) for column in RAW_MEAN_COLUMNS], dtype=np.float32)
        normalized_curve = np.asarray(
            [getattr(row, column) for column in NORMALIZED_MEAN_COLUMNS],
            dtype=np.float32,
        )
        label = f"cluster_{row.cluster_id} (n={row.sample_count})"
        axes[0].plot(x_axis, raw_curve, linewidth=2, label=label)
        axes[1].plot(x_axis, normalized_curve, linewidth=2, label=label)

    axes[0].set_title("Cluster Mean Daily Load Curves")
    axes[0].set_ylabel("Aggregate")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend()

    axes[1].set_title("Cluster Mean Normalized Curves")
    axes[1].set_xlabel("15min Slot")
    axes[1].set_ylabel("Normalized Aggregate")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend()

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def _write_mapping_template(output_path: Path, cluster_ids: list[int]) -> Path:
    payload = {
        "cluster_to_label": {
            str(cluster_id): f"请替换为簇{cluster_id}的标签名"
            for cluster_id in cluster_ids
        }
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return output_path


def run_fit(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, object]:
    experiment_config = load_experiment_config(config_path=config_path)
    output_dir = experiment_config.output.output_dir
    log = tqdm.write

    log("[阶段] 开始准备 KMeans 聚类")
    data_frame = load_day_feature_frame(experiment_config.data.data_path)
    raw_curves = extract_raw_curves(data_frame)
    normalized_curves = normalize_curves(
        raw_curves,
        normalization_mode=experiment_config.clustering.normalization_mode,
    )
    log(
        f"[数据] samples={len(data_frame)} feature_dim={normalized_curves.shape[1]} "
        f"n_clusters={experiment_config.clustering.n_clusters} "
        f"normalization={experiment_config.clustering.normalization_mode}"
    )

    cluster_result = fit_kmeans(
        data=normalized_curves,
        n_clusters=experiment_config.clustering.n_clusters,
        random_seed=experiment_config.clustering.random_seed,
        n_init=experiment_config.clustering.n_init,
        max_iter=experiment_config.clustering.max_iter,
        tol=experiment_config.clustering.tol,
    )
    labels = cluster_result["labels"]
    centers = cluster_result["centers"]
    distances = cluster_result["distances"]
    log(
        f"[聚类] inertia={cluster_result['inertia']:.4f} "
        f"iterations={cluster_result['iterations']}"
    )

    assignment_frame = data_frame.loc[:, ["sample_id", "house_id", "date"]].copy()
    assignment_frame["cluster_id"] = labels.astype(np.int32)
    assignment_frame["distance_to_centroid"] = distances.astype(np.float32)
    if "label_name" in data_frame.columns:
        assignment_frame["original_label_name"] = data_frame["label_name"].astype(str)

    profile_frame, representative_frame, cluster_summaries = _build_profiles(
        data_frame=data_frame,
        raw_curves=raw_curves,
        normalized_curves=normalized_curves,
        labels=labels,
        distances=distances,
        representatives_per_cluster=experiment_config.output.representatives_per_cluster,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    assignments_path = output_dir / "cluster_assignments.csv"
    profiles_path = output_dir / "cluster_profiles.csv"
    representatives_path = output_dir / "cluster_representatives.csv"
    model_path = output_dir / "cluster_model.npz"
    curves_path = output_dir / "cluster_curves.png"
    summary_path = output_dir / "cluster_summary.json"
    template_path = experiment_config.relabel.mapping_path

    assignment_frame.to_csv(assignments_path, index=False)
    profile_frame.to_csv(profiles_path, index=False)
    representative_frame.to_csv(representatives_path, index=False)
    np.savez(
        model_path,
        centers=centers.astype(np.float32),
        labels=labels.astype(np.int32),
        distances=distances.astype(np.float32),
    )
    _plot_cluster_curves(profile_frame, curves_path)
    _write_mapping_template(
        template_path,
        cluster_ids=[int(cluster_id) for cluster_id in sorted(np.unique(labels).tolist())],
    )

    original_label_distribution: dict[str, dict[str, int]] | None = None
    if "label_name" in data_frame.columns:
        crosstab = pd.crosstab(assignment_frame["cluster_id"], assignment_frame["original_label_name"])
        original_label_distribution = {
            str(cluster_id): {str(column): int(crosstab.loc[cluster_id, column]) for column in crosstab.columns}
            for cluster_id in crosstab.index
        }

    save_json_summary(
        summary_path,
        {
            "config": experiment_config.to_dict(),
            "inertia": float(cluster_result["inertia"]),
            "iterations": int(cluster_result["iterations"]),
            "cluster_summaries": cluster_summaries,
            "original_label_distribution": original_label_distribution,
            "artifacts": {
                "assignments": str(assignments_path),
                "profiles": str(profiles_path),
                "representatives": str(representatives_path),
                "model": str(model_path),
                "curves": str(curves_path),
                "mapping_template": str(template_path),
            },
        },
    )

    log(f"[产物] assignments={assignments_path}")
    log(f"[产物] profiles={profiles_path}")
    log(f"[产物] representatives={representatives_path}")
    log(f"[产物] curves={curves_path}")
    log(f"[产物] mapping_template={template_path}")
    return {
        "assignments_path": str(assignments_path),
        "profiles_path": str(profiles_path),
        "representatives_path": str(representatives_path),
        "summary_path": str(summary_path),
        "mapping_template": str(template_path),
    }


def main(config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    result = run_fit(config_path=config_path)
    tqdm.write(
        "KMeans 聚类完成，"
        f"profiles={result['profiles_path']} "
        f"mapping_template={result['mapping_template']}"
    )


if __name__ == "__main__":
    main()
