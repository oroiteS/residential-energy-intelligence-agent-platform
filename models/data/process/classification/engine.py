"""KMeans 聚类公共逻辑。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _compute_squared_distances(data: np.ndarray, centers: np.ndarray) -> np.ndarray:
    # 这里避免直接走大矩阵乘法路径。
    # 在部分 macOS / BLAS 组合下，即使输入有限，也可能在 matmul 时抛出 overflow / invalid 警告。
    data64 = data.astype(np.float64, copy=False)
    centers64 = centers.astype(np.float64, copy=False)
    num_samples = data64.shape[0]
    num_centers = centers64.shape[0]
    distances = np.empty((num_samples, num_centers), dtype=np.float32)

    chunk_size = 8192
    for start in range(0, num_samples, chunk_size):
        end = min(start + chunk_size, num_samples)
        chunk = data64[start:end]
        diff = chunk[:, None, :] - centers64[None, :, :]
        chunk_distances = np.sum(diff * diff, axis=2, dtype=np.float64)
        distances[start:end] = np.maximum(chunk_distances, 0.0).astype(np.float32)
    return distances


def _init_centers_kmeans_pp(data: np.ndarray, n_clusters: int, rng: np.random.Generator) -> np.ndarray:
    num_samples = data.shape[0]
    centers = np.empty((n_clusters, data.shape[1]), dtype=np.float32)
    first_index = int(rng.integers(0, num_samples))
    centers[0] = data[first_index]

    closest_dist_sq = _compute_squared_distances(data, centers[:1]).reshape(-1)
    for center_index in range(1, n_clusters):
        probability = closest_dist_sq / np.clip(closest_dist_sq.sum(), 1e-12, None)
        next_index = int(rng.choice(num_samples, p=probability))
        centers[center_index] = data[next_index]
        new_dist_sq = _compute_squared_distances(data, centers[center_index : center_index + 1]).reshape(-1)
        closest_dist_sq = np.minimum(closest_dist_sq, new_dist_sq)
    return centers


def _assign_labels(data: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    distances = _compute_squared_distances(data, centers)
    labels = distances.argmin(axis=1).astype(np.int32)
    min_distances = distances[np.arange(len(data)), labels]
    return labels, min_distances


def _update_centers(
    data: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    rng: np.random.Generator,
) -> np.ndarray:
    centers = np.empty((n_clusters, data.shape[1]), dtype=np.float32)
    for cluster_id in range(n_clusters):
        cluster_points = data[labels == cluster_id]
        if len(cluster_points) == 0:
            fallback_index = int(rng.integers(0, len(data)))
            centers[cluster_id] = data[fallback_index]
        else:
            centers[cluster_id] = cluster_points.mean(axis=0)
    return centers


def fit_kmeans(
    data: np.ndarray,
    n_clusters: int,
    random_seed: int,
    n_init: int,
    max_iter: int,
    tol: float,
) -> dict[str, object]:
    if data.ndim != 2:
        raise ValueError("KMeans 输入必须是二维矩阵")
    if len(data) < n_clusters:
        raise ValueError("样本数量不能小于 n_clusters")

    best_result: dict[str, object] | None = None
    for init_index in range(n_init):
        rng = np.random.default_rng(random_seed + init_index)
        centers = _init_centers_kmeans_pp(data, n_clusters=n_clusters, rng=rng)
        previous_inertia: float | None = None

        for iteration in range(1, max_iter + 1):
            labels, min_distances = _assign_labels(data, centers)
            inertia = float(min_distances.sum())
            updated_centers = _update_centers(data, labels, n_clusters=n_clusters, rng=rng)
            center_shift = float(np.linalg.norm(updated_centers - centers))
            centers = updated_centers

            if previous_inertia is not None and abs(previous_inertia - inertia) <= tol:
                break
            if center_shift <= tol:
                break
            previous_inertia = inertia

        result = {
            "labels": labels,
            "centers": centers,
            "inertia": inertia,
            "iterations": iteration,
            "distances": min_distances,
        }
        if best_result is None or result["inertia"] < best_result["inertia"]:
            best_result = result

    if best_result is None:
        raise RuntimeError("KMeans 未生成有效结果")
    return best_result


def save_json_summary(output_path: Path, payload: dict[str, object]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
