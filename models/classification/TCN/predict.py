"""TCN 分类推理入口。"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from classification.TCN.config import DEFAULT_CONFIG_PATH, detect_device, load_experiment_config
from classification.TCN.constants import LABELS, SEQUENCE_LENGTH
from classification.TCN.dataset import (
    ACTIVE_COLUMNS,
    AGGREGATE_COLUMNS,
    BURST_COLUMNS,
    INDEX_TO_LABEL,
)
from classification.TCN.engine import (
    build_model,
    checkpoint_to_normalization,
    load_checkpoint,
    save_json_summary,
)


class PredictionDataset(Dataset[tuple[torch.Tensor, int]]):
    """用于推理阶段的标准化数据集。"""

    def __init__(self, features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> None:
        safe_std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
        normalized = (features.astype(np.float32) - mean.astype(np.float32)) / safe_std
        self.features = normalized

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return torch.from_numpy(self.features[index]), index


def _single_feature_record_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    aggregate = payload.get("aggregate")
    active = payload.get("active_appliance_count")
    burst = payload.get("burst_event_count")

    if aggregate is None or active is None or burst is None:
        raise ValueError("单样本 json 需要包含 aggregate / active_appliance_count / burst_event_count")
    if not (len(aggregate) == len(active) == len(burst) == SEQUENCE_LENGTH):
        raise ValueError("单样本三条序列长度都必须为 96")

    record: dict[str, Any] = {
        "sample_id": payload.get("sample_id", "single_sample"),
        "house_id": payload.get("house_id", ""),
        "date": payload.get("date", ""),
    }
    for index, value in enumerate(aggregate):
        record[AGGREGATE_COLUMNS[index]] = float(value)
    for index, value in enumerate(active):
        record[ACTIVE_COLUMNS[index]] = float(value)
    for index, value in enumerate(burst):
        record[BURST_COLUMNS[index]] = float(value)
    return record


def _load_input_records(input_path: Path) -> pd.DataFrame:
    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        data_frame = pd.read_csv(input_path)
    elif suffix == ".json":
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            records = [_single_feature_record_from_payload(item) for item in payload]
        elif isinstance(payload, dict):
            records = [_single_feature_record_from_payload(payload)]
        else:
            raise ValueError("json 输入必须是对象或对象数组")
        data_frame = pd.DataFrame(records)
    else:
        raise ValueError(f"暂不支持的输入格式: {input_path.suffix}")

    required_columns = {"sample_id", "house_id", "date", *AGGREGATE_COLUMNS, *ACTIVE_COLUMNS, *BURST_COLUMNS}
    missing_columns = required_columns.difference(data_frame.columns)
    if missing_columns:
        raise ValueError(f"推理输入缺少必要字段: {sorted(missing_columns)}")
    return data_frame


def _extract_features_and_metadata(data_frame: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    aggregate = data_frame[AGGREGATE_COLUMNS].to_numpy(dtype=np.float32)
    active = data_frame[ACTIVE_COLUMNS].to_numpy(dtype=np.float32)
    burst = data_frame[BURST_COLUMNS].to_numpy(dtype=np.float32)
    features = np.stack([aggregate, active, burst], axis=-1)
    metadata = data_frame[[column for column in ["sample_id", "house_id", "date"] if column in data_frame.columns]].copy()
    if "label_name" in data_frame.columns:
        metadata["label_name"] = data_frame["label_name"]
    return features, metadata


def _predict_probabilities(
    features: np.ndarray,
    checkpoint_path: Path,
    experiment_config,
    batch_size: int,
) -> pd.DataFrame:
    device_name = detect_device()
    device = torch.device(device_name)
    checkpoint = load_checkpoint(checkpoint_path, device)
    feature_mean, feature_std = checkpoint_to_normalization(checkpoint)

    dataset = PredictionDataset(features=features, mean=feature_mean, std=feature_std)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_model(experiment_config.model).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    probability_batches: list[np.ndarray] = []
    with torch.no_grad():
        for feature_batch, _ in data_loader:
            feature_batch = feature_batch.to(device)
            logits = model(feature_batch)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            probability_batches.append(probabilities)

    probabilities = np.concatenate(probability_batches, axis=0)
    probability_columns = {
        f"prob_{label}": probabilities[:, index]
        for index, label in enumerate(LABELS)
    }
    prediction_indices = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)

    return pd.DataFrame(
        {
            "predicted_label": [INDEX_TO_LABEL[int(index)] for index in prediction_indices],
            "confidence": confidence,
            **probability_columns,
            "runtime_device": device_name,
            "runtime_loss": "CrossEntropyLoss",
        }
    )


def predict_batch_from_path(input_path: Path, config_path: Path = DEFAULT_CONFIG_PATH) -> pd.DataFrame:
    experiment_config = load_experiment_config(config_path=config_path)
    predict_config = experiment_config.predict
    checkpoint_path = predict_config.checkpoint_path or (experiment_config.train.output_dir / "best_model.pt")

    data_frame = _load_input_records(input_path)
    features, metadata = _extract_features_and_metadata(data_frame)
    prediction_df = _predict_probabilities(
        features=features,
        checkpoint_path=checkpoint_path,
        experiment_config=experiment_config,
        batch_size=predict_config.batch_size,
    )
    return pd.concat([metadata.reset_index(drop=True), prediction_df], axis=1)


def predict_single_sample(sample: dict[str, Any], config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    experiment_config = load_experiment_config(config_path=config_path)
    predict_config = experiment_config.predict
    checkpoint_path = predict_config.checkpoint_path or (experiment_config.train.output_dir / "best_model.pt")

    single_df = pd.DataFrame([_single_feature_record_from_payload(sample)])
    features, metadata = _extract_features_and_metadata(single_df)
    prediction_df = _predict_probabilities(
        features=features,
        checkpoint_path=checkpoint_path,
        experiment_config=experiment_config,
        batch_size=1,
    )
    result = pd.concat([metadata.reset_index(drop=True), prediction_df], axis=1).iloc[0].to_dict()
    return result


def run_prediction(
    input_path: Path,
    config_path: Path = DEFAULT_CONFIG_PATH,
    output_path: Path | None = None,
) -> Path:
    experiment_config = load_experiment_config(config_path=config_path)
    predict_config = experiment_config.predict
    prediction_df = predict_batch_from_path(input_path=input_path, config_path=config_path)

    if output_path is None:
        if input_path.suffix.lower() == ".json":
            output_path = predict_config.output_dir / "single_prediction.json"
        else:
            output_path = predict_config.output_dir / "prediction_results.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if input_path.suffix.lower() == ".json" and len(prediction_df) == 1:
        save_json_summary(output_path, prediction_df.iloc[0].to_dict())
    else:
        prediction_df.to_csv(output_path, index=False)
    return output_path


def main(input_path: Path, config_path: Path = DEFAULT_CONFIG_PATH, output_path: Path | None = None) -> None:
    result_path = run_prediction(input_path=input_path, config_path=config_path, output_path=output_path)
    print(f"推理完成，结果已写入: {result_path}")
