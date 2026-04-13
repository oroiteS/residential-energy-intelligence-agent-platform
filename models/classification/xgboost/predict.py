"""XGBoost 分类推理入口。"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from classification.TCN.dataset import AGGREGATE_COLUMNS
from classification.XGBoost.config import DEFAULT_CONFIG_PATH, load_experiment_config
from classification.XGBoost.constants import LABELS
from classification.XGBoost.dataset import INDEX_TO_LABEL, load_prediction_samples, samples_to_xy
from classification.XGBoost.engine import (
    ensure_xgboost_available,
    load_model,
    load_metadata,
    predict_probabilities,
    save_json_summary,
)


def _single_feature_record_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    aggregate = payload.get("aggregate")
    if aggregate is None or len(aggregate) != len(AGGREGATE_COLUMNS):
        raise ValueError("单样本 json 需要提供长度为 96 的 aggregate")
    date_value = payload.get("date")
    if date_value is None:
        raise ValueError("单样本 json 需要提供 date，用于构造星期特征")

    record: dict[str, Any] = {
        "sample_id": payload.get("sample_id", "single_sample"),
        "house_id": payload.get("house_id", ""),
        "date": str(date_value),
    }
    for index, value in enumerate(aggregate):
        record[AGGREGATE_COLUMNS[index]] = float(value)
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

    required_columns = {"sample_id", "house_id", "date", *AGGREGATE_COLUMNS}
    missing_columns = required_columns.difference(data_frame.columns)
    if missing_columns:
        raise ValueError(f"推理输入缺少必要字段: {sorted(missing_columns)}")
    return data_frame


def _predict_probabilities(
    features: np.ndarray,
    checkpoint_path: Path,
    num_boost_round: int,
) -> pd.DataFrame:
    booster = load_model(checkpoint_path)
    probabilities = predict_probabilities(
        booster,
        features,
        fallback_rounds=num_boost_round,
    )
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
            "runtime_library": "xgboost",
        }
    )


def predict_batch_from_path(input_path: Path, config_path: Path = DEFAULT_CONFIG_PATH) -> pd.DataFrame:
    ensure_xgboost_available()
    experiment_config = load_experiment_config(config_path=config_path)
    predict_config = experiment_config.predict
    checkpoint_path = predict_config.checkpoint_path or (experiment_config.train.output_dir / "best_model.json")
    data_frame = _load_input_records(input_path)
    samples = load_prediction_samples(data_frame)
    features, _, metadata = samples_to_xy(samples)
    prediction_df = _predict_probabilities(
        features=features,
        checkpoint_path=checkpoint_path,
        num_boost_round=experiment_config.model.num_boost_round,
    )
    if "label_name" in data_frame.columns:
        metadata["label_name"] = data_frame["label_name"].astype(str)
    return pd.concat([metadata.reset_index(drop=True), prediction_df], axis=1)


def predict_single_sample(sample: dict[str, Any], config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    single_df = pd.DataFrame([_single_feature_record_from_payload(sample)])
    samples = load_prediction_samples(single_df)
    features, _, metadata = samples_to_xy(samples)
    experiment_config = load_experiment_config(config_path=config_path)
    checkpoint_path = experiment_config.predict.checkpoint_path or (
        experiment_config.train.output_dir / "best_model.json"
    )
    prediction_df = _predict_probabilities(
        features=features,
        checkpoint_path=checkpoint_path,
        num_boost_round=experiment_config.model.num_boost_round,
    )
    return pd.concat([metadata.reset_index(drop=True), prediction_df], axis=1).iloc[0].to_dict()


def run_prediction(
    input_path: Path,
    config_path: Path = DEFAULT_CONFIG_PATH,
    output_path: Path | None = None,
) -> Path:
    experiment_config = load_experiment_config(config_path=config_path)
    prediction_df = predict_batch_from_path(input_path=input_path, config_path=config_path)
    checkpoint_path = experiment_config.predict.checkpoint_path or (
        experiment_config.train.output_dir / "best_model.json"
    )
    if output_path is None:
        if input_path.suffix.lower() == ".json":
            output_path = experiment_config.predict.output_dir / "single_prediction.json"
        else:
            output_path = experiment_config.predict.output_dir / "prediction_results.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if input_path.suffix.lower() == ".json" and len(prediction_df) == 1:
        save_json_summary(output_path, prediction_df.iloc[0].to_dict())
    else:
        prediction_df.to_csv(output_path, index=False)

    metadata_path = checkpoint_path.with_name("model_metadata.json")
    metadata = load_metadata(metadata_path)
    save_json_summary(
        experiment_config.predict.output_dir / "prediction_summary.json",
        {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "rows": len(prediction_df),
            "best_iteration": metadata.get("best_iteration"),
            "config": experiment_config.to_dict(),
        },
    )
    return output_path


def main(input_path: Path, config_path: Path = DEFAULT_CONFIG_PATH, output_path: Path | None = None) -> None:
    result_path = run_prediction(input_path=input_path, config_path=config_path, output_path=output_path)
    print(f"推理完成，结果已写入: {result_path}")
