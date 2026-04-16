"""分类推理服务。"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from app.config import Settings
from app.contracts import PredictRequest
from app.errors import ServiceUnavailableError, ValidationError
from app.inference.classification import (
    FEATURE_NAMES,
    LABEL_DISPLAY_NAMES,
    LABELS,
    SEQUENCE_LENGTH,
    get_checkpoint_path,
)


class ClassificationService:
    """XGBoost 分类推理封装。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.worker_script_path = (
            Path(__file__).resolve().parents[1]
            / "inference"
            / "classification_worker.py"
        )

    def health(self) -> dict[str, Any]:
        checkpoint_path = get_checkpoint_path(self.settings.classification_config_path)
        return {
            "status": "up",
            "service": "python-robyn-backend",
            "model_loaded": checkpoint_path.exists(),
            "classification_config_path": str(self.settings.classification_config_path),
            "classification_checkpoint_path": str(checkpoint_path),
        }

    def model_info(self) -> dict[str, Any]:
        return {
            "service_version": "v1",
            "supported_models": ["xgboost", "tft"],
            "classification": {
                "supported_models": ["xgboost"],
                "labels": LABELS,
                "label_definitions": [
                    {
                        "key": label,
                        "display_name": LABEL_DISPLAY_NAMES.get(label, label),
                    }
                    for label in LABELS
                ],
                "input_spec": {
                    "granularity": "15min",
                    "unit": "w",
                    "history_window": {
                        "unit": "day",
                        "value": 1,
                        "config_key": "model_history_window_config.classification_days",
                        "configurable": True,
                    },
                    "min_history_length": SEQUENCE_LENGTH,
                    "feature_names": list(FEATURE_NAMES),
                    "temporal_features_from_timestamp": True,
                    "derived_feature_count": 45,
                },
                "output_spec": {
                    "predicted_label": "string",
                    "confidence": "float",
                    "probabilities": "map[string,float]",
                },
            },
            "forecasting": {
                "supported_models": ["tft"],
                "request_mode": "time_range",
                "supported_granularities": ["15min"],
                "summary_schema": "ForecastSummary",
                "raw_output_schema": "predictions[96]",
                "input_spec": {
                    "granularity": "15min",
                    "unit": "w",
                    "history_window": {
                        "unit": "day",
                        "value": 7,
                    },
                    "min_history_length": 672,
                    "target_length": 96,
                    "raw_feature_names": [
                        "timestamp",
                        "aggregate",
                        "active_appliance_count",
                        "burst_event_count",
                    ],
                    "derived_feature_names": [
                        "aggregate",
                        "slot_sin",
                        "slot_cos",
                        "weekday_sin",
                        "weekday_cos",
                        "profile_prob_afternoon_peak",
                        "profile_prob_all_day_low",
                        "profile_prob_day_low_night_high",
                        "profile_prob_morning_peak",
                    ],
                    "temporal_features_from_timestamp": True,
                    "profile_prior_source": "xgboost_day_profile_classifier",
                },
            },
        }

    def _predict_via_worker(self, sample: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "sample": sample,
            "config_path": str(self.settings.classification_config_path),
        }
        try:
            completed = subprocess.run(
                [sys.executable, str(self.worker_script_path)],
                input=json.dumps(payload, ensure_ascii=False),
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ServiceUnavailableError(
                "CLASSIFICATION_TIMEOUT",
                "分类推理子进程执行超时",
            ) from exc

        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            raise ServiceUnavailableError(
                "CLASSIFICATION_FAILED",
                f"分类推理失败: {stderr or '分类子进程异常退出'}",
            )

        stdout = completed.stdout.strip()
        if not stdout:
            raise ServiceUnavailableError(
                "CLASSIFICATION_FAILED",
                "分类推理失败: 分类子进程未返回结果",
            )

        try:
            result = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise ServiceUnavailableError(
                "CLASSIFICATION_FAILED",
                "分类推理失败: 分类子进程返回了非法 JSON",
            ) from exc
        if not isinstance(result, dict):
            raise ServiceUnavailableError(
                "CLASSIFICATION_FAILED",
                "分类推理失败: 分类子进程返回格式错误",
            )
        return result

    def predict(self, request: PredictRequest) -> dict[str, Any]:
        if request.model_type != "xgboost":
            raise ValidationError("当前分类接口仅支持 xgboost")

        if len(request.series) != SEQUENCE_LENGTH:
            raise ValidationError("分类输入序列长度必须为 96")

        try:
            result = self._predict_via_worker(
                sample={
                    "sample_id": f"{request.dataset_id}_{request.window.start[:10]}",
                    "house_id": str(request.dataset_id),
                    "date": request.window.start[:10],
                    "aggregate": [point.aggregate for point in request.series],
                }
            )
        except FileNotFoundError as exc:
            raise ServiceUnavailableError("MODEL_NOT_LOADED", "分类模型权重不存在") from exc
        except ServiceUnavailableError:
            raise
        except Exception as exc:
            raise ServiceUnavailableError("CLASSIFICATION_FAILED", f"分类推理失败: {exc}") from exc

        return {
            "model_type": request.model_type,
            "sample_id": f"{request.dataset_id}_{request.window.start[:10]}",
            "house_id": str(request.dataset_id),
            "date": request.window.start[:10],
            "predicted_label": str(result.get("predicted_label", "")),
            "confidence": float(result.get("confidence", 0.0)),
            "probabilities": {
                str(label): float(probability)
                for label, probability in (result.get("probabilities", {}) or {}).items()
            },
            "runtime_library": str(result.get("runtime_library", "")),
        }
