"""分类推理服务。"""

from __future__ import annotations

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
    predict_single_sample,
)


class ClassificationService:
    """XGBoost 分类推理封装。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

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
            "supported_models": [
                "xgboost",
                "lstm",
                "transformer_encoder_direct",
                "transformer_encdec_direct",
            ],
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
                "supported_models": [
                    "lstm",
                    "transformer_encoder_direct",
                    "transformer_encdec_direct",
                ],
                "request_mode": "time_range",
                "supported_granularities": ["15min"],
                "summary_schema": "ForecastSummary",
                "raw_output_schema": "predictions[96]",
                "input_spec": {
                    "granularity": "15min",
                    "unit": "w",
                    "history_window_by_model": {
                        "lstm": {
                            "unit": "day",
                            "value": 7,
                        },
                        "transformer_encoder_direct": {
                            "unit": "day",
                            "value": 7,
                        },
                        "transformer_encdec_direct": {
                            "unit": "day",
                            "value": 7,
                        },
                    },
                    "min_history_length_by_model": {
                        "lstm": 672,
                        "transformer_encoder_direct": 672,
                        "transformer_encdec_direct": 672,
                    },
                    "target_length": 96,
                    "feature_names": [
                        "aggregate",
                        "slot_sin",
                        "slot_cos",
                        "weekday_sin",
                        "weekday_cos",
                        "active_appliance_count",
                        "burst_event_count",
                    ],
                    "temporal_features_from_timestamp": True,
                },
            },
        }

    def predict(self, request: PredictRequest) -> dict[str, Any]:
        if request.model_type != "xgboost":
            raise ValidationError("当前分类接口仅支持 xgboost")

        if len(request.series) != SEQUENCE_LENGTH:
            raise ValidationError("分类输入序列长度必须为 96")

        try:
            result = predict_single_sample(
                sample={
                    "sample_id": f"{request.dataset_id}_{request.window.start[:10]}",
                    "house_id": str(request.dataset_id),
                    "date": request.window.start[:10],
                    "aggregate": [point.aggregate for point in request.series],
                },
                config_path=self.settings.classification_config_path,
            )
        except FileNotFoundError as exc:
            raise ServiceUnavailableError("MODEL_NOT_LOADED", "分类模型权重不存在") from exc
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
