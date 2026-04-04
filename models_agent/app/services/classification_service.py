"""分类推理服务。"""

from __future__ import annotations

from typing import Any

from app.config import Settings
from app.contracts import PredictRequest
from app.errors import ServiceUnavailableError, ValidationError
from app.inference.classification import LABELS, SEQUENCE_LENGTH, predict_single_sample


class ClassificationService:
    """TCN 分类推理封装。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def health(self) -> dict[str, Any]:
        checkpoint_path = self.settings.classification_config_path
        return {
            "status": "up",
            "service": "python-robyn-backend",
            "model_loaded": checkpoint_path.exists(),
            "classification_config_path": str(checkpoint_path),
        }

    def model_info(self) -> dict[str, Any]:
        return {
            "service_version": "v1",
            "supported_models": ["tcn", "lstm"],
            "reserved_models": ["transformer"],
            "classification": {
                "labels": LABELS,
                "label_definitions": [
                    {"key": "day_high_night_low", "display_name": "白天高晚上低型"},
                    {"key": "day_low_night_high", "display_name": "白天低晚上高型"},
                    {"key": "all_day_high", "display_name": "全天高负载型"},
                    {"key": "all_day_low", "display_name": "全天低负载型"},
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
                    "feature_names": [
                        "aggregate",
                        "active_appliance_count",
                        "burst_event_count",
                    ],
                },
            },
            "forecasting": {
                "request_mode": "time_range",
                "supported_granularities": ["15min"],
                "summary_schema": "ForecastSummary",
                "raw_output_schema": "predictions[96]",
                "input_spec": {
                    "granularity": "15min",
                    "unit": "w",
                    "history_window": {
                        "unit": "day",
                        "value": 3,
                        "config_key": "model_history_window_config.forecast_history_days",
                        "configurable": True,
                    },
                    "min_history_length": 288,
                    "target_length": 96,
                    "feature_names": [
                        "aggregate",
                        "active_appliance_count",
                        "burst_event_count",
                    ],
                },
            },
        }

    def predict(self, request: PredictRequest) -> dict[str, Any]:
        if request.model_type != "tcn":
            raise ValidationError("当前分类接口仅支持 tcn")

        if len(request.series) != SEQUENCE_LENGTH:
            raise ValidationError("分类输入序列长度必须为 96")

        try:
            result = predict_single_sample(
                aggregate=[point.aggregate for point in request.series],
                active_appliance_count=[point.active_appliance_count for point in request.series],
                burst_event_count=[point.burst_event_count for point in request.series],
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
            "prob_day_high_night_low": float(result.get("prob_day_high_night_low", 0.0)),
            "prob_day_low_night_high": float(result.get("prob_day_low_night_high", 0.0)),
            "prob_all_day_high": float(result.get("prob_all_day_high", 0.0)),
            "prob_all_day_low": float(result.get("prob_all_day_low", 0.0)),
            "runtime_device": str(result.get("runtime_device", "")),
            "runtime_loss": str(result.get("runtime_loss", "")),
        }
