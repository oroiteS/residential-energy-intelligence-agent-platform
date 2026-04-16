"""预测服务。"""

from __future__ import annotations

from typing import Any

from app.config import Settings
from app.contracts import ForecastRequest
from app.errors import ServiceUnavailableError, ValidationError
from app.inference.forecast import (
    SUPPORTED_FORECAST_MODEL_TYPES,
    get_required_input_length,
    predict_single_sample_detailed,
)


class ForecastService:
    """预测封装。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def forecast(self, request: ForecastRequest) -> dict[str, Any]:
        normalized_model_type = request.model_type.strip().lower()
        if normalized_model_type not in SUPPORTED_FORECAST_MODEL_TYPES:
            raise ValidationError("当前预测接口仅支持 tft")

        required_input_length = get_required_input_length(
            self.settings.forecast_config_path,
            normalized_model_type,
        )

        if len(request.series) < required_input_length:
            raise ValidationError(
                f"预测输入历史序列长度不足，至少需要 {required_input_length} 个点"
            )

        recent_series = request.series[-required_input_length:]
        expected_profile_days = required_input_length // 96
        if request.profile_probability_days and (
            len(request.profile_probability_days) != expected_profile_days
        ):
            raise ValidationError(
                f"profile_probability_days 数量必须为 {expected_profile_days}"
            )

        try:
            forecast_result = predict_single_sample_detailed(
                sample={
                    "sample_id": f"{request.dataset_id}_{request.forecast_start[:10]}",
                    "house_id": str(request.dataset_id),
                    "input_start": recent_series[0].timestamp,
                    "input_end": recent_series[-1].timestamp,
                    "target_start": request.forecast_start,
                    "series": [
                        {
                            "timestamp": point.timestamp,
                            "aggregate": point.aggregate,
                            "active_appliance_count": point.active_appliance_count,
                            "burst_event_count": point.burst_event_count,
                        }
                        for point in recent_series
                    ],
                    "profile_probability_days": [
                        {
                            "date": item.date,
                            "probabilities": dict(item.probabilities),
                        }
                        for item in request.profile_probability_days
                    ],
                },
                config_path=self.settings.forecast_config_path,
                model_type=normalized_model_type,
            )
        except FileNotFoundError as exc:
            raise ServiceUnavailableError("MODEL_NOT_LOADED", "预测模型权重不存在") from exc
        except Exception as exc:
            raise ServiceUnavailableError("FORECAST_FAILED", f"预测推理失败: {exc}") from exc

        return {
            "model_type": normalized_model_type,
            "sample_id": f"{request.dataset_id}_{request.forecast_start[:10]}",
            "house_id": str(request.dataset_id),
            "input_start": recent_series[0].timestamp,
            "input_end": recent_series[-1].timestamp,
            "predictions": forecast_result["predictions"],
            "profile_probability_source": str(
                forecast_result.get("profile_probability_source", "")
            ),
            "profile_probability_days": list(
                forecast_result.get("profile_probability_days", [])
            ),
            "profile_prior": dict(forecast_result.get("profile_prior", {})),
        }
