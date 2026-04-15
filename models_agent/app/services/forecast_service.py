"""预测服务。"""

from __future__ import annotations

from typing import Any

from app.config import Settings
from app.contracts import ForecastRequest
from app.errors import ServiceUnavailableError, ValidationError
from app.inference.forecast import (
    SUPPORTED_FORECAST_MODEL_TYPES,
    get_required_input_length,
    predict_single_sample,
)


class ForecastService:
    """预测封装。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def forecast(self, request: ForecastRequest) -> dict[str, Any]:
        if request.model_type not in set(SUPPORTED_FORECAST_MODEL_TYPES).union(
            {"transformer"}
        ):
            raise ValidationError(
                "当前预测接口仅支持 lstm / transformer_encoder_direct / "
                "transformer_encdec_direct"
            )

        required_input_length = get_required_input_length(
            self.settings.forecast_config_path,
            request.model_type,
        )

        if len(request.series) < required_input_length:
            raise ValidationError(
                f"预测输入历史序列长度不足，至少需要 {required_input_length} 个点"
            )

        recent_series = request.series[-required_input_length:]
        try:
            predictions = predict_single_sample(
                sample={
                    "sample_id": f"{request.dataset_id}_{request.forecast_start[:10]}",
                    "house_id": str(request.dataset_id),
                    "input_start": recent_series[0].timestamp,
                    "input_end": recent_series[-1].timestamp,
                    "series": [
                        {
                            "timestamp": point.timestamp,
                            "aggregate": point.aggregate,
                            "active_appliance_count": point.active_appliance_count,
                            "burst_event_count": point.burst_event_count,
                        }
                        for point in recent_series
                    ],
                },
                config_path=self.settings.forecast_config_path,
                model_type=request.model_type,
            )
        except FileNotFoundError as exc:
            raise ServiceUnavailableError("MODEL_NOT_LOADED", "预测模型权重不存在") from exc
        except Exception as exc:
            raise ServiceUnavailableError("FORECAST_FAILED", f"预测推理失败: {exc}") from exc

        return {
            "model_type": request.model_type,
            "sample_id": f"{request.dataset_id}_{request.forecast_start[:10]}",
            "house_id": str(request.dataset_id),
            "input_start": recent_series[0].timestamp,
            "input_end": recent_series[-1].timestamp,
            "predictions": predictions,
        }
