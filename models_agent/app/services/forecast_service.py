"""预测与回测服务。"""

from __future__ import annotations

import math
from typing import Any

from app.config import Settings
from app.contracts import BacktestRequest, ForecastRequest, TimeSeriesPoint
from app.errors import ServiceUnavailableError, ValidationError
from app.inference.forecast import INPUT_LENGTH, TARGET_LENGTH, predict_single_sample


class ForecastService:
    """预测与回测封装。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def forecast(self, request: ForecastRequest) -> dict[str, Any]:
        if request.model_type not in {"lstm", "transformer"}:
            raise ValidationError("当前预测接口仅支持 lstm 或 transformer")

        if len(request.series) < INPUT_LENGTH:
            raise ValidationError("预测输入历史序列长度不足，至少需要 288 个点")

        recent_series = request.series[-INPUT_LENGTH:]
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

    def backtest(self, request: BacktestRequest) -> dict[str, Any]:
        ordered_series = sorted(request.series, key=lambda point: point.timestamp)
        history_series: list[TimeSeriesPoint] = []
        actual_series: list[TimeSeriesPoint] = []

        for point in ordered_series:
            if point.timestamp < request.backtest_start:
                history_series.append(point)
            elif request.backtest_start <= point.timestamp <= request.backtest_end:
                actual_series.append(point)

        if len(history_series) < INPUT_LENGTH:
            raise ValidationError("回测历史序列不足，至少需要 288 个点")
        if len(actual_series) < TARGET_LENGTH:
            raise ValidationError("回测目标区间不足，至少需要 96 个点")

        forecast_payload = ForecastRequest(
            model_type=request.model_type,
            dataset_id=request.dataset_id,
            forecast_start=request.backtest_start,
            forecast_end=request.backtest_end,
            granularity=request.granularity,
            unit=request.unit,
            series=history_series[-INPUT_LENGTH:],
        )
        forecast_result = self.forecast(forecast_payload)
        predicted_values = forecast_result["predictions"][:TARGET_LENGTH]
        actual_values = [point.aggregate for point in actual_series[:TARGET_LENGTH]]

        prediction_items = []
        absolute_errors = []
        squared_errors = []
        actual_sum = 0.0
        smape_terms = []

        for index, actual_value in enumerate(actual_values):
            predicted_value = float(predicted_values[index])
            prediction_items.append(
                {
                    "timestamp": actual_series[index].timestamp,
                    "actual": float(actual_value),
                    "predicted": predicted_value,
                }
            )
            error = abs(actual_value - predicted_value)
            absolute_errors.append(error)
            squared_errors.append((actual_value - predicted_value) ** 2)
            actual_sum += abs(actual_value)
            denominator = (abs(actual_value) + abs(predicted_value)) / 2
            if denominator > 0:
                smape_terms.append(error / denominator)

        mae = sum(absolute_errors) / len(absolute_errors)
        rmse = math.sqrt(sum(squared_errors) / len(squared_errors))
        smape = (sum(smape_terms) / len(smape_terms) * 100) if smape_terms else 0.0
        wape = (sum(absolute_errors) / actual_sum * 100) if actual_sum > 0 else 0.0

        return {
            "model_type": request.model_type,
            "backtest_start": request.backtest_start,
            "backtest_end": request.backtest_end,
            "granularity": request.granularity,
            "predictions": prediction_items,
            "metrics": {
                "mae": round(mae, 6),
                "rmse": round(rmse, 6),
                "smape": round(smape, 6),
                "wape": round(wape, 6),
            },
        }
