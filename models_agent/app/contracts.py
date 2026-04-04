"""请求与响应结构。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from app.errors import ValidationError


@dataclass(slots=True)
class TimeSeriesPoint:
    timestamp: str
    aggregate: float
    active_appliance_count: int
    burst_event_count: int

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TimeSeriesPoint":
        try:
            timestamp = str(payload["timestamp"])
            datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except Exception as exc:
            raise ValidationError("timestamp 必须是合法 ISO 8601 时间") from exc

        try:
            return cls(
                timestamp=timestamp,
                aggregate=float(payload["aggregate"]),
                active_appliance_count=int(payload["active_appliance_count"]),
                burst_event_count=int(payload["burst_event_count"]),
            )
        except KeyError as exc:
            raise ValidationError(f"时序点缺少字段: {exc.args[0]}") from exc
        except (TypeError, ValueError) as exc:
            raise ValidationError("时序点字段类型不正确") from exc


@dataclass(slots=True)
class Window:
    start: str
    end: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Window":
        if "start" not in payload or "end" not in payload:
            raise ValidationError("window 必须包含 start 和 end")
        return cls(start=str(payload["start"]), end=str(payload["end"]))


def _load_series(payload: dict[str, Any]) -> list[TimeSeriesPoint]:
    series = payload.get("series")
    if not isinstance(series, list) or not series:
        raise ValidationError("series 必须是非空数组")
    return [TimeSeriesPoint.from_dict(item) for item in series]


@dataclass(slots=True)
class PredictRequest:
    model_type: str
    dataset_id: int
    window: Window
    series: list[TimeSeriesPoint]
    granularity: str
    unit: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PredictRequest":
        metadata = payload.get("metadata", {}) or {}
        return cls(
            model_type=str(payload.get("model_type", "tcn")),
            dataset_id=int(payload["dataset_id"]),
            window=Window.from_dict(payload.get("window", {})),
            series=_load_series(payload),
            granularity=str(metadata.get("granularity", "15min")),
            unit=str(metadata.get("unit", "w")),
        )


@dataclass(slots=True)
class ForecastRequest:
    model_type: str
    dataset_id: int
    forecast_start: str
    forecast_end: str
    granularity: str
    unit: str
    series: list[TimeSeriesPoint]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ForecastRequest":
        metadata = payload.get("metadata", {}) or {}
        try:
            return cls(
                model_type=str(payload.get("model_type", "lstm")),
                dataset_id=int(payload["dataset_id"]),
                forecast_start=str(payload["forecast_start"]),
                forecast_end=str(payload["forecast_end"]),
                granularity=str(payload.get("granularity", "15min")),
                unit=str(metadata.get("unit", "w")),
                series=_load_series(payload),
            )
        except KeyError as exc:
            raise ValidationError(f"请求缺少字段: {exc.args[0]}") from exc


@dataclass(slots=True)
class BacktestRequest:
    model_type: str
    dataset_id: int
    backtest_start: str
    backtest_end: str
    granularity: str
    unit: str
    series: list[TimeSeriesPoint]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BacktestRequest":
        metadata = payload.get("metadata", {}) or {}
        try:
            return cls(
                model_type=str(payload.get("model_type", "lstm")),
                dataset_id=int(payload["dataset_id"]),
                backtest_start=str(payload["backtest_start"]),
                backtest_end=str(payload["backtest_end"]),
                granularity=str(payload.get("granularity", "15min")),
                unit=str(metadata.get("unit", "w")),
                series=_load_series(payload),
            )
        except KeyError as exc:
            raise ValidationError(f"请求缺少字段: {exc.args[0]}") from exc


@dataclass(slots=True)
class AgentHistoryItem:
    role: str
    content: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AgentHistoryItem":
        role = str(payload.get("role", "")).strip()
        content = str(payload.get("content", "")).strip()
        if role not in {"system", "user", "assistant"}:
            raise ValidationError("history.role 只支持 system / user / assistant")
        if not content:
            raise ValidationError("history.content 不能为空")
        return cls(role=role, content=content)


@dataclass(slots=True)
class AgentAskRequest:
    dataset_id: int
    session_id: int
    question: str
    history: list[AgentHistoryItem]
    context: dict[str, Any]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AgentAskRequest":
        question = str(payload.get("question", "")).strip()
        if not question:
            raise ValidationError("question 不能为空")

        history_payload = payload.get("history", []) or []
        if not isinstance(history_payload, list):
            raise ValidationError("history 必须是数组")

        context_payload = payload.get("context", {}) or {}
        if not isinstance(context_payload, dict):
            raise ValidationError("context 必须是对象")

        return cls(
            dataset_id=int(payload["dataset_id"]),
            session_id=int(payload["session_id"]),
            question=question,
            history=[AgentHistoryItem.from_dict(item) for item in history_payload],
            context=context_payload,
        )
