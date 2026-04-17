"""请求与响应结构。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from app.errors import ValidationError


AGENT_CONTEXT_OBJECT_KEYS = {
    "dataset",
    "analysis_summary",
    "classification_result",
    "forecast_summary",
    "recent_history_summary",
    "user_preferences",
    "conversation_state",
}
AGENT_CONTEXT_LIST_KEYS = {
    "rule_advices",
    "historical_classification_results",
    "future_classification_results",
    "future_forecast_summaries",
}


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


def _normalize_agent_context_payload(payload: Any) -> dict[str, Any]:
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ValidationError("context 必须是对象")

    normalized = dict(payload)

    for key in AGENT_CONTEXT_OBJECT_KEYS:
        value = normalized.get(key, {})
        if value is None:
            value = {}
        if not isinstance(value, dict):
            raise ValidationError(f"context.{key} 必须是对象")
        normalized[key] = value

    for list_key in AGENT_CONTEXT_LIST_KEYS:
        raw_items = normalized.get(list_key, [])
        if raw_items is None:
            raw_items = []
        if not isinstance(raw_items, list):
            raise ValidationError(f"context.{list_key} 必须是数组")
        normalized[list_key] = raw_items

    raw_rule_advices = normalized.get("rule_advices", [])
    for index, item in enumerate(raw_rule_advices):
        if not isinstance(item, (dict, str)):
            raise ValidationError(
                f"context.rule_advices[{index}] 只支持对象或字符串"
            )
    try:
        from app.agent.state import AgentContext
    except ModuleNotFoundError:
        return normalized

    try:
        validated_context = AgentContext.from_payload(normalized)
    except Exception as exc:
        raise ValidationError(f"context 结构不合法: {exc}") from exc
    return validated_context.model_dump()


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
            model_type=str(payload.get("model_type", "xgboost")),
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
    profile_probability_days: list["ProfileProbabilityDay"]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ForecastRequest":
        metadata = payload.get("metadata", {}) or {}
        try:
            return cls(
                model_type=str(payload.get("model_type", "tft")),
                dataset_id=int(payload["dataset_id"]),
                forecast_start=str(payload["forecast_start"]),
                forecast_end=str(payload["forecast_end"]),
                granularity=str(payload.get("granularity", "15min")),
                unit=str(metadata.get("unit", "w")),
                series=_load_series(payload),
                profile_probability_days=[
                    ProfileProbabilityDay.from_dict(item)
                    for item in (payload.get("profile_probability_days", []) or [])
                ],
            )
        except KeyError as exc:
            raise ValidationError(f"请求缺少字段: {exc.args[0]}") from exc


@dataclass(slots=True)
class ProfileProbabilityDay:
    date: str
    probabilities: dict[str, float]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProfileProbabilityDay":
        try:
            date = str(payload["date"]).strip()
        except KeyError as exc:
            raise ValidationError(
                "profile_probability_days 缺少字段: date"
            ) from exc
        if not date:
            raise ValidationError("profile_probability_days.date 不能为空")
        try:
            datetime.fromisoformat(date)
        except Exception as exc:
            raise ValidationError(
                "profile_probability_days.date 必须是合法日期"
            ) from exc

        raw_probabilities = payload.get("probabilities", {})
        if not isinstance(raw_probabilities, dict) or not raw_probabilities:
            raise ValidationError(
                "profile_probability_days.probabilities 必须是非空对象"
            )

        normalized_probabilities: dict[str, float] = {}
        for label, probability in raw_probabilities.items():
            try:
                normalized_probability = float(probability)
            except (TypeError, ValueError) as exc:
                raise ValidationError(
                    "profile_probability_days.probabilities 的值必须是数字"
                ) from exc
            if normalized_probability < 0.0 or normalized_probability > 1.0:
                raise ValidationError(
                    "profile_probability_days.probabilities 的值必须在 0 到 1 之间"
                )
            normalized_probabilities[str(label)] = normalized_probability

        return cls(
            date=date,
            probabilities=normalized_probabilities,
        )

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

        context_payload = _normalize_agent_context_payload(payload.get("context", {}))

        return cls(
            dataset_id=int(payload["dataset_id"]),
            session_id=int(payload["session_id"]),
            question=question,
            history=[AgentHistoryItem.from_dict(item) for item in history_payload],
            context=context_payload,
        )


@dataclass(slots=True)
class AgentReportSummaryRequest:
    dataset_id: int
    context: dict[str, Any]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AgentReportSummaryRequest":
        context_payload = _normalize_agent_context_payload(payload.get("context", {}))

        return cls(
            dataset_id=int(payload["dataset_id"]),
            context=context_payload,
        )


@dataclass(slots=True)
class PDFRenderRequest:
    markdown: str
    title: str
    author: str
    date: str
    theme: str
    cover: bool
    toc: bool
    header_title: str
    footer_left: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PDFRenderRequest":
        markdown = str(payload.get("markdown", "")).strip()
        if not markdown:
            raise ValidationError("markdown 不能为空")

        title = str(payload.get("title", "")).strip()
        if not title:
            raise ValidationError("title 不能为空")

        return cls(
            markdown=markdown,
            title=title,
            author=str(payload.get("author", "")).strip(),
            date=str(payload.get("date", "")).strip(),
            theme=str(payload.get("theme", "github-light")).strip() or "github-light",
            cover=bool(payload.get("cover", False)),
            toc=bool(payload.get("toc", False)),
            header_title=str(payload.get("header_title", "")).strip(),
            footer_left=str(payload.get("footer_left", "")).strip(),
        )
