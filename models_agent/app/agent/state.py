"""Agent 结构化状态定义。"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


LABEL_TEXT_MAPPING = {
    "afternoon_peak": "下午高峰型",
    "day_low_night_high": "晚上高峰型",
    "all_day_low": "全天平稳型",
    "morning_peak": "上午高峰型",
}

LEGACY_LABEL_MAPPING = {
    "daytime_active": "afternoon_peak",
    "daytime_peak_strong": "morning_peak",
    "flat_stable": "all_day_low",
    "night_dominant": "day_low_night_high",
    "day_high_night_low": "afternoon_peak",
    "all_day_high": "morning_peak",
}

REPORT_SECTION_ORDER = ["总体概览", "行为判断", "预测风险", "附注"]
SUPPORTED_CLASSIFICATION_SCHEMA_VERSIONS = {"v1"}
SUPPORTED_FORECAST_SCHEMA_VERSIONS = {"v1"}
SUPPORTED_CLASSIFICATION_MODEL_TYPES = {"xgboost"}
SUPPORTED_FORECAST_MODEL_TYPES = {"tft"}
SUPPORTED_FORECAST_HORIZONS = {"1d"}
SUPPORTED_RISK_FLAGS = {
    "evening_peak",
    "daytime_peak",
    "high_baseload",
    "abnormal_rise",
    "peak_overlap_risk",
}


def label_text(label: str) -> str:
    """将标签键转换为可读中文。"""

    return LABEL_TEXT_MAPPING.get(label, label)


def normalize_classification_label(label: str) -> str:
    """将历史标签统一映射到当前分类标签体系。"""

    normalized = str(label).strip()
    if not normalized:
        return normalized
    return LEGACY_LABEL_MAPPING.get(normalized, normalized)


class AgentIntent(str, Enum):
    """问答意图。"""

    OVERVIEW = "overview"
    CLASSIFICATION = "classification"
    FORECAST = "forecast"
    ADVICE = "advice"
    RISK = "risk"
    FOLLOW_UP = "follow_up"


class ConfidenceLevel(str, Enum):
    """输出置信等级。"""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class LooseModel(BaseModel):
    """允许额外字段透传的基础模型。"""

    model_config = ConfigDict(extra="allow")


class DatasetInfo(LooseModel):
    """数据集信息。"""

    name: str | None = None


class AnalysisSummary(LooseModel):
    """统计摘要。"""

    peak_ratio: float | None = None
    daily_avg_kwh: float | None = None
    max_load_w: float | None = None


class ClassificationResult(LooseModel):
    """分类结果。"""

    schema_version: str = "v1"
    model_type: str = "xgboost"
    predicted_label: str | None = None
    confidence: float | None = None
    label_display_name: str | None = None
    probabilities: dict[str, float] = Field(default_factory=dict)

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, value: str) -> str:
        normalized = str(value).strip() or "v1"
        if normalized not in SUPPORTED_CLASSIFICATION_SCHEMA_VERSIONS:
            raise ValueError("classification_result.schema_version 仅支持 v1")
        return normalized

    @field_validator("model_type")
    @classmethod
    def validate_model_type(cls, value: str) -> str:
        normalized = str(value).strip() or "xgboost"
        if normalized not in SUPPORTED_CLASSIFICATION_MODEL_TYPES:
            raise ValueError("classification_result.model_type 仅支持 xgboost")
        return normalized

    @field_validator("predicted_label")
    @classmethod
    def validate_predicted_label(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = normalize_classification_label(value)
        if normalized not in LABEL_TEXT_MAPPING:
            raise ValueError("classification_result.predicted_label 不在支持的标签集合中")
        return normalized

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value: float | None) -> float | None:
        if value is None:
            return None
        normalized = float(value)
        if normalized < 0 or normalized > 1:
            raise ValueError("classification_result.confidence 必须在 0 到 1 之间")
        return normalized

    @field_validator("probabilities")
    @classmethod
    def validate_probabilities(cls, value: dict[str, float]) -> dict[str, float]:
        normalized: dict[str, float] = {}
        for label, probability in dict(value or {}).items():
            normalized_label = normalize_classification_label(label)
            if normalized_label not in LABEL_TEXT_MAPPING:
                raise ValueError("classification_result.probabilities 包含未定义标签")
            probability_value = float(probability)
            if probability_value < 0 or probability_value > 1:
                raise ValueError("classification_result.probabilities 的值必须在 0 到 1 之间")
            normalized[normalized_label] = probability_value
        return normalized

    @model_validator(mode="after")
    def populate_derived_fields(self) -> "ClassificationResult":
        if self.predicted_label and not self.label_display_name:
            self.label_display_name = label_text(self.predicted_label)
        if self.confidence is None and self.predicted_label and self.probabilities:
            self.confidence = self.probabilities.get(self.predicted_label)
        return self


class ForecastSummary(LooseModel):
    """预测摘要。"""

    schema_version: str = "v1"
    model_type: str = "tft"
    forecast_horizon: str = "1d"
    peak_period: str | None = None
    predicted_avg_load_w: float | None = None
    predicted_peak_load_w: float | None = None
    predicted_total_kwh: float | None = None
    risk_flags: list[str] = Field(default_factory=list)
    confidence_hint: str | None = None

    @field_validator("schema_version")
    @classmethod
    def validate_forecast_schema_version(cls, value: str) -> str:
        normalized = str(value).strip() or "v1"
        if normalized not in SUPPORTED_FORECAST_SCHEMA_VERSIONS:
            raise ValueError("forecast_summary.schema_version 仅支持 v1")
        return normalized

    @field_validator("model_type")
    @classmethod
    def validate_forecast_model_type(cls, value: str) -> str:
        normalized = str(value).strip() or "tft"
        if normalized not in SUPPORTED_FORECAST_MODEL_TYPES:
            raise ValueError("forecast_summary.model_type 不在支持的模型集合中")
        return normalized

    @field_validator("forecast_horizon")
    @classmethod
    def validate_forecast_horizon(cls, value: str) -> str:
        normalized = str(value).strip() or "1d"
        if normalized not in SUPPORTED_FORECAST_HORIZONS:
            raise ValueError("forecast_summary.forecast_horizon 当前仅支持 1d")
        return normalized

    @field_validator(
        "predicted_avg_load_w",
        "predicted_peak_load_w",
        "predicted_total_kwh",
    )
    @classmethod
    def validate_non_negative_numeric_fields(cls, value: float | None) -> float | None:
        if value is None:
            return None
        normalized = float(value)
        if normalized < 0:
            raise ValueError("forecast_summary 数值字段不能为负数")
        return normalized

    @field_validator("risk_flags")
    @classmethod
    def validate_risk_flags(cls, value: list[str]) -> list[str]:
        normalized: list[str] = []
        for item in list(value or []):
            flag = str(item).strip()
            if flag not in SUPPORTED_RISK_FLAGS:
                raise ValueError("forecast_summary.risk_flags 包含未定义标签")
            if flag not in normalized:
                normalized.append(flag)
        return normalized

    @field_validator("confidence_hint")
    @classmethod
    def validate_confidence_hint(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip().lower()
        if normalized not in {"high", "medium", "low"}:
            raise ValueError("forecast_summary.confidence_hint 仅支持 high / medium / low")
        return normalized


class TimelineClassificationResult(LooseModel):
    """时间线分类结果。"""

    date: str | None = None
    predicted_label: str | None = None
    confidence: float | None = None
    label_display_name: str | None = None
    probabilities: dict[str, float] = Field(default_factory=dict)
    source: str | None = None
    day_offset: int | None = None

    @field_validator("predicted_label")
    @classmethod
    def validate_predicted_label(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = normalize_classification_label(value)
        if normalized not in LABEL_TEXT_MAPPING:
            raise ValueError("timeline_classification.predicted_label 不在支持的标签集合中")
        return normalized

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value: float | None) -> float | None:
        if value is None:
            return None
        normalized = float(value)
        if normalized < 0 or normalized > 1:
            raise ValueError("timeline_classification.confidence 必须在 0 到 1 之间")
        return normalized

    @field_validator("probabilities")
    @classmethod
    def validate_probabilities(cls, value: dict[str, float]) -> dict[str, float]:
        normalized: dict[str, float] = {}
        for label, probability in dict(value or {}).items():
            normalized_label = normalize_classification_label(label)
            if normalized_label not in LABEL_TEXT_MAPPING:
                raise ValueError("timeline_classification.probabilities 包含未定义标签")
            probability_value = float(probability)
            if probability_value < 0 or probability_value > 1:
                raise ValueError("timeline_classification.probabilities 的值必须在 0 到 1 之间")
            normalized[normalized_label] = probability_value
        return normalized

    @model_validator(mode="after")
    def populate_derived_fields(self) -> "TimelineClassificationResult":
        if self.predicted_label and not self.label_display_name:
            self.label_display_name = label_text(self.predicted_label)
        if self.confidence is None and self.predicted_label and self.probabilities:
            self.confidence = self.probabilities.get(self.predicted_label)
        return self


class RecentHistorySummary(LooseModel):
    """近期历史摘要。"""

    avg_active_appliance_count: float | None = None
    avg_burst_event_count: float | None = None
    max_load_w: float | None = None


class RuleAdvice(LooseModel):
    """规则建议。"""

    key: str
    action: str
    summary: str | None = None
    reason: str | None = None
    priority: int = 50
    category: str | None = None

    @classmethod
    def from_any(cls, payload: Any, index: int) -> "RuleAdvice":
        if isinstance(payload, str):
            text = payload.strip()
            return cls(key=f"rule_advice_{index}", action=text, summary=text)

        if isinstance(payload, dict):
            action = str(payload.get("action") or payload.get("summary") or "").strip()
            summary = str(payload.get("summary") or action).strip() or None
            reason = str(payload.get("reason") or "").strip() or None
            raw_priority = payload.get("priority", payload.get("score", 50))
            try:
                priority = int(raw_priority)
            except (TypeError, ValueError):
                priority = 50

            return cls(
                key=str(payload.get("key") or f"rule_advice_{index}"),
                action=action or f"执行规则建议 {index + 1}",
                summary=summary,
                reason=reason,
                priority=priority,
                category=str(payload.get("category") or "").strip() or None,
                **{
                    field: value
                    for field, value in payload.items()
                    if field not in {"key", "action", "summary", "reason", "priority", "score", "category"}
                },
            )

        return cls(key=f"rule_advice_{index}", action=f"执行规则建议 {index + 1}")


class UserPreferences(LooseModel):
    """用户偏好。"""

    focus: str | None = None
    objective: str | None = None
    preferred_response_style: str | None = None
    comfort_priority: str | None = None
    saving_priority: str | None = None


class ConversationState(LooseModel):
    """会话状态。"""

    active_goal: str | None = None
    last_intent: str | None = None
    last_actions: list[str] = Field(default_factory=list)


class AgentContext(LooseModel):
    """Agent 统一上下文。"""

    dataset: DatasetInfo = Field(default_factory=DatasetInfo)
    analysis_summary: AnalysisSummary = Field(default_factory=AnalysisSummary)
    classification_result: ClassificationResult = Field(default_factory=ClassificationResult)
    forecast_summary: ForecastSummary = Field(default_factory=ForecastSummary)
    historical_classification_results: list[TimelineClassificationResult] = Field(default_factory=list)
    future_classification_results: list[TimelineClassificationResult] = Field(default_factory=list)
    future_forecast_summaries: list[ForecastSummary] = Field(default_factory=list)
    recent_history_summary: RecentHistorySummary = Field(default_factory=RecentHistorySummary)
    rule_advices: list[RuleAdvice] = Field(default_factory=list)
    user_preferences: UserPreferences = Field(default_factory=UserPreferences)
    conversation_state: ConversationState = Field(default_factory=ConversationState)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "AgentContext":
        data = dict(payload or {})
        raw_rule_advices = data.get("rule_advices", []) or []
        data["rule_advices"] = [
            RuleAdvice.from_any(item, index)
            for index, item in enumerate(raw_rule_advices)
        ]
        data["historical_classification_results"] = [
            TimelineClassificationResult.model_validate(item)
            for item in (data.get("historical_classification_results", []) or [])
        ]
        data["future_classification_results"] = [
            TimelineClassificationResult.model_validate(item)
            for item in (data.get("future_classification_results", []) or [])
        ]
        data["future_forecast_summaries"] = [
            ForecastSummary.model_validate(item)
            for item in (data.get("future_forecast_summaries", []) or [])
        ]
        return cls.model_validate(data)


class CitationItem(BaseModel):
    """结构化引用项。"""

    key: str = Field(min_length=1)
    label: str = Field(min_length=1)
    value: Any


class MissingInformationItem(BaseModel):
    """信息缺口。"""

    key: str = Field(min_length=1)
    question: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class AdviceCandidate(BaseModel):
    """候选建议。"""

    key: str = Field(min_length=1)
    title: str = Field(min_length=1)
    action: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    priority_score: int = Field(default=50, ge=0, le=100)
    evidence_keys: list[str] = Field(default_factory=list)
    category: str = Field(default="general")


class EvidenceItem(BaseModel):
    """结构化证据。"""

    key: str = Field(min_length=1)
    label: str = Field(min_length=1)
    value: Any
    source: str = Field(min_length=1)
    priority_score: int = Field(default=50, ge=0, le=100)
    summary: str = Field(min_length=1)

    def to_citation(self) -> CitationItem:
        return CitationItem(key=self.key, label=self.label, value=self.value)


class AgentOutput(BaseModel):
    """问答输出。"""

    answer: str = Field(min_length=1)
    citations: list[CitationItem] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    missing_information: list[MissingInformationItem] = Field(default_factory=list)
    confidence_level: ConfidenceLevel = ConfidenceLevel.MEDIUM


class ReportSection(BaseModel):
    """PDF 报告章节。"""

    title: str = Field(min_length=1)
    body: str = Field(min_length=1)


class ReportSummaryOutput(BaseModel):
    """结构化报告摘要。"""

    title: str = Field(min_length=1)
    overview: str = Field(min_length=1)
    sections: list[ReportSection] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


class SessionMemorySnapshot(BaseModel):
    """会话级短期记忆。"""

    session_id: int
    last_intent: AgentIntent | None = None
    active_goal: str | None = None
    recent_questions: list[str] = Field(default_factory=list)
    recent_actions: list[str] = Field(default_factory=list)
    pending_missing_information: list[MissingInformationItem] = Field(default_factory=list)

    def to_prompt_payload(self) -> dict[str, Any]:
        return {
            "last_intent": self.last_intent.value if self.last_intent else None,
            "active_goal": self.active_goal,
            "recent_questions": self.recent_questions[-3:],
            "recent_actions": self.recent_actions[:3],
            "pending_missing_information": [
                item.model_dump() for item in self.pending_missing_information[:3]
            ],
        }
