"""证据构建与信息缺口识别。"""

from __future__ import annotations

from typing import Any

from app.agent.state import (
    AgentContext,
    AgentIntent,
    ConfidenceLevel,
    EvidenceItem,
    MissingInformationItem,
    SessionMemorySnapshot,
    label_text,
)


class EvidenceBuilder:
    """从结构化上下文中提取证据。"""

    def build(
        self,
        context: AgentContext,
        intent: AgentIntent,
        question: str,
        memory: SessionMemorySnapshot | None,
    ) -> list[EvidenceItem]:
        evidence: list[EvidenceItem] = []

        self._append_if_present(
            evidence,
            key="daily_avg_kwh",
            label="日均用电量",
            value=context.analysis_summary.daily_avg_kwh,
            source="analysis_summary",
            priority_score=70,
            summary=self._build_numeric_summary("日均用电量", context.analysis_summary.daily_avg_kwh, "kWh"),
        )
        self._append_if_present(
            evidence,
            key="peak_ratio",
            label="峰时占比",
            value=context.analysis_summary.peak_ratio,
            source="analysis_summary",
            priority_score=78,
            summary=self._build_ratio_summary("峰时占比", context.analysis_summary.peak_ratio),
        )

        predicted_label = context.classification_result.predicted_label
        if predicted_label:
            evidence.append(
                EvidenceItem(
                    key="predicted_label",
                    label="行为类型",
                    value=predicted_label,
                    source="classification_result",
                    priority_score=85 if intent in {AgentIntent.CLASSIFICATION, AgentIntent.ADVICE} else 70,
                    summary=f"分类结果显示当前用电模式为{label_text(predicted_label)}。",
                )
            )

        self._append_if_present(
            evidence,
            key="confidence",
            label="分类置信度",
            value=context.classification_result.confidence,
            source="classification_result",
            priority_score=76,
            summary=self._build_ratio_summary("分类置信度", context.classification_result.confidence),
        )

        historical_classification_summary = self._build_classification_timeline_summary(
            context.historical_classification_results,
            prefix="过去分类结果",
        )
        if historical_classification_summary:
            evidence.append(
                EvidenceItem(
                    key="historical_classification_results",
                    label="历史分类轨迹",
                    value=[
                        item.model_dump()
                        for item in context.historical_classification_results
                    ],
                    source="historical_classification_results",
                    priority_score=72,
                    summary=historical_classification_summary,
                )
            )

        if context.forecast_summary.peak_period:
            evidence.append(
                EvidenceItem(
                    key="forecast_peak_period",
                    label="预测高负荷时段",
                    value=context.forecast_summary.peak_period,
                    source="forecast_summary",
                    priority_score=88 if intent in {AgentIntent.FORECAST, AgentIntent.RISK, AgentIntent.ADVICE} else 72,
                    summary=f"预测高负荷时段集中在 {context.forecast_summary.peak_period}。",
                )
            )

        self._append_if_present(
            evidence,
            key="predicted_avg_load_w",
            label="预测平均负荷",
            value=context.forecast_summary.predicted_avg_load_w,
            source="forecast_summary",
            priority_score=73,
            summary=self._build_numeric_summary("预测平均负荷", context.forecast_summary.predicted_avg_load_w, "W"),
        )
        self._append_if_present(
            evidence,
            key="predicted_peak_load_w",
            label="预测峰值负荷",
            value=context.forecast_summary.predicted_peak_load_w,
            source="forecast_summary",
            priority_score=84,
            summary=self._build_numeric_summary("预测峰值负荷", context.forecast_summary.predicted_peak_load_w, "W"),
        )
        self._append_if_present(
            evidence,
            key="predicted_total_kwh",
            label="预测总用电量",
            value=context.forecast_summary.predicted_total_kwh,
            source="forecast_summary",
            priority_score=82,
            summary=self._build_numeric_summary("预测总用电量", context.forecast_summary.predicted_total_kwh, "kWh"),
        )

        if context.forecast_summary.risk_flags:
            evidence.append(
                EvidenceItem(
                    key="risk_flags",
                    label="预测风险标签",
                    value=context.forecast_summary.risk_flags,
                    source="forecast_summary",
                    priority_score=86 if intent in {AgentIntent.RISK, AgentIntent.ADVICE} else 68,
                    summary=f"预测风险标签包括 {context.forecast_summary.risk_flags}。",
                )
            )

        future_forecast_summary = self._build_future_forecast_summary(
            context.future_forecast_summaries
        )
        if future_forecast_summary:
            evidence.append(
                EvidenceItem(
                    key="future_forecast_summaries",
                    label="未来多日预测",
                    value=[item.model_dump() for item in context.future_forecast_summaries],
                    source="future_forecast_summaries",
                    priority_score=83 if intent in {AgentIntent.FORECAST, AgentIntent.RISK, AgentIntent.ADVICE} else 69,
                    summary=future_forecast_summary,
                )
            )

        future_classification_summary = self._build_classification_timeline_summary(
            context.future_classification_results,
            prefix="未来分类结果",
        )
        if future_classification_summary:
            evidence.append(
                EvidenceItem(
                    key="future_classification_results",
                    label="未来分类判断",
                    value=[item.model_dump() for item in context.future_classification_results],
                    source="future_classification_results",
                    priority_score=80 if intent in {AgentIntent.FORECAST, AgentIntent.ADVICE, AgentIntent.OVERVIEW} else 66,
                    summary=future_classification_summary,
                )
            )

        self._append_if_present(
            evidence,
            key="avg_active_appliance_count",
            label="平均活跃电器数",
            value=context.recent_history_summary.avg_active_appliance_count,
            source="recent_history_summary",
            priority_score=60,
            summary=self._build_numeric_summary(
                "平均活跃电器数",
                context.recent_history_summary.avg_active_appliance_count,
                "个",
            ),
        )
        self._append_if_present(
            evidence,
            key="avg_burst_event_count",
            label="平均突发事件数",
            value=context.recent_history_summary.avg_burst_event_count,
            source="recent_history_summary",
            priority_score=60,
            summary=self._build_numeric_summary(
                "平均突发事件数",
                context.recent_history_summary.avg_burst_event_count,
                "",
            ),
        )

        if memory and memory.active_goal:
            evidence.append(
                EvidenceItem(
                    key="active_goal",
                    label="当前关注目标",
                    value=memory.active_goal,
                    source="session_memory",
                    priority_score=55,
                    summary=f"当前会话的主要关注点是 {memory.active_goal}。",
                )
            )

        return sorted(evidence, key=lambda item: item.priority_score, reverse=True)

    def identify_missing_information(
        self,
        context: AgentContext,
        intent: AgentIntent,
        question: str,
    ) -> list[MissingInformationItem]:
        missing: list[MissingInformationItem] = []

        if intent in {AgentIntent.CLASSIFICATION, AgentIntent.ADVICE, AgentIntent.OVERVIEW}:
            if not context.classification_result.predicted_label:
                missing.append(
                    MissingInformationItem(
                        key="classification_result",
                        question="请先提供当天分类结果，或补充当天的用电模式判断。",
                        reason="缺少分类结果，无法稳定判断当前日类型。",
                    )
                )

        if intent in {AgentIntent.FORECAST, AgentIntent.RISK, AgentIntent.ADVICE, AgentIntent.OVERVIEW}:
            if (
                not context.forecast_summary.peak_period
                and context.forecast_summary.predicted_peak_load_w is None
                and context.forecast_summary.predicted_total_kwh is None
                and not context.forecast_summary.risk_flags
            ):
                missing.append(
                    MissingInformationItem(
                        key="forecast_summary",
                        question="请补充未来一天的预测摘要，例如高负荷时段或风险标签。",
                        reason="缺少预测结果，无法给出稳定的次日前瞻判断。",
                    )
                )

        if intent == AgentIntent.ADVICE and not context.rule_advices:
            missing.append(
                MissingInformationItem(
                    key="rule_advices",
                    question="如果你有规则引擎生成的建议，请一并提供，这样建议排序会更稳。",
                    reason="缺少规则建议时，只能根据分类和预测做启发式建议。",
                )
            )

        if "舒适" in question and not context.user_preferences.comfort_priority:
            missing.append(
                MissingInformationItem(
                    key="comfort_priority",
                    question="你更偏向节能优先，还是舒适度优先？",
                    reason="涉及舒适度取舍，需要用户偏好才能更准确排序建议。",
                )
            )

        deduplicated: list[MissingInformationItem] = []
        seen_keys: set[str] = set()
        for item in missing:
            if item.key not in seen_keys:
                deduplicated.append(item)
                seen_keys.add(item.key)
        return deduplicated

    def score_confidence(
        self,
        context: AgentContext,
        evidence: list[EvidenceItem],
        missing_information: list[MissingInformationItem],
    ) -> ConfidenceLevel:
        if missing_information:
            if len(evidence) >= 4 and context.classification_result.confidence and context.classification_result.confidence >= 0.75:
                return ConfidenceLevel.MEDIUM
            return ConfidenceLevel.LOW

        if len(evidence) >= 5 and (
            (context.classification_result.confidence is not None and context.classification_result.confidence >= 0.75)
            or context.forecast_summary.predicted_peak_load_w is not None
        ):
            return ConfidenceLevel.HIGH

        return ConfidenceLevel.MEDIUM

    def _append_if_present(
        self,
        target: list[EvidenceItem],
        *,
        key: str,
        label: str,
        value: Any,
        source: str,
        priority_score: int,
        summary: str,
    ) -> None:
        if value is None or summary == "":
            return
        target.append(
            EvidenceItem(
                key=key,
                label=label,
                value=value,
                source=source,
                priority_score=priority_score,
                summary=summary,
            )
        )

    def _build_numeric_summary(self, label: str, value: float | None, unit: str) -> str:
        if value is None:
            return ""
        unit_suffix = f" {unit}" if unit else ""
        return f"{label}约为 {float(value):.2f}{unit_suffix}。"

    def _build_ratio_summary(self, label: str, value: float | None) -> str:
        if value is None:
            return ""
        ratio = float(value)
        if ratio <= 1:
            ratio *= 100
        return f"{label}约为 {ratio:.2f}%。"

    def _build_classification_timeline_summary(
        self,
        items: list[Any],
        *,
        prefix: str,
    ) -> str:
        if not items:
            return ""
        label_counts: dict[str, int] = {}
        for item in items:
            predicted_label = getattr(item, "predicted_label", None)
            if not predicted_label:
                continue
            label_counts[predicted_label] = label_counts.get(predicted_label, 0) + 1
        if not label_counts:
            return ""
        ordered = sorted(label_counts.items(), key=lambda pair: (-pair[1], pair[0]))
        summary = "，".join(
            f"{label_text(label)} {count} 天" for label, count in ordered[:3]
        )
        return f"{prefix}显示主要分布为：{summary}。"

    def _build_future_forecast_summary(self, items: list[Any]) -> str:
        if not items:
            return ""
        valid_items = [
            item
            for item in items
            if item.predicted_peak_load_w is not None
        ]
        if not valid_items:
            return ""
        peak_item = max(valid_items, key=lambda item: float(item.predicted_peak_load_w or 0.0))
        peak_day = (
            getattr(peak_item, "day_offset", None)
            or getattr(peak_item, "date", None)
            or "未知日期"
        )
        return (
            f"未来 {len(items)} 天预测已纳入分析，其中峰值最高的是 {peak_day}，"
            f"预测峰值负荷约 {float(peak_item.predicted_peak_load_w):.2f} W。"
        )
