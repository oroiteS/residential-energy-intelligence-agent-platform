"""结构化结论到自然语言回答的渲染。"""

from __future__ import annotations

from app.agent.state import (
    AdviceCandidate,
    AgentContext,
    AgentIntent,
    AgentOutput,
    ConfidenceLevel,
    EvidenceItem,
    MissingInformationItem,
    SessionMemorySnapshot,
    label_text,
)


class ResponseRenderer:
    """生成降级回答与默认文案。"""

    def render(
        self,
        *,
        question: str,
        intent: AgentIntent,
        context: AgentContext,
        evidence: list[EvidenceItem],
        advice_candidates: list[AdviceCandidate],
        missing_information: list[MissingInformationItem],
        confidence_level: ConfidenceLevel,
        memory: SessionMemorySnapshot | None,
    ) -> AgentOutput:
        answer = self._build_answer(
            question=question,
            intent=intent,
            context=context,
            evidence=evidence,
            advice_candidates=advice_candidates,
            missing_information=missing_information,
            confidence_level=confidence_level,
            memory=memory,
        )
        citations = [item.to_citation() for item in evidence[:6]]
        actions = [item.action for item in advice_candidates[:5]]
        return AgentOutput(
            answer=answer,
            citations=citations,
            actions=actions,
            missing_information=missing_information,
            confidence_level=confidence_level,
        )

    def _build_answer(
        self,
        *,
        question: str,
        intent: AgentIntent,
        context: AgentContext,
        evidence: list[EvidenceItem],
        advice_candidates: list[AdviceCandidate],
        missing_information: list[MissingInformationItem],
        confidence_level: ConfidenceLevel,
        memory: SessionMemorySnapshot | None,
    ) -> str:
        sentences: list[str] = ["当前回答基于已有分类结果、预测摘要和规则建议自动整理。"]

        if memory and memory.last_intent and memory.recent_actions:
            sentences.append(f"你上一轮主要在关注 {memory.last_intent.value}，上次给出的动作中优先项是：{memory.recent_actions[0]}。")

        if intent in {AgentIntent.CLASSIFICATION, AgentIntent.OVERVIEW, AgentIntent.FOLLOW_UP}:
            predicted_label = context.classification_result.predicted_label
            if predicted_label:
                sentences.append(f"当前日类型判断为 {label_text(predicted_label)}。")

        if intent in {AgentIntent.FORECAST, AgentIntent.RISK, AgentIntent.OVERVIEW, AgentIntent.ADVICE, AgentIntent.FOLLOW_UP}:
            if context.forecast_summary.peak_period:
                sentences.append(f"预测高负荷时段集中在 {context.forecast_summary.peak_period}。")
            if context.forecast_summary.risk_flags:
                sentences.append(f"当前预测风险标签包括 {context.forecast_summary.risk_flags}。")

        if context.analysis_summary.peak_ratio is not None:
            ratio = float(context.analysis_summary.peak_ratio)
            if ratio <= 1:
                ratio *= 100
            sentences.append(f"峰时占比约为 {ratio:.2f}%。")

        if advice_candidates:
            top_candidate = advice_candidates[0]
            sentences.append(f"在当前问题下，最值得优先执行的动作是：{top_candidate.action}")
            sentences.append(f"原因是：{top_candidate.rationale}")

        if missing_information:
            sentences.append("目前仍有部分信息不足，以下判断应视为保守建议。")

        if confidence_level == ConfidenceLevel.LOW:
            sentences.append("当前回答置信度较低，建议先补充分类结果或预测摘要。")
        elif confidence_level == ConfidenceLevel.MEDIUM:
            sentences.append("当前回答置信度中等，适合用于初步判断和动作排序。")
        else:
            sentences.append("当前回答置信度较高，可直接作为本轮优先建议。")

        return "".join(sentences)

