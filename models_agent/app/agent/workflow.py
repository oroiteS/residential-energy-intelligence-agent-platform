"""Agent 工作流。"""

from __future__ import annotations

from dataclasses import dataclass

from app.agent.advice_planner import AdvicePlanner
from app.agent.evidence_builder import EvidenceBuilder
from app.agent.intent_router import IntentRouter
from app.agent.memory import ShortTermMemoryManager
from app.agent.report_builder import ReportBuilder
from app.agent.response_renderer import ResponseRenderer
from app.agent.state import (
    AdviceCandidate,
    AgentContext,
    AgentIntent,
    AgentOutput,
    ConfidenceLevel,
    EvidenceItem,
    MissingInformationItem,
    ReportSummaryOutput,
    SessionMemorySnapshot,
)
from app.contracts import AgentAskRequest, AgentReportSummaryRequest


@dataclass(slots=True)
class PreparedAskResult:
    """问答工作流的中间结果。"""

    context: AgentContext
    memory: SessionMemorySnapshot
    intent: AgentIntent
    evidence: list[EvidenceItem]
    advice_candidates: list[AdviceCandidate]
    missing_information: list[MissingInformationItem]
    confidence_level: ConfidenceLevel
    fallback_output: AgentOutput


@dataclass(slots=True)
class PreparedReportResult:
    """报告工作流的中间结果。"""

    context: AgentContext
    evidence: list[EvidenceItem]
    advice_candidates: list[AdviceCandidate]
    confidence_level: ConfidenceLevel
    fallback_output: ReportSummaryOutput


class AgentWorkflow:
    """问答主工作流。"""

    def __init__(
        self,
        *,
        memory_manager: ShortTermMemoryManager,
        intent_router: IntentRouter,
        evidence_builder: EvidenceBuilder,
        advice_planner: AdvicePlanner,
        response_renderer: ResponseRenderer,
    ) -> None:
        self.memory_manager = memory_manager
        self.intent_router = intent_router
        self.evidence_builder = evidence_builder
        self.advice_planner = advice_planner
        self.response_renderer = response_renderer

    def prepare(self, request: AgentAskRequest) -> PreparedAskResult:
        context = AgentContext.from_payload(request.context)
        memory = self.memory_manager.get(request.session_id, request.history)
        intent = self.intent_router.route(request.question, request.history, context, memory)
        evidence = self.evidence_builder.build(context, intent, request.question, memory)
        missing_information = self.evidence_builder.identify_missing_information(
            context,
            intent,
            request.question,
        )
        advice_candidates = self.advice_planner.plan(
            context,
            intent,
            request.question,
            evidence,
            memory,
        )
        confidence_level = self.evidence_builder.score_confidence(
            context,
            evidence,
            missing_information,
        )
        fallback_output = self.response_renderer.render(
            question=request.question,
            intent=intent,
            context=context,
            evidence=evidence,
            advice_candidates=advice_candidates,
            missing_information=missing_information,
            confidence_level=confidence_level,
            memory=memory,
        )

        return PreparedAskResult(
            context=context,
            memory=memory,
            intent=intent,
            evidence=evidence,
            advice_candidates=advice_candidates,
            missing_information=missing_information,
            confidence_level=confidence_level,
            fallback_output=fallback_output,
        )

    def commit(
        self,
        *,
        session_id: int,
        intent: AgentIntent,
        question: str,
        actions: list[str],
        missing_information: list[MissingInformationItem],
        context: AgentContext,
    ) -> SessionMemorySnapshot:
        active_goal = (
            context.user_preferences.objective
            or context.user_preferences.focus
            or context.conversation_state.active_goal
        )
        return self.memory_manager.update(
            session_id,
            intent=intent,
            question=question,
            actions=actions,
            missing_information=missing_information,
            active_goal=active_goal,
        )


class ReportWorkflow:
    """报告摘要工作流。"""

    def __init__(
        self,
        *,
        evidence_builder: EvidenceBuilder,
        advice_planner: AdvicePlanner,
        report_builder: ReportBuilder,
    ) -> None:
        self.evidence_builder = evidence_builder
        self.advice_planner = advice_planner
        self.report_builder = report_builder

    def prepare(self, request: AgentReportSummaryRequest) -> PreparedReportResult:
        context = AgentContext.from_payload(request.context)
        intent = AgentIntent.OVERVIEW
        evidence = self.evidence_builder.build(context, intent, "生成报告摘要", None)
        missing_information = self.evidence_builder.identify_missing_information(
            context,
            intent,
            "生成报告摘要",
        )
        confidence_level = self.evidence_builder.score_confidence(
            context,
            evidence,
            missing_information,
        )
        advice_candidates = self.advice_planner.plan(
            context,
            AgentIntent.ADVICE,
            "生成报告摘要",
            evidence,
            None,
        )
        fallback_output = self.report_builder.build(
            context=context,
            evidence=evidence,
            advice_candidates=advice_candidates,
            confidence_level=confidence_level,
        )
        return PreparedReportResult(
            context=context,
            evidence=evidence,
            advice_candidates=advice_candidates,
            confidence_level=confidence_level,
            fallback_output=fallback_output,
        )
