"""智能体问答服务。"""

from __future__ import annotations

import json
from typing import Any

from app.agent.advice_planner import AdvicePlanner
from app.agent.evidence_builder import EvidenceBuilder
from app.agent.intent_router import IntentRouter
from app.agent.memory import ShortTermMemoryManager
from app.agent.report_builder import ReportBuilder
from app.agent.response_renderer import ResponseRenderer
from app.agent.state import (
    AgentOutput,
    CitationItem,
    MissingInformationItem,
    ReportSection,
    ReportSummaryOutput,
    REPORT_SECTION_ORDER,
)
from app.agent.workflow import AgentWorkflow, PreparedAskResult, PreparedReportResult, ReportWorkflow
from app.config import Settings
from app.contracts import AgentAskRequest, AgentReportSummaryRequest


class AgentService:
    """Agent 服务门面。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.memory_manager = ShortTermMemoryManager()
        self.agent_workflow = AgentWorkflow(
            memory_manager=self.memory_manager,
            intent_router=IntentRouter(),
            evidence_builder=EvidenceBuilder(),
            advice_planner=AdvicePlanner(),
            response_renderer=ResponseRenderer(),
        )
        self.report_workflow = ReportWorkflow(
            evidence_builder=EvidenceBuilder(),
            advice_planner=AdvicePlanner(),
            report_builder=ReportBuilder(),
        )

    def health(self) -> dict[str, Any]:
        return {
            "status": "up",
            "service": "python-robyn-backend",
            "agent_ready": self._can_use_langchain(),
            "llm_configured": bool(
                self.settings.llm_base_url and self.settings.llm_api_key and self.settings.llm_model
            ),
            "memory_session_count": self.memory_manager.session_count(),
        }

    def ask(self, request: AgentAskRequest) -> dict[str, Any]:
        prepared = self.agent_workflow.prepare(request)
        fallback_response = self._ask_result_to_payload(prepared.fallback_output)

        if not self._can_use_langchain():
            self.agent_workflow.commit(
                session_id=request.session_id,
                intent=prepared.intent,
                question=request.question,
                actions=prepared.fallback_output.actions,
                missing_information=prepared.fallback_output.missing_information,
                context=prepared.context,
            )
            return {
                **fallback_response,
                "intent": prepared.intent.value,
                "degraded": True,
                "error_reason": self._build_unavailable_reason(),
            }

        try:
            result = self._run_langchain(prepared, request)
            self.agent_workflow.commit(
                session_id=request.session_id,
                intent=prepared.intent,
                question=request.question,
                actions=result["actions"],
                missing_information=prepared.fallback_output.missing_information,
                context=prepared.context,
            )
            return {
                **result,
                "intent": prepared.intent.value,
                "degraded": False,
                "error_reason": None,
            }
        except Exception:
            self.agent_workflow.commit(
                session_id=request.session_id,
                intent=prepared.intent,
                question=request.question,
                actions=prepared.fallback_output.actions,
                missing_information=prepared.fallback_output.missing_information,
                context=prepared.context,
            )
            return {
                **fallback_response,
                "intent": prepared.intent.value,
                "degraded": True,
                "error_reason": "LLM_REQUEST_FAILED",
            }

    def summarize_report(self, request: AgentReportSummaryRequest) -> dict[str, Any]:
        prepared = self.report_workflow.prepare(request)
        fallback = prepared.fallback_output.model_dump()

        if not self._can_use_langchain():
            return {
                **fallback,
                "degraded": True,
                "error_reason": self._build_unavailable_reason(),
            }

        try:
            result = self._run_langchain_report_summary(prepared)
            return {
                **result,
                "degraded": False,
                "error_reason": None,
            }
        except Exception:
            return {
                **fallback,
                "degraded": True,
                "error_reason": "LLM_REQUEST_FAILED",
            }

    def _can_use_langchain(self) -> bool:
        try:
            import langchain_openai  # noqa: F401
            import langchain_core  # noqa: F401
        except ModuleNotFoundError:
            return False

        return bool(
            self.settings.llm_base_url and self.settings.llm_api_key and self.settings.llm_model
        )

    def _build_unavailable_reason(self) -> str:
        try:
            import langchain_openai  # noqa: F401
            import langchain_core  # noqa: F401
        except ModuleNotFoundError:
            return "LANGCHAIN_UNAVAILABLE"

        if not self.settings.llm_base_url:
            return "LLM_BASE_URL_MISSING"
        if not self.settings.llm_api_key:
            return "LLM_API_KEY_MISSING"
        if not self.settings.llm_model:
            return "LLM_MODEL_MISSING"
        return "LLM_NOT_CONFIGURED"

    def _run_langchain(
        self,
        prepared: PreparedAskResult,
        request: AgentAskRequest,
    ) -> dict[str, Any]:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(
            base_url=self.settings.llm_base_url,
            api_key=self.settings.llm_api_key,
            model=self.settings.llm_model,
            temperature=self.settings.llm_temperature,
            timeout=self.settings.llm_timeout_seconds,
        )
        structured_model = model.with_structured_output(
            AgentOutput,
            method="function_calling",
            strict=True,
        )

        history_messages: list[Any] = []
        for item in request.history:
            if item.role == "assistant":
                history_messages.append(AIMessage(content=item.content))
            elif item.role == "system":
                history_messages.append(SystemMessage(content=item.content))
            else:
                history_messages.append(HumanMessage(content=item.content))

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "你是居民用电分析与节能建议助手。"
                        "你只能基于工作流提供的结构化证据、候选建议和会话记忆回答。"
                        "禁止编造上下文中不存在的指标、设备结论或节省金额。"
                        "如果证据不足，必须在 missing_information 中明确指出。"
                        "输出必须包含 answer、citations、actions、missing_information、confidence_level。"
                        "actions 应优先复用候选建议中的动作描述。"
                    ),
                ),
                ("placeholder", "{history_messages}"),
                (
                    "human",
                    (
                        "请根据下面的工作流结果生成回答。\n"
                        "问题：{question}\n"
                        "意图：{intent}\n"
                        "结构化上下文：{context_json}\n"
                        "会话记忆：{memory_json}\n"
                        "证据列表：{evidence_json}\n"
                        "候选建议：{advice_candidates_json}\n"
                        "信息缺口：{missing_information_json}\n"
                        "降级参考：{fallback_output_json}\n"
                        "要求：\n"
                        "1. 用中文回答，优先解释证据，再给动作。\n"
                        "2. citations 只允许引用证据列表中的 key。\n"
                        "3. 如果无法稳定判断，要明确写出限制条件。\n"
                        "4. 不要添加任何设备级确定性归因。\n"
                    ),
                ),
            ]
        )

        chain = prompt | structured_model
        parsed = chain.invoke(
            {
                "history_messages": history_messages,
                "question": request.question,
                "intent": prepared.intent.value,
                "context_json": json.dumps(prepared.context.model_dump(), ensure_ascii=False),
                "memory_json": json.dumps(prepared.memory.to_prompt_payload(), ensure_ascii=False),
                "evidence_json": json.dumps([item.model_dump() for item in prepared.evidence], ensure_ascii=False),
                "advice_candidates_json": json.dumps(
                    [item.model_dump() for item in prepared.advice_candidates],
                    ensure_ascii=False,
                ),
                "missing_information_json": json.dumps(
                    [item.model_dump() for item in prepared.missing_information],
                    ensure_ascii=False,
                ),
                "fallback_output_json": json.dumps(
                    prepared.fallback_output.model_dump(),
                    ensure_ascii=False,
                ),
            }
        )

        validated_output = self._normalize_output(parsed, prepared)
        return self._ask_result_to_payload(validated_output)

    def _run_langchain_report_summary(
        self,
        prepared: PreparedReportResult,
    ) -> dict[str, Any]:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(
            base_url=self.settings.llm_base_url,
            api_key=self.settings.llm_api_key,
            model=self.settings.llm_model,
            temperature=self.settings.llm_temperature,
            timeout=self.settings.llm_report_timeout_seconds,
        )
        structured_model = model.with_structured_output(
            ReportSummaryOutput,
            method="function_calling",
            strict=True,
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "你是居民用电分析报告撰写助手。"
                        "你只能根据提供的结构化证据、候选建议和降级草稿整理报告。"
                        "禁止扩写出不存在的数据或设备级确定性结论。"
                        "输出必须包含 title、overview、sections、recommendations。"
                    ),
                ),
                (
                    "human",
                    (
                        "请根据下面的工作流结果整理报告摘要。\n"
                        "结构化上下文：{context_json}\n"
                        "证据列表：{evidence_json}\n"
                        "候选建议：{advice_candidates_json}\n"
                        "降级草稿：{fallback_summary_json}\n"
                        "要求：\n"
                        "1. title 用中文，不超过 24 个字。\n"
                        "2. sections 必须且只能输出以下四个章节：总体概览、行为判断、预测风险、附注。\n"
                        "3. recommendations 输出 2 到 5 条可执行建议。\n"
                        "4. 所有内容都必须可回溯到上下文或证据。\n"
                    ),
                ),
            ]
        )

        chain = prompt | structured_model
        parsed = chain.invoke(
            {
                "context_json": json.dumps(prepared.context.model_dump(), ensure_ascii=False),
                "evidence_json": json.dumps([item.model_dump() for item in prepared.evidence], ensure_ascii=False),
                "advice_candidates_json": json.dumps(
                    [item.model_dump() for item in prepared.advice_candidates],
                    ensure_ascii=False,
                ),
                "fallback_summary_json": json.dumps(
                    prepared.fallback_output.model_dump(),
                    ensure_ascii=False,
                ),
            }
        )

        normalized = self._normalize_report_summary_output(parsed, prepared.fallback_output)
        return normalized.model_dump()

    def _normalize_output(
        self,
        parsed: AgentOutput | dict[str, Any],
        prepared: PreparedAskResult,
    ) -> AgentOutput:
        output = parsed if isinstance(parsed, AgentOutput) else AgentOutput.model_validate(parsed)

        if not output.citations:
            output = output.model_copy(update={"citations": prepared.fallback_output.citations})
        if not output.actions:
            output = output.model_copy(update={"actions": prepared.fallback_output.actions})
        if not output.missing_information:
            output = output.model_copy(
                update={"missing_information": prepared.fallback_output.missing_information}
            )

        sanitized_citations = self._sanitize_citations(output.citations, prepared)
        sanitized_actions = self._sanitize_actions(output.actions, prepared.fallback_output.actions)
        sanitized_missing_information = self._sanitize_missing_information(
            output.missing_information,
            prepared.fallback_output.missing_information,
        )

        return output.model_copy(
            update={
                "citations": sanitized_citations,
                "actions": sanitized_actions,
                "missing_information": sanitized_missing_information,
            }
        )

    def _normalize_report_summary_output(
        self,
        parsed: ReportSummaryOutput | dict[str, Any],
        fallback_summary: ReportSummaryOutput,
    ) -> ReportSummaryOutput:
        output = parsed if isinstance(parsed, ReportSummaryOutput) else ReportSummaryOutput.model_validate(parsed)

        title = str(output.title).strip() or fallback_summary.title
        overview = str(output.overview).strip() or fallback_summary.overview
        sections = self._ordered_report_sections(output.sections, overview)
        if not sections:
            sections = self._ordered_report_sections(fallback_summary.sections, fallback_summary.overview)

        recommendations: list[str] = []
        for item in output.recommendations:
            cleaned = str(item).strip()
            if cleaned and cleaned not in recommendations:
                recommendations.append(cleaned)
        if not recommendations:
            recommendations = fallback_summary.recommendations[:]

        return ReportSummaryOutput(
            title=title,
            overview=overview,
            sections=sections,
            recommendations=recommendations[:5],
        )

    def _sanitize_citations(
        self,
        citations: list[CitationItem],
        prepared: PreparedAskResult,
    ) -> list[CitationItem]:
        allowed_mapping = {item.key: item.to_citation() for item in prepared.evidence}
        sanitized: list[CitationItem] = []
        for citation in citations:
            if citation.key in allowed_mapping:
                sanitized.append(allowed_mapping[citation.key])

        if not sanitized:
            sanitized = prepared.fallback_output.citations[:]
        return sanitized[:6]

    def _sanitize_actions(
        self,
        actions: list[str],
        fallback_actions: list[str],
    ) -> list[str]:
        sanitized: list[str] = []
        for action in actions:
            cleaned = str(action).strip()
            if cleaned and cleaned not in sanitized:
                sanitized.append(cleaned)
        if not sanitized:
            sanitized = fallback_actions[:]
        return sanitized[:5]

    def _sanitize_missing_information(
        self,
        missing_information: list[MissingInformationItem],
        fallback_missing_information: list[MissingInformationItem],
    ) -> list[MissingInformationItem]:
        allowed_mapping = {item.key: item for item in fallback_missing_information}
        sanitized: list[MissingInformationItem] = []
        for item in missing_information:
            if item.key in allowed_mapping:
                sanitized.append(allowed_mapping[item.key])
        if not sanitized:
            sanitized = fallback_missing_information[:]
        return sanitized[:3]

    def _ordered_report_sections(
        self,
        sections: list[ReportSection],
        overview: str,
    ) -> list[ReportSection]:
        mapping = {item.title: item.body for item in sections if item.title in REPORT_SECTION_ORDER and item.body.strip()}
        if overview.strip():
            mapping["总体概览"] = overview.strip()
        ordered_sections: list[ReportSection] = []
        for title in REPORT_SECTION_ORDER:
            body = mapping.get(title, "").strip()
            if body:
                ordered_sections.append(ReportSection(title=title, body=body))
        return ordered_sections

    def _ask_result_to_payload(self, output: AgentOutput) -> dict[str, Any]:
        return {
            "answer": output.answer,
            "citations": [item.model_dump() for item in output.citations],
            "actions": output.actions,
            "missing_information": [item.model_dump() for item in output.missing_information],
            "confidence_level": output.confidence_level.value,
        }
