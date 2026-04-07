"""智能体问答服务。"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from app.config import Settings
from app.contracts import AgentAskRequest, AgentReportSummaryRequest


class CitationItem(BaseModel):
    """结构化引用项。"""

    key: str = Field(min_length=1)
    label: str = Field(min_length=1)
    value: Any


class AgentOutput(BaseModel):
    """智能体结构化输出。"""

    answer: str = Field(min_length=1)
    citations: list[CitationItem] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)


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


REPORT_SECTION_ORDER = ["总体概览", "行为判断", "预测风险", "附注"]


class AgentService:
    """LangChain 智能体封装。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def health(self) -> dict[str, Any]:
        return {
            "status": "up",
            "service": "python-robyn-backend",
            "agent_ready": self._can_use_langchain(),
            "llm_configured": bool(
                self.settings.llm_base_url and self.settings.llm_api_key and self.settings.llm_model
            ),
        }

    def ask(self, request: AgentAskRequest) -> dict[str, Any]:
        citations = self._build_citations(request.context)
        fallback_answer = self._build_fallback_answer(request.question, request.context)
        fallback_actions = self._build_actions(request.context)

        if not self._can_use_langchain():
            return {
                "answer": fallback_answer,
                "citations": citations,
                "actions": fallback_actions,
                "degraded": True,
                "error_reason": self._build_unavailable_reason(),
            }

        try:
            result = self._run_langchain(request, citations, fallback_actions)
            result["degraded"] = False
            result["error_reason"] = None
            return result
        except Exception:
            return {
                "answer": fallback_answer,
                "citations": citations,
                "actions": fallback_actions,
                "degraded": True,
                "error_reason": "LLM_REQUEST_FAILED",
            }

    def summarize_report(self, request: AgentReportSummaryRequest) -> dict[str, Any]:
        fallback = self._build_fallback_report_summary(request.context)

        if not self._can_use_langchain():
            return {
                **fallback,
                "degraded": True,
                "error_reason": self._build_unavailable_reason(),
            }

        try:
            result = self._run_langchain_report_summary(request, fallback)
            result["degraded"] = False
            result["error_reason"] = None
            return result
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
        request: AgentAskRequest,
        citations: list[dict[str, Any]],
        fallback_actions: list[str],
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

        messages: list[Any] = []
        for item in request.history:
            if item.role == "assistant":
                messages.append(AIMessage(content=item.content))
            elif item.role == "system":
                messages.append(SystemMessage(content=item.content))
            else:
                messages.append(HumanMessage(content=item.content))

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "你是居民用电分析与节能建议助手。"
                        "只能根据提供的上下文回答，禁止编造不存在的数据。"
                        "如果上下文无法支持某个判断，必须明确说明无法判断。"
                        "输出必须包含 answer、citations、actions。"
                        "citations 只允许引用上下文中已经出现的指标、分类结果、预测结果或规则建议。"
                        "actions 必须是简洁、可执行的短句。"
                    ),
                ),
                ("placeholder", "{history_messages}"),
                (
                    "human",
                    (
                        "请根据下面的问题和上下文生成回答。\n"
                        "问题：{question}\n"
                        "上下文：{context_json}\n"
                        "候选引用：{fallback_citations_json}\n"
                        "候选动作：{fallback_actions_json}\n"
                        "要求：\n"
                        "1. answer 用中文回答，简洁但具体。\n"
                        "2. citations 必须优先复用候选引用，且不要新增上下文里没有的指标。\n"
                        "3. actions 输出 1 到 5 条。\n"
                    ),
                ),
            ]
        )

        chain = prompt | structured_model
        parsed = chain.invoke(
            {
                "history_messages": messages,
                "question": request.question,
                "context_json": json.dumps(request.context, ensure_ascii=False),
                "fallback_citations_json": json.dumps(citations, ensure_ascii=False),
                "fallback_actions_json": json.dumps(fallback_actions, ensure_ascii=False),
            }
        )

        validated_output = self._normalize_output(parsed, citations, fallback_actions)

        return {
            "answer": validated_output.answer,
            "citations": [item.model_dump() for item in validated_output.citations],
            "actions": validated_output.actions,
        }

    def _run_langchain_report_summary(
        self,
        request: AgentReportSummaryRequest,
        fallback_summary: dict[str, Any],
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
                        "只能根据提供的上下文整理报告，禁止编造不存在的数据。"
                        "报告语气要专业、简洁、可直接用于 PDF 导出。"
                        "输出必须包含 title、overview、sections、recommendations。"
                    ),
                ),
                (
                    "human",
                    (
                        "请根据下面的上下文整理一份居民用电分析报告摘要。\n"
                        "上下文：{context_json}\n"
                        "降级参考：{fallback_summary_json}\n"
                        "要求：\n"
                        "1. title 用中文，不超过 24 个字。\n"
                        "2. overview 用 2 到 4 句概括整体情况。\n"
                        "3. sections 必须且只能输出以下四个章节：总体概览、行为判断、预测风险、附注。\n"
                        "4. 每个章节都要有 title 和 body，title 必须与要求完全一致。\n"
                        "5. recommendations 输出 2 到 5 条，必须是可执行建议。\n"
                        "6. 所有内容都必须能在上下文中找到依据。\n"
                    ),
                ),
            ]
        )

        chain = prompt | structured_model
        parsed = chain.invoke(
            {
                "context_json": json.dumps(request.context, ensure_ascii=False),
                "fallback_summary_json": json.dumps(fallback_summary, ensure_ascii=False),
            }
        )

        normalized = self._normalize_report_summary_output(parsed, fallback_summary)
        return normalized.model_dump()

    def _normalize_output(
        self,
        parsed: AgentOutput | dict[str, Any],
        fallback_citations: list[dict[str, Any]],
        fallback_actions: list[str],
    ) -> AgentOutput:
        if isinstance(parsed, AgentOutput):
            output = parsed
        else:
            output = AgentOutput.model_validate(parsed)

        if not output.citations:
            output = output.model_copy(update={"citations": [CitationItem.model_validate(item) for item in fallback_citations]})
        if not output.actions:
            output = output.model_copy(update={"actions": fallback_actions[:5]})

        sanitized_citations = self._sanitize_citations(output.citations, fallback_citations)
        sanitized_actions = self._sanitize_actions(output.actions, fallback_actions)

        return output.model_copy(update={"citations": sanitized_citations, "actions": sanitized_actions})

    def _normalize_report_summary_output(
        self,
        parsed: ReportSummaryOutput | dict[str, Any],
        fallback_summary: dict[str, Any],
    ) -> ReportSummaryOutput:
        if isinstance(parsed, ReportSummaryOutput):
            output = parsed
        else:
            output = ReportSummaryOutput.model_validate(parsed)

        title = str(output.title).strip() or str(fallback_summary["title"]).strip()
        overview = str(output.overview).strip() or str(fallback_summary["overview"]).strip()

        fallback_sections = self._ordered_report_sections(
            fallback_summary.get("sections", []),
            fallback_summary["overview"],
        )
        sanitized_sections = self._ordered_report_sections(
            [{"title": item.title, "body": item.body} for item in output.sections],
            overview,
        )
        if not sanitized_sections:
            sanitized_sections = fallback_sections

        fallback_recommendations = [
            str(item).strip()
            for item in fallback_summary.get("recommendations", [])
            if str(item).strip()
        ]
        sanitized_recommendations: list[str] = []
        for item in output.recommendations:
            value = str(item).strip()
            if value and value not in sanitized_recommendations:
                sanitized_recommendations.append(value)
        if not sanitized_recommendations:
            sanitized_recommendations = fallback_recommendations

        return ReportSummaryOutput(
            title=title,
            overview=overview,
            sections=sanitized_sections,
            recommendations=sanitized_recommendations[:5],
        )

    def _ordered_report_sections(
        self,
        raw_sections: list[dict[str, Any]],
        overview: str,
    ) -> list[ReportSection]:
        sections_by_title: dict[str, str] = {}
        for item in raw_sections:
            title = str(item.get("title", "")).strip()
            body = str(item.get("body", "")).strip()
            if title in REPORT_SECTION_ORDER and body:
                sections_by_title[title] = body

        if overview.strip():
            sections_by_title["总体概览"] = overview.strip()

        ordered_sections: list[ReportSection] = []
        for title in REPORT_SECTION_ORDER:
            body = sections_by_title.get(title, "").strip()
            if body:
                ordered_sections.append(ReportSection(title=title, body=body))
        return ordered_sections

    def _sanitize_citations(
        self,
        citations: list[CitationItem],
        fallback_citations: list[dict[str, Any]],
    ) -> list[CitationItem]:
        allowed_keys = {str(item.get("key", "")) for item in fallback_citations}
        fallback_mapping = {
            str(item.get("key", "")): CitationItem.model_validate(item)
            for item in fallback_citations
            if item.get("key")
        }

        sanitized: list[CitationItem] = []
        for citation in citations:
            if citation.key in allowed_keys:
                sanitized.append(citation)
            elif citation.key in fallback_mapping:
                sanitized.append(fallback_mapping[citation.key])

        if not sanitized:
            sanitized = list(fallback_mapping.values())
        return sanitized[:6]

    def _sanitize_actions(self, actions: list[str], fallback_actions: list[str]) -> list[str]:
        sanitized: list[str] = []
        for action in actions:
            cleaned = str(action).strip()
            if cleaned and cleaned not in sanitized:
                sanitized.append(cleaned)

        if not sanitized:
            sanitized = fallback_actions[:]
        return sanitized[:5]

    def _build_citations(self, context: dict[str, Any]) -> list[dict[str, Any]]:
        citations: list[dict[str, Any]] = []
        analysis_summary = context.get("analysis_summary", {}) or {}
        classification_result = context.get("classification_result", {}) or {}
        forecast_summary = context.get("forecast_summary", {}) or {}

        if "peak_ratio" in analysis_summary:
            citations.append(
                {"key": "peak_ratio", "label": "峰时占比", "value": analysis_summary["peak_ratio"]}
            )
        if "daily_avg_kwh" in analysis_summary:
            citations.append(
                {"key": "daily_avg_kwh", "label": "日均用电量", "value": analysis_summary["daily_avg_kwh"]}
            )
        if "predicted_label" in classification_result:
            citations.append(
                {
                    "key": "predicted_label",
                    "label": "行为类型",
                    "value": classification_result["predicted_label"],
                }
            )
        if "confidence" in classification_result:
            citations.append(
                {"key": "confidence", "label": "分类置信度", "value": classification_result["confidence"]}
            )
        if "peak_period" in forecast_summary:
            citations.append(
                {"key": "forecast_peak_period", "label": "预测高负荷时段", "value": forecast_summary["peak_period"]}
            )
        if "risk_flags" in forecast_summary:
            citations.append(
                {"key": "risk_flags", "label": "预测风险标签", "value": forecast_summary["risk_flags"]}
            )
        return citations

    def _build_actions(self, context: dict[str, Any]) -> list[str]:
        rule_advices = context.get("rule_advices", []) or []
        actions = []

        for item in rule_advices:
            if isinstance(item, dict):
                action = item.get("action") or item.get("summary")
                if action:
                    actions.append(str(action))

        if not actions:
            classification_result = context.get("classification_result", {}) or {}
            predicted_label = classification_result.get("predicted_label")
            if predicted_label == "day_low_night_high":
                actions.extend(["优先检查晚间持续运行设备", "将热水器改为定时运行"])
            elif predicted_label == "day_high_night_low":
                actions.extend(["将白天高耗电任务错峰执行", "重点排查白天集中启停设备"])
            else:
                actions.extend(["复查峰时段高耗电设备", "优先调整可延后负荷到低负荷时段"])

        deduplicated_actions: list[str] = []
        for action in actions:
            if action not in deduplicated_actions:
                deduplicated_actions.append(action)
        return deduplicated_actions[:5]

    def _build_fallback_answer(self, question: str, context: dict[str, Any]) -> str:
        analysis_summary = context.get("analysis_summary", {}) or {}
        classification_result = context.get("classification_result", {}) or {}
        forecast_summary = context.get("forecast_summary", {}) or {}

        sentence_parts = ["当前智能问答已自动降级，我将基于已有分析结果给出建议。"]

        peak_ratio = analysis_summary.get("peak_ratio")
        if peak_ratio is not None:
            sentence_parts.append(f"峰时占比约为 {float(peak_ratio):.2f}。")

        predicted_label = classification_result.get("predicted_label")
        if predicted_label:
            label_text = {
                "day_high_night_low": "白天高晚上低型",
                "day_low_night_high": "白天低晚上高型",
                "all_day_high": "全天高负载型",
                "all_day_low": "全天低负载型",
            }.get(predicted_label, predicted_label)
            sentence_parts.append(f"当前行为类型判断为 {label_text}。")

        peak_period = forecast_summary.get("peak_period")
        if peak_period:
            sentence_parts.append(f"预测高负荷时段集中在 {peak_period}。")

        if "夜间" in question and predicted_label == "day_low_night_high":
            sentence_parts.append("这说明晚间负荷明显偏高，建议优先排查夜间持续运行设备。")
        elif "明天" in question and peak_period:
            sentence_parts.append("可以提前避开该高负荷时段安排洗衣、热水器或充电等任务。")
        else:
            sentence_parts.append("建议结合规则建议优先处理峰时段和持续负荷较高的设备。")

        return "".join(sentence_parts)

    def _build_fallback_report_summary(self, context: dict[str, Any]) -> dict[str, Any]:
        dataset = context.get("dataset", {}) or {}
        analysis_summary = context.get("analysis_summary", {}) or {}
        classification_result = context.get("classification_result", {}) or {}
        forecast_summary = context.get("forecast_summary", {}) or {}
        recent_history_summary = context.get("recent_history_summary", {}) or {}

        title = "居民用电分析报告"
        dataset_name = str(dataset.get("name", "")).strip()
        if dataset_name:
            title = f"{title} - {dataset_name}"

        overview_parts: list[str] = []
        if "daily_avg_kwh" in analysis_summary:
            overview_parts.append(
                f"日均用电量约为 {float(analysis_summary['daily_avg_kwh']):.2f} kWh。"
            )
        if "peak_ratio" in analysis_summary:
            overview_parts.append(
                f"峰时占比约为 {float(analysis_summary['peak_ratio']) * 100:.2f}%。"
            )
        predicted_label = str(classification_result.get("predicted_label", "")).strip()
        if predicted_label:
            overview_parts.append(f"行为类型判断为 {self._label_text(predicted_label)}。")
        peak_period = str(forecast_summary.get("peak_period", "")).strip()
        if peak_period:
            overview_parts.append(f"预测高负荷时段集中在 {peak_period}。")
        if not overview_parts:
            overview_parts.append("当前智能体已降级，报告摘要基于已有统计分析结果自动整理。")
        overview = "".join(overview_parts)

        behavior_parts: list[str] = []
        if predicted_label:
            behavior_parts.append(f"当前家庭用电行为被识别为 {self._label_text(predicted_label)}。")
        if "confidence" in classification_result:
            behavior_parts.append(
                f"分类置信度约为 {float(classification_result['confidence']) * 100:.2f}%。"
            )

        if "avg_active_appliance_count" in recent_history_summary:
            behavior_parts.append(
                f"最近窗口内平均活跃电器数量约 {float(recent_history_summary['avg_active_appliance_count']):.2f} 个。"
            )
        if "avg_burst_event_count" in recent_history_summary:
            behavior_parts.append(
                f"平均突发事件数约 {float(recent_history_summary['avg_burst_event_count']):.2f}。"
            )
        behavior = "".join(behavior_parts) or "当前缺少明确的行为分类结果，建议结合历史曲线进一步确认用电模式。"

        risk_parts: list[str] = []
        if "predicted_avg_load_w" in forecast_summary:
            risk_parts.append(
                f"预测平均负荷约 {float(forecast_summary['predicted_avg_load_w']):.2f} W。"
            )
        if "predicted_peak_load_w" in forecast_summary:
            risk_parts.append(
                f"预测峰值负荷约 {float(forecast_summary['predicted_peak_load_w']):.2f} W。"
            )
        risk_flags = forecast_summary.get("risk_flags", [])
        if risk_flags:
            risk_parts.append(f"预测风险标签包括 {risk_flags}。")
        if peak_period:
            risk_parts.append(f"预测高负荷时段集中在 {peak_period}。")
        if "max_load_w" in recent_history_summary:
            risk_parts.append(
                f"历史窗口内最高负荷约 {float(recent_history_summary['max_load_w']):.2f} W。"
            )
        risk = "".join(risk_parts) or "当前缺少预测结果，暂无法给出稳定的风险判断。"

        recommendations = self._build_actions(context)
        if not recommendations:
            recommendations = ["建议优先排查峰时段与夜间持续运行设备。"]
        note = "本报告依据统计分析、行为分类、负荷预测与规则建议自动生成，适合作为阶段性分析与归档材料。"

        return {
            "title": title,
            "overview": overview,
            "sections": [
                {"title": "总体概览", "body": overview},
                {"title": "行为判断", "body": behavior},
                {"title": "预测风险", "body": risk},
                {"title": "附注", "body": note},
            ],
            "recommendations": recommendations,
        }

    def _label_text(self, label: str) -> str:
        return {
            "day_high_night_low": "白天高晚上低型",
            "day_low_night_high": "白天低晚上高型",
            "all_day_high": "全天高负载型",
            "all_day_low": "全天低负载型",
        }.get(label, label)
