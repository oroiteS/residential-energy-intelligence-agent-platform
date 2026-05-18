from __future__ import annotations

import json
from datetime import datetime
from time import perf_counter
from typing import Any

from flask import current_app

from app.models import ChatSession
from app.services.analysis_service import get_analysis_payload
from app.services.chat_service import append_message, create_session
from app.services.classification_service import get_latest_classification
from app.services.dataset_service import get_dataset_or_404
from app.services.detection_service import (
    detection_dto,
    ensure_current_detection,
    get_latest_detection_record,
)
from app.services.forecast_service import get_latest_forecast
from app.services.llm_client import can_use_llm, create_chat_model, llm_log_target, unavailable_reason


CLASSIFICATION_TAXONOMY = [
    {
        "label": "外出波动型",
        "meaning": "通常工作日居家用电较少，周末、晚间或返家后用电明显抬升，峰时用电占比较高，用电波动较明显，常见于外出工作人员或作息不完全固定的家庭。",
        "advice_focus": "重点关注周末和晚间集中用电，建议把可调整的家务、洗浴、充电等用电尽量错开峰时段。",
    },
    {
        "label": "峰时集中型",
        "meaning": "用电更多集中在峰时段，说明烹饪、洗浴、家务或其他高功率活动可能集中发生在电价或电网压力较高的时段。",
        "advice_focus": "重点提供错峰用电建议，优先转移可延后的高功率用电。",
    },
    {
        "label": "中高用量型",
        "meaning": "整体用电量处于中高水平，峰谷结构和日间波动通常相对平稳，代表家庭设备使用较频繁但不一定存在异常。",
        "advice_focus": "重点关注未来趋势、峰谷比例和持续偏高时段，给出温和的用电优化建议。",
    },
    {
        "label": "高耗持续型",
        "meaning": "多个日期或多个时段持续保持较高用电水平，可能存在长时间运行设备、待机用电偏高或固定高耗习惯，需要进一步复核。",
        "advice_focus": "重点建议复核夜间、无人时段和连续高耗时段，避免把原因确定归咎于某个具体设备。",
    },
    {
        "label": "规律低耗型",
        "meaning": "整体用电量较低且波动较小，峰谷结构相对稳定，通常代表较规律、较节制的用电习惯。",
        "advice_focus": "重点给出保持性建议和轻量提醒，不应制造不必要的风险感。",
    },
]


def ask_agent(*, dataset_id: int, question: str, session_id: int | None, history: list[dict] | None) -> dict:
    dataset = get_dataset_or_404(dataset_id)
    session = ChatSession.query.get(session_id) if session_id else None
    if session is None:
        session_payload = create_session(dataset_id, title=question[:20] or "新会话")
        session = ChatSession.query.get(session_payload["id"])

    append_message(session_id=session.id, role="user", content=question)

    context = _build_context(dataset_id)
    fallback_payload = _build_fallback_payload(
        session_id=session.id,
        question=question,
        context=context,
        error_reason=unavailable_reason(),
    )

    if can_use_llm():
        payload = _ask_llm(
            session_id=session.id,
            question=question,
            history=history or [],
            context=context,
            fallback=fallback_payload,
        )
    else:
        payload = fallback_payload

    append_message(
        session_id=session.id,
        role="assistant",
        content=payload["answer"],
        assistant_payload=payload,
        model_name=current_app.config.get("LLM_MODEL") or "backend-rule-agent",
    )
    return payload


def _build_context(dataset_id: int) -> dict[str, Any]:
    analysis = get_analysis_payload(dataset_id)
    classification = get_latest_classification(dataset_id)
    forecast = get_latest_forecast(dataset_id)
    current_detection = _get_detection_context(dataset_id, window_role="current", ensure_current=True)
    future_detection = _get_detection_context(dataset_id, window_role="future")

    return {
        "dataset_id": dataset_id,
        "analysis": analysis,
        "classification": classification,
        "classification_taxonomy": CLASSIFICATION_TAXONOMY,
        "current_detection": current_detection,
        "future_detection": future_detection,
        "forecast": {
            "id": forecast.id,
            "summary": forecast.summary,
            "forecast_start": forecast.forecast_start.isoformat(),
            "forecast_end": forecast.forecast_end.isoformat(),
        } if forecast else None,
    }


def _get_detection_context(dataset_id: int, *, window_role: str, ensure_current: bool = False) -> dict | None:
    try:
        if ensure_current and window_role == "current":
            return detection_dto(ensure_current_detection(dataset_id))
        return detection_dto(get_latest_detection_record(dataset_id, window_role=window_role))
    except Exception as exc:
        current_app.logger.warning(
            "[agent] 读取异常检测上下文失败 dataset_id=%s window_role=%s error=%s",
            dataset_id,
            window_role,
            exc,
        )
        return None


def _build_fallback_payload(
    *,
    session_id: int,
    question: str,
    context: dict[str, Any],
    error_reason: str,
) -> dict:
    analysis = context["analysis"]
    classification = context.get("classification")
    forecast = context.get("forecast")
    current_detection = context.get("current_detection")
    future_detection = context.get("future_detection")

    answer_lines = [
        f"当前总用电量统计为 {analysis['summary']['total_kwh']} kWh。",
        f"当前行为判断：{classification['predicted_label']}。" if classification else "当前还没有分类结果。",
        _format_detection_line("当前异常检测", current_detection),
        f"最近一次未来 7 天预测总量约为 {forecast['summary'].get('predicted_total_kwh')} kWh。" if forecast else "当前还没有预测结果。",
        _format_detection_line("未来窗口异常检测", future_detection) if future_detection else "",
    ]

    return {
        "session_id": session_id,
        "answer": " ".join(line for line in answer_lines if line),
        "citations": _citations_from_context(context),
        "actions": [
            "优先查看数据详情页中的峰谷占比变化。",
            "如未来预测持续走高，建议提前执行错峰安排。",
        ],
        "degraded": True,
        "error_reason": error_reason,
        "created_at": datetime.now().astimezone().isoformat(),
        "intent": _infer_intent(question),
        "confidence_level": "medium",
        "missing_information": [],
    }


def _ask_llm(
    *,
    session_id: int,
    question: str,
    history: list[dict],
    context: dict[str, Any],
    fallback: dict,
) -> dict:
    try:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
        from langchain_core.prompts import ChatPromptTemplate

        messages = []
        for item in history[-12:]:
            role = item.get("role")
            content = item.get("content", "")
            if role == "assistant":
                messages.append(AIMessage(content=content))
            elif role == "system":
                messages.append(SystemMessage(content=content))
            else:
                messages.append(HumanMessage(content=content))

        model = create_chat_model()
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "你是居民用电分析与节能建议助手。"
                        "只能基于提供的结构化上下文回答，禁止编造数据。"
                        "不要给出设备级确定性归因。"
                        "结构化上下文中的 classification_taxonomy 是分类体系解释；"
                        "系统将居民用电情况分为 5 类，回答涉及分类时必须结合这些类别含义解释。"
                        "建议动作需要由你根据统计、分类、异常检测和预测结果自行生成，"
                        "不要依赖预设规则建议。"
                        "请输出 JSON 对象，字段必须包含 answer、citations、actions、"
                        "missing_information、confidence_level。"
                        "answer 只写结论、依据解释和关键判断，不要写编号建议清单，"
                        "也不要把具体操作步骤堆在 answer 中。"
                        "citations 必须从“可引用证据”中选择，且必须是对象数组，"
                        "每一项都包含 key、label、value，禁止只返回字符串 key；"
                        "例如：{{\"key\":\"classification\",\"label\":\"当前分类\",\"value\":\"高耗持续型\"}}。"
                        "actions 才能承载具体建议，必须是字符串数组，"
                        "每条是一个用户可以执行的具体动作，避免与 answer 重复。"
                        "涉及具体设备时，只能作为排查示例，不得表述为已经确定的原因；"
                        "必须使用“可能、建议复核、优先检查”等不确定性措辞。"
                        "只有在确实缺少用户侧数据导致无法回答时，missing_information 才输出非空数组；"
                        "如果当前证据已经足够回答，必须输出空数组。"
                    ),
                ),
                ("placeholder", "{history_messages}"),
                (
                    "human",
                    (
                        "用户问题：{question}\n"
                        "结构化上下文：{context_json}\n"
                        "可引用证据：{citations_json}\n"
                        "请给出自然、具体、可执行的回答。"
                    ),
                ),
            ]
        )
        chain = prompt | model
        target = llm_log_target()
        started_at = perf_counter()
        current_app.logger.info(
            "[llm][ask] 正在发送给 %s model=%s dataset_id=%s session_id=%s",
            target["base_url"],
            target["model"],
            context.get("dataset_id"),
            session_id,
        )
        raw = chain.invoke(
            {
                "history_messages": messages,
                "question": question,
                "context_json": json.dumps(context, ensure_ascii=False, default=str),
                "citations_json": json.dumps(_citations_from_context(context), ensure_ascii=False, default=str),
            }
        )
        elapsed_ms = (perf_counter() - started_at) * 1000
        current_app.logger.info(
            "[llm][ask] 大模型回复完成 model=%s duration_ms=%.1f",
            target["model"],
            elapsed_ms,
        )
        payload = _parse_llm_payload(getattr(raw, "content", str(raw)), fallback)
        return {
            **payload,
            "session_id": session_id,
            "degraded": False,
            "error_reason": None,
            "created_at": datetime.now().astimezone().isoformat(),
            "intent": _infer_intent(question),
        }
    except Exception as exc:
        target = llm_log_target()
        current_app.logger.exception(
            "[llm][ask] 发送失败 base_url=%s model=%s error=%s",
            target["base_url"],
            target["model"],
            exc,
        )
        return {
            **fallback,
            "error_reason": "LLM_REQUEST_FAILED",
        }


def _sanitize_missing_information(value: Any) -> list[dict]:
    if not isinstance(value, list):
        return []
    items: list[dict] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            continue
        question = str(item.get("question") or "").strip()
        reason = str(item.get("reason") or "").strip()
        if not question or not reason:
            continue
        key = str(item.get("key") or f"missing_{index + 1}").strip()
        items.append({"key": key, "question": question, "reason": reason})
    return items[:3]


def _sanitize_citations(value: Any, fallback: list[dict]) -> list[dict]:
    if not isinstance(value, list):
        return fallback
    citations: list[dict] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            continue
        key = str(item.get("key") or f"citation_{index + 1}").strip()
        label = str(item.get("label") or "").strip()
        if "value" not in item or not key or not label:
            continue
        citations.append({
            "key": key,
            "label": label,
            "value": item["value"],
        })
    return citations or fallback


def _sanitize_actions(value: Any, fallback: list[str]) -> list[str]:
    if not isinstance(value, list):
        return fallback
    actions = [str(item).strip() for item in value if str(item).strip()]
    return actions or fallback


def _parse_llm_payload(raw: str, fallback: dict) -> dict:
    raw = _extract_json_text(raw)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {
            **fallback,
            "answer": raw.strip() or fallback["answer"],
            "actions": fallback["actions"],
            "missing_information": [],
            "confidence_level": "medium",
        }

    return {
        "answer": str(parsed.get("answer") or fallback["answer"]),
        "citations": _sanitize_citations(parsed.get("citations"), fallback["citations"]),
        "actions": _sanitize_actions(parsed.get("actions"), fallback["actions"]),
        "missing_information": _sanitize_missing_information(parsed.get("missing_information")),
        "confidence_level": parsed.get("confidence_level") if parsed.get("confidence_level") in {"high", "medium", "low"} else "medium",
    }


def _format_detection_line(label: str, detection: dict | None) -> str:
    if not detection:
        return f"{label}暂无结果。"
    status = "建议复核" if detection.get("is_anomaly") else "未触发异常"
    score = detection.get("anomaly_score", 0)
    reasons = detection.get("reasons") or []
    reason_text = ""
    if reasons:
        first_reason = reasons[0]
        if isinstance(first_reason, dict):
            reason_text = f"，主要原因：{first_reason.get('message') or first_reason.get('rule_name')}"
        else:
            reason_text = f"，主要原因：{first_reason}"
    return f"{label}：{status}，偏离分数 {score}{reason_text}。"


def _extract_json_text(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return text


def _citations_from_context(context: dict[str, Any]) -> list[dict]:
    analysis = context["analysis"]
    classification = context.get("classification")
    forecast = context.get("forecast")
    current_detection = context.get("current_detection")
    future_detection = context.get("future_detection")
    citations = [
        {"key": "total_kwh", "label": "总用电量", "value": analysis["summary"]["total_kwh"]},
        {"key": "daily_avg_kwh", "label": "日均用电量", "value": analysis["summary"]["daily_avg_kwh"]},
    ]
    if classification:
        citations.append({"key": "classification", "label": "当前分类", "value": classification["predicted_label"]})
    if current_detection:
        citations.append({
            "key": "current_detection",
            "label": "当前异常检测",
            "value": "建议复核" if current_detection.get("is_anomaly") else "未触发异常",
        })
        citations.append({
            "key": "current_anomaly_score",
            "label": "当前偏离分数",
            "value": current_detection.get("anomaly_score", 0),
        })
    if forecast:
        citations.append({"key": "forecast_total", "label": "预测总量", "value": forecast["summary"].get("predicted_total_kwh")})
    if future_detection:
        citations.append({
            "key": "future_detection",
            "label": "未来窗口异常检测",
            "value": "建议复核" if future_detection.get("is_anomaly") else "未触发异常",
        })
        citations.append({
            "key": "future_anomaly_score",
            "label": "未来偏离分数",
            "value": future_detection.get("anomaly_score", 0),
        })
    return citations


def _infer_intent(question: str) -> str:
    text = question.lower()
    if "预测" in text or "未来" in text:
        return "forecast"
    if "分类" in text or "类型" in text:
        return "classification"
    if "异常" in text or "风险" in text:
        return "risk"
    if "建议" in text or "节能" in text:
        return "advice"
    return "overview"
