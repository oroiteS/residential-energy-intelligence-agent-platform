"""问答意图路由。"""

from __future__ import annotations

from app.agent.state import AgentContext, AgentIntent, SessionMemorySnapshot
from app.contracts import AgentHistoryItem


class IntentRouter:
    """根据问题和上下文决定路由意图。"""

    _ADVICE_KEYWORDS = ("建议", "怎么做", "怎么办", "如何", "优化", "省电", "节能", "调整")
    _FORECAST_KEYWORDS = ("明天", "明日", "预测", "未来", "接下来", "高峰", "峰值", "负荷")
    _CLASSIFICATION_KEYWORDS = ("类型", "模式", "分类", "习惯", "行为", "今天是什么类型")
    _RISK_KEYWORDS = ("风险", "异常", "告警", "预警", "危险")
    _FOLLOW_UP_KEYWORDS = ("为什么", "那", "这个", "具体", "先看", "继续", "然后")

    def route(
        self,
        question: str,
        history: list[AgentHistoryItem],
        context: AgentContext,
        memory: SessionMemorySnapshot | None,
    ) -> AgentIntent:
        normalized_question = question.strip().lower()

        if any(keyword in question for keyword in self._RISK_KEYWORDS):
            return AgentIntent.RISK

        if any(keyword in question for keyword in self._ADVICE_KEYWORDS):
            return AgentIntent.ADVICE

        if any(keyword in question for keyword in self._FORECAST_KEYWORDS):
            return AgentIntent.FORECAST

        if any(keyword in question for keyword in self._CLASSIFICATION_KEYWORDS):
            return AgentIntent.CLASSIFICATION

        if self._looks_like_follow_up(question, history, memory):
            return memory.last_intent if memory and memory.last_intent else AgentIntent.FOLLOW_UP

        if context.forecast_summary.peak_period or context.forecast_summary.risk_flags:
            if "今天" not in normalized_question and "昨日" not in normalized_question:
                return AgentIntent.FORECAST

        if context.classification_result.predicted_label:
            return AgentIntent.CLASSIFICATION

        return AgentIntent.OVERVIEW

    def _looks_like_follow_up(
        self,
        question: str,
        history: list[AgentHistoryItem],
        memory: SessionMemorySnapshot | None,
    ) -> bool:
        short_question = len(question.strip()) <= 12
        keyword_match = any(keyword in question for keyword in self._FOLLOW_UP_KEYWORDS)
        has_context = bool(history or (memory and memory.recent_questions))
        return short_question and keyword_match and has_context

