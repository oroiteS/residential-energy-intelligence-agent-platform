"""建议候选生成与排序。"""

from __future__ import annotations

from app.agent.state import (
    AdviceCandidate,
    AgentContext,
    AgentIntent,
    EvidenceItem,
    SessionMemorySnapshot,
)


class AdvicePlanner:
    """根据上下文生成候选建议。"""

    def plan(
        self,
        context: AgentContext,
        intent: AgentIntent,
        question: str,
        evidence: list[EvidenceItem],
        memory: SessionMemorySnapshot | None,
    ) -> list[AdviceCandidate]:
        candidates: list[AdviceCandidate] = []

        for rule_advice in context.rule_advices:
            rationale = rule_advice.reason or rule_advice.summary or "来自规则引擎的候选建议。"
            candidates.append(
                AdviceCandidate(
                    key=rule_advice.key,
                    title=rule_advice.summary or rule_advice.action,
                    action=rule_advice.action,
                    rationale=rationale,
                    priority_score=min(max(rule_advice.priority, 40), 95),
                    evidence_keys=self._pick_evidence_keys(evidence, limit=2),
                    category=rule_advice.category or "rule",
                )
            )

        predicted_label = context.classification_result.predicted_label
        if predicted_label == "day_low_night_high":
            candidates.extend(
                [
                    AdviceCandidate(
                        key="night_baseload_check",
                        title="夜间持续负荷排查",
                        action="优先检查夜间持续运行设备，并把可定时设备改为按需运行。",
                        rationale="分类结果显示夜间负荷偏高，先从夜间基线和定时运行设备入手最稳。",
                        priority_score=84,
                        evidence_keys=["predicted_label", "confidence"],
                        category="classification",
                    ),
                    AdviceCandidate(
                        key="night_water_heater_schedule",
                        title="夜间热水器时段优化",
                        action="将热水器设置为更短的定时窗口，避免整段夜间持续加热。",
                        rationale="夜间高负荷型常见于持续保温或长时段运行设备。",
                        priority_score=80,
                        evidence_keys=["predicted_label"],
                        category="classification",
                    ),
                ]
            )
        elif predicted_label == "afternoon_peak":
            candidates.extend(
                [
                    AdviceCandidate(
                        key="afternoon_peak_shift",
                        title="下午高峰错峰",
                        action="将下午可延后的高耗电任务尽量错开执行，减少午后到傍晚的集中启停。",
                        rationale="分类结果显示下午时段更容易形成高峰，优先削减下午叠加负荷。",
                        priority_score=82,
                        evidence_keys=["predicted_label", "peak_ratio"],
                        category="classification",
                    )
                ]
            )
        elif predicted_label == "morning_peak":
            candidates.extend(
                [
                    AdviceCandidate(
                        key="morning_peak_reduce",
                        title="上午高峰压降",
                        action="尽量错开早间集中启动的厨房、电热或清洁类设备，降低上午高峰叠加。",
                        rationale="分类结果显示上午负荷更高，优先控制早间集中启动设备更直接。",
                        priority_score=86,
                        evidence_keys=["predicted_label", "peak_ratio"],
                        category="classification",
                    )
                ]
            )
        elif predicted_label == "all_day_low":
            candidates.append(
                AdviceCandidate(
                    key="maintain_low_load",
                    title="维持平稳负荷模式",
                    action="继续维持当前平稳用电习惯，并重点关注未来高峰时段是否出现异常抬升。",
                    rationale="当前整体负荷较平稳，更适合做趋势监控而不是强干预。",
                    priority_score=60,
                    evidence_keys=["predicted_label"],
                    category="classification",
                )
            )

        peak_period = context.forecast_summary.peak_period
        if peak_period:
            candidates.append(
                AdviceCandidate(
                    key="forecast_peak_shift",
                    title="次日高峰避让",
                    action=f"尽量避开 {peak_period} 的高负荷窗口安排洗衣、热水器或充电等任务。",
                    rationale="预测结果已经给出高负荷时段，优先做时段错峰最直接。",
                    priority_score=88,
                    evidence_keys=["forecast_peak_period", "predicted_peak_load_w"],
                    category="forecast",
                )
            )

        if context.forecast_summary.risk_flags:
            candidates.append(
                AdviceCandidate(
                    key="risk_flag_followup",
                    title="风险标签优先处理",
                    action="优先处理预测风险标签对应的高风险时段，避免多个高功率设备同时运行。",
                    rationale="风险标签通常比普通统计特征更接近需要优先响应的问题。",
                    priority_score=90,
                    evidence_keys=["risk_flags", "forecast_peak_period"],
                    category="risk",
                )
            )

        if memory and memory.recent_actions:
            candidates.append(
                AdviceCandidate(
                    key="memory_followup",
                    title="延续上轮建议",
                    action=f"优先确认你上一轮提到的建议是否已经执行：{memory.recent_actions[0]}",
                    rationale="当前是连续对话，先确认上轮动作执行情况，可以避免重复给建议。",
                    priority_score=78,
                    evidence_keys=["active_goal"],
                    category="memory",
                )
            )

        scored_candidates = [self._score_candidate(item, intent, question) for item in candidates]
        return self._deduplicate(scored_candidates)[:5]

    def _score_candidate(
        self,
        candidate: AdviceCandidate,
        intent: AgentIntent,
        question: str,
    ) -> AdviceCandidate:
        score = candidate.priority_score
        question_text = question.strip()

        if intent == AgentIntent.ADVICE:
            score += 8
        if intent == AgentIntent.RISK and candidate.category in {"risk", "forecast"}:
            score += 10
        if intent == AgentIntent.FORECAST and candidate.category == "forecast":
            score += 8
        if "夜间" in question_text and "夜" in candidate.action:
            score += 8
        if ("明天" in question_text or "明日" in question_text) and candidate.category in {"forecast", "risk"}:
            score += 8
        if "峰" in question_text and "高峰" in candidate.title:
            score += 6

        return candidate.model_copy(update={"priority_score": min(score, 100)})

    def _deduplicate(self, candidates: list[AdviceCandidate]) -> list[AdviceCandidate]:
        deduplicated: list[AdviceCandidate] = []
        seen_actions: set[str] = set()

        for candidate in sorted(candidates, key=lambda item: item.priority_score, reverse=True):
            if candidate.action in seen_actions:
                continue
            deduplicated.append(candidate)
            seen_actions.add(candidate.action)

        return deduplicated

    def _pick_evidence_keys(self, evidence: list[EvidenceItem], limit: int) -> list[str]:
        return [item.key for item in evidence[:limit]]
