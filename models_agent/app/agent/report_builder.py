"""报告摘要构建。"""

from __future__ import annotations

from app.agent.state import (
    AgentContext,
    AdviceCandidate,
    ConfidenceLevel,
    EvidenceItem,
    ReportSection,
    ReportSummaryOutput,
    REPORT_SECTION_ORDER,
    label_text,
)


class ReportBuilder:
    """构建报告摘要的降级版本。"""

    def build(
        self,
        *,
        context: AgentContext,
        evidence: list[EvidenceItem],
        advice_candidates: list[AdviceCandidate],
        confidence_level: ConfidenceLevel,
    ) -> ReportSummaryOutput:
        dataset_name = (context.dataset.name or "").strip()
        title = "居民用电分析报告"
        if dataset_name:
            title = f"{title} - {dataset_name}"

        overview_parts: list[str] = []
        if context.analysis_summary.daily_avg_kwh is not None:
            overview_parts.append(f"日均用电量约为 {float(context.analysis_summary.daily_avg_kwh):.2f} kWh。")
        if context.analysis_summary.peak_ratio is not None:
            ratio = float(context.analysis_summary.peak_ratio)
            if ratio <= 1:
                ratio *= 100
            overview_parts.append(f"峰时占比约为 {ratio:.2f}%。")
        if context.classification_result.predicted_label:
            overview_parts.append(f"行为类型判断为 {label_text(context.classification_result.predicted_label)}。")
        if context.historical_classification_results:
            history_label = self._dominant_classification_label(
                context.historical_classification_results
            )
            if history_label:
                overview_parts.append(f"过去多天分类以 {label_text(history_label)} 为主。")
        if context.forecast_summary.peak_period:
            overview_parts.append(f"预测高负荷时段集中在 {context.forecast_summary.peak_period}。")
        if context.future_forecast_summaries:
            peak_day_text = self._peak_future_day_text(context.future_forecast_summaries)
            if peak_day_text:
                overview_parts.append(peak_day_text)
        if not overview_parts:
            overview_parts.append("当前报告基于已有统计、分类与预测结果自动整理。")
        overview_parts.append(f"当前总结置信等级为 {confidence_level.value}。")
        overview = "".join(overview_parts)

        behavior_parts: list[str] = []
        if context.classification_result.predicted_label:
            behavior_parts.append(f"当前家庭用电行为被识别为 {label_text(context.classification_result.predicted_label)}。")
        if context.classification_result.confidence is not None:
            behavior_parts.append(f"分类置信度约为 {float(context.classification_result.confidence) * 100:.2f}%。")
        if context.future_classification_results:
            future_label = self._dominant_classification_label(
                context.future_classification_results
            )
            if future_label:
                behavior_parts.append(f"未来多天分类判断以 {label_text(future_label)} 为主。")
        if context.recent_history_summary.avg_active_appliance_count is not None:
            behavior_parts.append(
                f"平均活跃电器数量约 {float(context.recent_history_summary.avg_active_appliance_count):.2f} 个。"
            )
        if context.recent_history_summary.avg_burst_event_count is not None:
            behavior_parts.append(
                f"平均突发事件数约 {float(context.recent_history_summary.avg_burst_event_count):.2f}。"
            )
        behavior = "".join(behavior_parts) or "当前缺少足够的行为识别结果，行为判断仍偏保守。"

        risk_parts: list[str] = []
        if context.forecast_summary.predicted_avg_load_w is not None:
            risk_parts.append(f"预测平均负荷约 {float(context.forecast_summary.predicted_avg_load_w):.2f} W。")
        if context.forecast_summary.predicted_peak_load_w is not None:
            risk_parts.append(f"预测峰值负荷约 {float(context.forecast_summary.predicted_peak_load_w):.2f} W。")
        if context.forecast_summary.predicted_total_kwh is not None:
            risk_parts.append(f"预测总用电量约 {float(context.forecast_summary.predicted_total_kwh):.2f} kWh。")
        if context.forecast_summary.risk_flags:
            risk_parts.append(f"预测风险标签包括 {context.forecast_summary.risk_flags}。")
        if context.forecast_summary.peak_period:
            risk_parts.append(f"需要重点关注 {context.forecast_summary.peak_period} 的高负荷窗口。")
        risk = "".join(risk_parts) or "当前缺少稳定的预测风险信息，建议补充预测摘要。"

        note = (
            "本报告由 agent 工作流自动生成，已统一整合分类结果、预测摘要、统计信息与候选建议。"
            f"当前共纳入 {len(evidence)} 条结构化证据。"
        )

        sections = self._ordered_sections(
            [
                ReportSection(title="总体概览", body=overview),
                ReportSection(title="行为判断", body=behavior),
                ReportSection(title="预测风险", body=risk),
                ReportSection(title="附注", body=note),
            ]
        )
        recommendations = [item.action for item in advice_candidates[:5]]
        if not recommendations:
            recommendations = ["建议优先补充预测摘要，并先检查峰时段与持续运行设备。"]

        return ReportSummaryOutput(
            title=title,
            overview=overview,
            sections=sections,
            recommendations=recommendations,
        )

    def _ordered_sections(self, sections: list[ReportSection]) -> list[ReportSection]:
        mapping = {item.title: item for item in sections if item.title in REPORT_SECTION_ORDER}
        return [mapping[title] for title in REPORT_SECTION_ORDER if title in mapping]

    def _dominant_classification_label(self, items: list[object]) -> str | None:
        label_counts: dict[str, int] = {}
        for item in items:
            predicted_label = getattr(item, "predicted_label", None)
            if not predicted_label:
                continue
            label_counts[predicted_label] = label_counts.get(predicted_label, 0) + 1
        if not label_counts:
            return None
        return max(label_counts.items(), key=lambda pair: (pair[1], pair[0]))[0]

    def _peak_future_day_text(self, items: list[object]) -> str:
        valid_items = [
            item for item in items if getattr(item, "predicted_peak_load_w", None) is not None
        ]
        if not valid_items:
            return ""
        peak_item = max(
            valid_items,
            key=lambda item: float(getattr(item, "predicted_peak_load_w", 0.0) or 0.0),
        )
        day_offset = getattr(peak_item, "day_offset", None)
        if day_offset is not None:
            day_label = f"未来第 {day_offset} 天"
        else:
            day_label = str(getattr(peak_item, "date", "后续某天"))
        return f"{day_label} 的预测峰值最高。"
