from __future__ import annotations

import json
import pickle
from functools import lru_cache
from typing import Any, Protocol, Sequence, cast

import numpy as np

from models.classification import FEATURE_COLUMNS, extract_window_features
from models.common import ARTIFACTS_DIR


class IsolationForestLike(Protocol):
    """异常检测模型的最小推理协议。"""

    def decision_function(self, X: Any) -> Any: ...

    def predict(self, X: Any) -> Any: ...


@lru_cache(maxsize=1)
def _load_detection_artifacts() -> tuple[IsolationForestLike, dict]:
    """加载异常检测模型和统计规则阈值。"""

    iforest_path = ARTIFACTS_DIR / "detection" / "isolation_forest" / "isolation_forest.pkl"
    thresholds_path = ARTIFACTS_DIR / "detection" / "statistical_rules" / "rule_thresholds.json"
    with iforest_path.open("rb") as file:
        model = cast(IsolationForestLike, pickle.load(file))
    thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))
    return model, thresholds


def detect_daily_window(window_rows: Sequence[dict], history_rows: Sequence[dict], *, window_role: str) -> dict:
    """对一个日级窗口执行异常检测。

    Isolation Forest 给出整体异常分数；
    统计规则补充可解释原因，两者任一触发都会标记为异常。
    """

    if not window_rows:
        raise ValueError("异常检测窗口不能为空")

    # 检测复用分类特征，保证“分类解释”和“异常解释”基于同一组窗口指标。
    features = extract_window_features(window_rows)
    model, thresholds = _load_detection_artifacts()
    vector = np.array([[features[column] for column in FEATURE_COLUMNS]], dtype=np.float64)

    # decision_function 越低通常表示越异常；
    # predict 输出 -1 表示 Isolation Forest 判定为异常点。
    score = float(model.decision_function(vector)[0])
    label = int(model.predict(vector)[0])
    reasons = _apply_rules(features, thresholds)
    is_anomaly = label == -1 or bool(reasons)
    severity = _severity(score, reasons)

    return {
        "window_role": window_role,
        "is_anomaly": is_anomaly,
        "anomaly_score": round(score, 6),
        "severity": severity,
        "reasons": reasons,
        "feature_summary": {
            key: round(float(features[key]), 4)
            for key in FEATURE_COLUMNS
        },
    }


def _apply_rules(features: dict[str, float], thresholds: dict) -> list[dict]:
    """应用统计阈值规则生成异常原因。

    thresholds 中的 P5/P95 来自训练或统计样本；
    高于 P95 或低于 P5 的特征会生成可解释的复核原因。
    """

    p5 = thresholds.get("P5", {})
    p95 = thresholds.get("P95", {})

    # 规则列表集中定义“检查哪个特征、看高还是低、触发后如何解释”。
    # 这样后续调整解释文案或阈值方向时不需要改检测主流程。
    rules = [
        (
            "avg_energy_global_high",
            "avg_energy",
            "high",
            "最近 7 天日均总用电显著高于常见水平，建议核对是否存在新增大功率设备、空调/取暖持续运行或家庭成员作息变化",
        ),
        (
            "avg_energy_global_low",
            "avg_energy",
            "low",
            "最近 7 天日均总用电显著低于常见水平，可能对应外出、设备停用、数据漏采或计量异常，建议结合实际居住状态复核",
        ),
        (
            "max_energy_global_high",
            "max_energy",
            "high",
            "窗口内出现单日总用电尖峰，说明至少有一天用电远高于常见水平，建议定位该日是否有集中使用大功率设备",
        ),
        (
            "min_energy_global_low",
            "min_energy",
            "low",
            "窗口内出现单日总用电低谷，若当天并非外出或停电，建议检查数据是否缺失或采集是否中断",
        ),
        (
            "std_energy_global_high",
            "std_energy",
            "high",
            "最近 7 天日用电差异较大，说明不同日期之间负荷变化明显，建议结合家庭作息或设备启停记录复核",
        ),
        (
            "volatility_global_high",
            "volatility",
            "high",
            "相邻日期之间用电跳变明显，可能存在短期作息改变、设备集中启停或数据采集波动",
        ),
        (
            "avg_peak_global_high",
            "avg_peak",
            "high",
            "峰时段日均用电量偏高，说明高负荷更多发生在峰时段，建议优先排查可转移到谷时段的洗衣、热水、充电等负荷",
        ),
        (
            "avg_valley_global_high",
            "avg_valley",
            "high",
            "谷时段日均用电量本身偏高；谷时用电占比高不算异常，但若绝对电量持续偏高，建议复核夜间常开设备或待机负荷",
        ),
        (
            "peak_ratio_global_high",
            "peak_ratio",
            "high",
            "峰时用电占比偏高，说明用电更集中在峰时段，建议考虑错峰安排可延后的用电任务",
        ),
        (
            "peak_valley_ratio_high",
            "peak_valley_ratio",
            "high",
            "峰谷电量比偏高，说明峰时负荷相对谷时明显偏重，建议关注峰时集中启动设备的情况",
        ),
        (
            "load_factor_global_low",
            "load_factor",
            "low",
            "负荷均衡度偏低，说明窗口内存在明显尖峰而非平稳用电，建议查看是否有集中短时高耗设备",
        ),
        (
            "weekend_workday_ratio_high",
            "weekend_workday_ratio",
            "high",
            "周末用电明显高于工作日，可能对应休息日在家时间增加或集中使用设备，建议结合实际作息判断是否合理",
        ),
        (
            "weekend_workday_ratio_low",
            "weekend_workday_ratio",
            "low",
            "周末用电明显低于工作日，可能对应周末外出、空置或采集缺口，建议结合居住状态复核",
        ),
        (
            "med_mean_ratio_global_low",
            "med_mean_ratio",
            "low",
            "中位数明显低于均值，说明少数高耗日期拉高了整体水平，建议优先定位这些高耗日期的具体原因",
        ),
    ]

    reasons: list[dict] = []
    for rule_name, feature, direction, reason_text in rules:
        value = float(features[feature])
        if direction == "high":
            threshold = float(p95.get(feature, np.inf))
            triggered = value > threshold
            severity = value / max(threshold, 1e-6) if triggered else 0.0
        else:
            threshold = float(p5.get(feature, -np.inf))
            triggered = value < threshold
            severity = threshold / max(value, 1e-6) if triggered else 0.0

        if triggered:
            reasons.append(
                {
                    "rule": rule_name,
                    "feature": feature,
                    "method": "percentile",
                    "severity": round(float(severity), 2),
                    "reason": f"{reason_text}（当前值 {value:.4f}，阈值 {threshold:.4f}）。",
                }
            )

    # trend_rel 同时关心快速上升和快速下降，因此使用绝对值和双侧阈值判断。
    trend_value = abs(float(features["trend_rel"]))
    trend_low_threshold = float(p5.get("trend_rel", 0.0))
    trend_high_threshold = float(p95.get("trend_rel", 0.0))
    trend_threshold = max(abs(trend_low_threshold), abs(trend_high_threshold))
    if trend_threshold > 0 and trend_value > trend_threshold:
        trend_direction = "上升" if float(features["trend_rel"]) > 0 else "下降"
        reasons.append(
            {
                "rule": "trend_rel_extreme",
                "feature": "trend_rel",
                "method": "abs_percentile",
                "severity": round(trend_value / trend_threshold, 2),
                "reason": f"7 天用电趋势{trend_direction}过快，说明窗口内用电水平发生连续变化，建议结合天气、作息或设备使用变化复核（当前绝对值 {trend_value:.4f}，阈值 {trend_threshold:.4f}）。",
            }
        )
    return sorted(reasons, key=lambda item: item["severity"], reverse=True)


def _severity(score: float, reasons: list[dict]) -> str:
    """根据模型分数和规则数量估计异常严重程度。"""

    if score < -0.08 or len(reasons) >= 3:
        return "high"
    if score < 0 or reasons:
        return "medium"
    return "low"
