#!/usr/bin/env python3
"""统计规则引擎：为异常样本提供可解释的原因。

不依赖模型训练，基于：
- 用户自身历史统计（3σ 偏离）
- 全局百分位数（阈值外）
生成人类可读的异常原因清单。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

matplotlib.rcParams["font.family"] = ["Arial Unicode MS", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

FEATURE_COLUMNS = [
    # 与分类和 Isolation Forest 共用的 16 维窗口特征。
    # 规则引擎基于这些特征生成可读的异常原因。
    "avg_energy",
    "std_energy",
    "max_energy",
    "min_energy",
    "avg_peak",
    "avg_valley",
    "peak_valley_ratio",
    "peak_ratio",
    "valley_ratio",
    "load_factor",
    "workday_avg",
    "weekend_avg",
    "weekend_workday_ratio",
    "trend_rel",
    "volatility",
    "med_mean_ratio",
]


# ── 规则定义 ──────────────────────────────────────────────

@dataclass
class Rule:
    """一条可解释异常规则。

    feature 指明检查哪个特征；
    method 指明使用用户自身 3σ、全局百分位或绝对值百分位；
    direction 指明异常方向；
    reason_template 是最终写入报告或前端的解释文本模板。
    """

    name: str
    feature: str  # 检查哪个特征
    method: str   # "sigma" | "percentile" | "abs_percentile"
    direction: str  # "high" | "low" | "both"
    reason_template: str

    def check(
        self,
        value: float,
        user_stats: dict | None,
        global_p: dict,
        sigma_mult: float,
    ) -> tuple[bool, float]:
        """判断某个特征值是否触发规则。

        返回值：
        - triggered: 是否触发；
        - severity: 偏离程度，后续用于排序或解释。
        """

        if self.method == "sigma":
            # 用户自身 3σ 规则：判断当前窗口是否明显偏离该用户历史习惯。
            if user_stats is None:
                return False, 0.0
            mu = user_stats.get("mean", 0.0)
            std = user_stats.get("std", 1e-8)
            z = (value - mu) / max(std, 1e-8)
            if self.direction == "high" and z > sigma_mult:
                return True, abs(z)
            if self.direction == "low" and z < -sigma_mult:
                return True, abs(z)
            if self.direction == "both" and abs(z) > sigma_mult:
                return True, abs(z)
            return False, abs(z)

        elif self.method == "percentile":
            # 全局百分位规则：判断当前窗口是否位于全体用户分布的极端位置。
            lo = global_p.get(f"{self.feature}_lo", -np.inf)
            hi = global_p.get(f"{self.feature}_hi", np.inf)
            if self.direction == "high" and value > hi:
                return True, value / max(hi, 1e-8)
            if self.direction == "low" and value < lo:
                return True, max(lo, 1e-8) / max(value, 1e-8)
            if self.direction == "both" and (value > hi or value < lo):
                if value > hi:
                    return True, value / max(hi, 1e-8)
                else:
                    return True, max(lo, 1e-8) / max(value, 1e-8)
            return False, 1.0

        elif self.method == "abs_percentile":
            # 趋势类特征同时关心快速上升和快速下降，因此使用绝对值阈值。
            hi = global_p.get(f"{self.feature}_abs_hi", np.inf)
            if abs(value) > hi:
                return True, abs(value) / max(hi, 1e-8)
            return False, 1.0

        return False, 0.0


RULES = [
    # ── 总量维度 ──
    Rule("avg_energy_user_high", "avg_energy", "sigma", "high",
         "周均用电量超过用户历史均值 {z:.1f} 个标准差（{value:.2f} vs {mu:.2f}）"),
    Rule("avg_energy_global_high", "avg_energy", "percentile", "high",
         "周均用电量超过全体用户 95% 分位（{value:.2f}）"),
    Rule("avg_energy_global_low", "avg_energy", "percentile", "low",
         "周均用电量低于全体用户 5% 分位（{value:.2f}）"),
    Rule("max_energy_user_high", "max_energy", "sigma", "high",
         "单日最高用电量超过用户历史均值 {z:.1f} 个标准差（{value:.2f} vs {mu:.2f}）"),
    Rule("min_energy_user_low", "min_energy", "sigma", "low",
         "单日最低用电量低于用户历史均值 {z:.1f} 个标准差（{value:.2f} vs {mu:.2f}）"),

    # ── 波动维度 ──
    Rule("std_energy_global_high", "std_energy", "percentile", "high",
         "日用电波动超过全体用户 95% 分位，本周用电起伏剧烈（标准差 {value:.2f}）"),
    Rule("volatility_global_high", "volatility", "percentile", "high",
         "逐日波动性位列全体用户前 5%，用电忽高忽低（{value:.3f}）"),

    # ── 峰谷结构 ──
    Rule("avg_peak_global_high", "avg_peak", "percentile", "high",
         "峰时段日均用电量偏高，说明较高用电更多发生在峰时段，建议优先排查可转移到谷时段的洗衣、热水、充电等用电任务"),
    Rule("avg_valley_global_high", "avg_valley", "percentile", "high",
         "谷时段日均用电量本身偏高；谷时用电占比高不算异常，但若绝对电量持续偏高，建议复核夜间常开设备或待机用电"),
    Rule("peak_ratio_global_high", "peak_ratio", "percentile", "high",
         "峰时用电占比偏高（{value:.1%}），说明用电更集中在峰时段，建议考虑错峰安排可延后的用电任务"),
    Rule("peak_valley_ratio_high", "peak_valley_ratio", "percentile", "high",
         "峰谷比偏高（{value:.1f}），说明峰时用电相对谷时明显偏重，建议关注峰时集中启动设备的情况"),
    Rule("load_factor_global_low", "load_factor", "percentile", "low",
         "用电均衡度过低（{value:.2f}），存在单日峰值拉高均值"),

    # ── 趋势维度 ──
    Rule("trend_rel_extreme", "trend_rel", "abs_percentile", "both",
         "7 天用电趋势变化极端（{value:.3f}），短期涨跌剧烈"),

    # ── 作息维度 ──
    Rule("weekend_workday_ratio_user", "weekend_workday_ratio", "sigma", "both",
         "周末/工作日用电比偏离用户习惯 {z:.1f} 个标准差（{value:.2f} vs {mu:.2f}）"),
]


# ── 统计计算 ──────────────────────────────────────────────

def load_config(config_path: Path) -> dict:
    """读取统计规则配置。"""

    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_features(features_path: Path) -> pd.DataFrame:
    """读取窗口特征并校验规则所需列。"""

    df = pd.read_csv(features_path, encoding="utf-8")
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"特征列缺失：{missing}")
    return df


def compute_user_stats(df: pd.DataFrame) -> dict[str, dict]:
    """计算每个用户各特征的历史均值和标准差。

    用户级统计用于判断“相对自己是否异常”，例如某用户本周是否明显高于自己历史水平。
    """

    grouped = df.groupby("user_id")
    stats: dict[str, dict] = {}
    for uid, group in grouped:
        user_stats = {}
        for feat in FEATURE_COLUMNS:
            vals = group[feat].values.astype(np.float64)
            user_stats[feat] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
            }
        stats[uid] = user_stats
    return stats


def compute_global_percentiles(df: pd.DataFrame, cfg: dict) -> dict[str, float]:
    """计算全局百分位阈值。

    P5/P95 用于判断“相对全体用户是否极端”，是后端推理阶段规则阈值的来源。
    """

    lo = float(cfg["rules"]["percentile_lower"])
    hi = float(cfg["rules"]["percentile_upper"])
    thresholds: dict[str, float] = {}
    for feat in FEATURE_COLUMNS:
        vals = df[feat].values.astype(np.float64)
        thresholds[f"{feat}_lo"] = float(np.percentile(vals, lo))
        thresholds[f"{feat}_hi"] = float(np.percentile(vals, hi))
        # 对趋势类特征存绝对值百分位
        thresholds[f"{feat}_abs_hi"] = float(np.percentile(np.abs(vals), hi))
    return thresholds


def apply_rules(
    row: pd.Series,
    user_stats: dict[str, dict] | None,
    global_p: dict,
    sigma_mult: float,
) -> list[dict]:
    """对单个窗口样本应用全部规则。"""

    uid = row["user_id"]
    reasons = []
    for rule in RULES:
        value = float(row[rule.feature])
        us = user_stats.get(uid, {}).get(rule.feature) if user_stats else None
        triggered, severity = rule.check(value, us, global_p, sigma_mult)
        if triggered:
            mu = us.get("mean", 0) if us else 0
            reason_text = rule.reason_template.format(
                value=value, z=severity, mu=mu,
            )
            reasons.append({
                "rule": rule.name,
                "feature": rule.feature,
                "method": rule.method,
                "severity": round(severity, 2),
                "reason": reason_text,
            })
    return reasons


# ── 可视化 ────────────────────────────────────────────────

RULE_LABELS_SHORT = {
    "avg_energy_user_high": "总量（3σ·高）",
    "avg_energy_global_high": "总量（P95·高）",
    "avg_energy_global_low": "总量（P5·低）",
    "max_energy_user_high": "最高（3σ）",
    "min_energy_user_low": "最低（3σ）",
    "std_energy_global_high": "波动（P95）",
    "volatility_global_high": "逐日波动性（P95）",
    "avg_peak_global_high": "峰时电量（P95）",
    "avg_valley_global_high": "谷时电量（P95）",
    "peak_ratio_global_high": "峰占比（P95）",
    "peak_valley_ratio_high": "峰谷比（P95）",
    "load_factor_global_low": "均衡度（P5）",
    "trend_rel_extreme": "趋势极端",
    "weekend_workday_ratio_user": "作息偏离（3σ）",
}


def plot_rule_counts(rule_counter: dict[str, int], output_dir: Path) -> Path:
    """规则触发频次柱状图。"""
    names = list(rule_counter.keys())
    counts = list(rule_counter.values())
    labels = [RULE_LABELS_SHORT.get(n, n) for n in names]
    colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, len(names)))
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(labels[::-1], counts[::-1], color=colors[::-1])
    for bar, count in zip(bars, counts[::-1]):
        ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height() / 2,
                str(count), va="center", fontsize=8)
    ax.set_xlabel("触发次数")
    ax.set_title("异常规则触发频次")
    plt.tight_layout()
    out = output_dir / "rule_counts.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_user_anomaly_hist(anomaly_per_user: pd.Series, output_dir: Path) -> Path:
    """每用户异常窗口数直方图。"""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(anomaly_per_user.values, bins=40, color="steelblue",
            alpha=0.8, edgecolor="white", linewidth=0.3)
    ax.axvline(x=anomaly_per_user.median(), color="tomato", linestyle="--",
               linewidth=1.5, label=f"中位数 = {anomaly_per_user.median():.0f}")
    ax.set_xlabel("异常窗口数")
    ax.set_ylabel("用户数")
    ax.set_title("每用户被标记异常窗口数量分布")
    ax.legend()
    plt.tight_layout()
    out = output_dir / "user_anomaly_hist.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_rule_cooccurrence(rule_matrix: np.ndarray, rule_names: list[str],
                           output_dir: Path) -> Path:
    """规则共现热力图。"""
    cooccur = np.dot(rule_matrix.T, rule_matrix)
    np.fill_diagonal(cooccur, 0)
    labels = [RULE_LABELS_SHORT.get(n, n) for n in rule_names]
    n = len(labels)
    figsize = max(8, n * 0.55)
    fig, ax = plt.subplots(figsize=(figsize, figsize))
    sns.heatmap(
        cooccur, annot=True, fmt="d", cmap="YlOrRd",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, cbar_kws={"label": "共现次数"}, ax=ax,
        annot_kws={"fontsize": 7},
    )
    ax.set_title("规则共现矩阵（同时触发次数）")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()
    out = output_dir / "rule_cooccurrence.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ── 主流程 ────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """解析统计规则入口参数。"""

    default_config = Path(__file__).resolve().parent / "config" / "default.yaml"
    parser = argparse.ArgumentParser(description="统计规则引擎异常原因解释")
    parser.add_argument("--config", type=Path, default=default_config)
    return parser.parse_args()


def main() -> None:
    """统计规则主流程。

    先计算用户级统计和全局百分位阈值，再优先解释 Isolation Forest 标记的异常样本；
    如果没有 Isolation Forest 输出，则对全量窗口应用规则。
    """

    args = parse_args()
    cfg = load_config(args.config)
    sigma_mult = float(cfg["rules"]["sigma_multiplier"])

    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / cfg["output"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载 7 天窗口特征。
    features_path = project_root / cfg["data"]["features_path"]
    df = load_features(features_path)
    print(f"加载样本数：{len(df)}")

    # 计算两类基准：
    # user_stats 用于用户自身 3σ 偏离；
    # global_p 用于全体样本 P5/P95 极端值。
    user_stats = compute_user_stats(df)
    print(f"用户统计数：{len(user_stats)}")

    global_p = compute_global_percentiles(df, cfg)
    thresholds = {
        f"P{cfg['rules']['percentile_lower']}": {
            feat: round(global_p[f"{feat}_lo"], 4)
            for feat in FEATURE_COLUMNS
        },
        f"P{cfg['rules']['percentile_upper']}": {
            feat: round(global_p[f"{feat}_hi"], 4)
            for feat in FEATURE_COLUMNS
        },
    }
    threshold_path = output_dir / cfg["output"]["threshold_file"]
    with open(threshold_path, "w", encoding="utf-8") as f:
        json.dump(thresholds, f, ensure_ascii=False, indent=2)
    print(f"规则阈值已保存：{threshold_path}")

    # 确定分析目标：优先用 IForest 标记的异常样本，否则分析全量。
    # 这样规则引擎主要承担“解释异常原因”，而不是替代无监督模型筛选异常。
    anomaly_path = project_root / cfg["data"]["anomaly_scores_path"]
    if anomaly_path.exists():
        anomaly_df = pd.read_csv(anomaly_path, encoding="utf-8")
        anomaly_ids = set(
            tuple(r) for r in anomaly_df.loc[anomaly_df["is_anomaly"] == True,
            ["user_id", "window_start", "window_end"]].values
        )
        target = df[df.apply(
            lambda row: (row["user_id"], row["window_start"], row["window_end"]) in anomaly_ids,
            axis=1,
        )].copy()
        print(f"从 IForest 结果中筛选异常样本：{len(target)}")
    else:
        target = df.copy()
        print(f"未找到 IForest 结果，分析全量样本：{len(target)}")

    # 逐样本应用规则，并统计每条规则的触发次数。
    rows = []
    rule_counter: dict[str, int] = {}
    rule_idx: dict[str, int] = {}  # rule name → column index
    triggered_per_sample: list[set[str]] = []  # 每个样本触发了哪些规则
    for _, row in target.iterrows():
        reasons = apply_rules(row, user_stats, global_p, sigma_mult)
        triggered_set: set[str] = set()
        for r in reasons:
            rule_counter[r["rule"]] = rule_counter.get(r["rule"], 0) + 1
            if r["rule"] not in rule_idx:
                rule_idx[r["rule"]] = len(rule_idx)
            triggered_set.add(r["rule"])
        triggered_per_sample.append(triggered_set)
        rows.append({
            "user_id": row["user_id"],
            "window_start": row["window_start"],
            "window_end": row["window_end"],
            "reason_count": len(reasons),
            "reasons": json.dumps(reasons, ensure_ascii=False),
        })

    result = pd.DataFrame(rows)
    result = result.sort_values("reason_count", ascending=False).reset_index(drop=True)

    reasons_path = output_dir / cfg["output"]["anomaly_reasons_file"]
    result.to_csv(reasons_path, index=False, encoding="utf-8")
    print(f"异常原因已写入：{reasons_path}（{len(result)} 样本）")

    triggered = result[result["reason_count"] > 0]
    print(f"触发 ≥1 条规则的样本数：{len(triggered)} / {len(result)}")

    # 构建规则共现矩阵（样本 × 规则 二值）
    rule_names_ordered = sorted(rule_idx, key=lambda k: rule_idx[k])
    rule_matrix = np.zeros((len(target), len(rule_names_ordered)), dtype=np.int32)
    for i, ts in enumerate(triggered_per_sample):
        for name in ts:
            rule_matrix[i, rule_idx[name]] = 1

    # 每用户异常窗口数
    anomaly_per_user = result.groupby("user_id").size().sort_values(ascending=False)

    # 图表
    plot_paths = []
    plot_paths.append(plot_rule_counts(rule_counter, output_dir))
    plot_paths.append(plot_user_anomaly_hist(anomaly_per_user, output_dir))
    plot_paths.append(plot_rule_cooccurrence(rule_matrix, rule_names_ordered, output_dir))
    print(f"图表已保存：\n" + "\n".join(str(p) for p in plot_paths))

    # 规则触发频次摘要
    summary = {
        "total_samples_analyzed": len(target),
        "samples_with_reasons": len(triggered),
        "avg_reasons_per_sample": round(float(result["reason_count"].mean()), 2),
        "rule_counts": dict(sorted(rule_counter.items(), key=lambda x: -x[1])),
    }
    summary_path = output_dir / cfg["output"]["rule_summary_file"]
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"规则摘要已写入：{summary_path}")


if __name__ == "__main__":
    main()
