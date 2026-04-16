"""模型比较的统计检验。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def compare_model_vs_baseline(
    per_house_metrics: pd.DataFrame,
) -> dict[str, Any]:
    if per_house_metrics.empty:
        return {
            "house_count": 0,
            "test_name": None,
            "message": "逐家庭指标为空，无法执行统计比较",
        }

    model_values = per_house_metrics["model_mae"].to_numpy(dtype=float)
    baseline_values = per_house_metrics["baseline_mae"].to_numpy(dtype=float)
    differences = model_values - baseline_values
    valid_mask = np.isfinite(differences)
    model_values = model_values[valid_mask]
    baseline_values = baseline_values[valid_mask]
    differences = differences[valid_mask]
    house_count = int(len(differences))
    if house_count == 0:
        return {
            "house_count": 0,
            "test_name": None,
            "message": "逐家庭误差差值全部无效，无法执行统计比较",
        }

    summary: dict[str, Any] = {
        "house_count": house_count,
        "model_mae_mean": float(np.mean(model_values)),
        "baseline_mae_mean": float(np.mean(baseline_values)),
        "mae_improvement_mean": float(np.mean(baseline_values - model_values)),
        "mae_improvement_median": float(np.median(baseline_values - model_values)),
        "difference_mean": float(np.mean(differences)),
        "difference_std": float(np.std(differences, ddof=1)) if house_count > 1 else 0.0,
    }

    shapiro_p = None
    if 3 <= house_count <= 5000:
        try:
            shapiro_p = float(stats.shapiro(differences).pvalue)
        except Exception:
            shapiro_p = None
    summary["shapiro_pvalue"] = shapiro_p

    pooled_std = float(np.std(differences, ddof=1)) if house_count > 1 else 0.0
    summary["effect_size_dz"] = float(np.mean(differences) / pooled_std) if pooled_std > 0 else 0.0

    if house_count < 3:
        summary["test_name"] = None
        summary["message"] = "家庭数少于 3，仅输出描述性统计"
        return summary

    if shapiro_p is not None and shapiro_p >= 0.05 and house_count >= 8:
        test_result = stats.ttest_rel(model_values, baseline_values, alternative="less")
        summary.update(
            {
                "test_name": "paired_t_test",
                "statistic": float(test_result.statistic),
                "pvalue": float(test_result.pvalue),
            }
        )
        return summary

    non_zero_mask = np.abs(differences) > 1e-8
    if int(non_zero_mask.sum()) < 3:
        summary["test_name"] = None
        summary["message"] = "有效差值不足 3，无法执行 Wilcoxon 检验"
        return summary

    test_result = stats.wilcoxon(
        model_values[non_zero_mask],
        baseline_values[non_zero_mask],
        alternative="less",
        zero_method="wilcox",
    )
    summary.update(
        {
            "test_name": "wilcoxon_signed_rank",
            "statistic": float(test_result.statistic),
            "pvalue": float(test_result.pvalue),
        }
    )
    return summary
