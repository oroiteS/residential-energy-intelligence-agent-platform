"""XGBoost 分类任务常量。"""

from __future__ import annotations

SEQUENCE_LENGTH = 96

BLOCK_SIZE = 12
NUM_BLOCKS = SEQUENCE_LENGTH // BLOCK_SIZE
AGGREGATE_COLUMNS = tuple(f"aggregate_{index:03d}" for index in range(SEQUENCE_LENGTH))

TABULAR_FEATURE_NAMES = (
    "full_mean",
    "full_std",
    "full_min",
    "full_max",
    "full_range",
    "load_factor",
    "p10",
    "p25",
    "p50",
    "p75",
    "p90",
    "day_mean",
    "night_mean",
    "day_std",
    "night_std",
    "day_night_diff",
    "day_night_ratio",
    "night_day_ratio",
    "morning_mean",
    "daytime_mean",
    "evening_mean",
    "overnight_mean",
    "peak_value",
    "peak_slot_index_norm",
    "valley_value",
    "valley_slot_index_norm",
    "peak_to_mean_ratio",
    "ramp_abs_mean",
    "ramp_abs_std",
    "ramp_abs_max",
    "ramp_up_mean",
    "ramp_down_mean",
    "high_load_ratio",
    "low_load_ratio",
    "weekday_sin",
    "weekday_cos",
    "is_weekend",
    "block_mean_00",
    "block_mean_01",
    "block_mean_02",
    "block_mean_03",
    "block_mean_04",
    "block_mean_05",
    "block_mean_06",
    "block_mean_07",
)

__all__ = [
    "AGGREGATE_COLUMNS",
    "BLOCK_SIZE",
    "NUM_BLOCKS",
    "SEQUENCE_LENGTH",
    "TABULAR_FEATURE_NAMES",
]
