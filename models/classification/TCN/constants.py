"""TCN 分类任务常量。"""

from __future__ import annotations

LABELS = [
    "day_high_night_low",
    "day_low_night_high",
    "all_day_high",
    "all_day_low",
]

SEQUENCE_LENGTH = 96
FEATURE_NAMES = (
    "aggregate",
    "slot_sin",
    "slot_cos",
    "weekday_sin",
    "weekday_cos",
)
INPUT_CHANNELS = len(FEATURE_NAMES)
