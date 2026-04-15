"""LSTM 预测任务常量。"""

from __future__ import annotations

STEPS_PER_DAY = 96
INPUT_DAYS = 7
TARGET_DAYS = 1
INPUT_LENGTH = INPUT_DAYS * STEPS_PER_DAY
TARGET_LENGTH = TARGET_DAYS * STEPS_PER_DAY
ALL_FEATURE_NAMES = (
    "aggregate",
    "active_appliance_count",
    "burst_event_count",
    "slot_sin",
    "slot_cos",
    "weekday_sin",
    "weekday_cos",
)
