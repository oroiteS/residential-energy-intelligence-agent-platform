"""LSTM 预测任务常量。"""

from __future__ import annotations

INPUT_LENGTH = 288
TARGET_LENGTH = 96
ALL_FEATURE_NAMES = (
    "aggregate",
    "active_appliance_count",
    "burst_event_count",
    "slot_sin",
    "slot_cos",
    "weekday_sin",
    "weekday_cos",
)
