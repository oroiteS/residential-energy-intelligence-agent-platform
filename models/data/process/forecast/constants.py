"""预测任务数据构造共享常量。"""

from __future__ import annotations

import re

STEPS_PER_DAY = 96
INPUT_DAYS = 7
TARGET_DAYS = 1
INPUT_LENGTH = INPUT_DAYS * STEPS_PER_DAY
TARGET_LENGTH = TARGET_DAYS * STEPS_PER_DAY
BASE_FEATURE_NAMES = (
    "aggregate",
    "slot_sin",
    "slot_cos",
    "weekday_sin",
    "weekday_cos",
)


def _normalize_label_name(label_name: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z]+", "_", str(label_name)).strip("_").lower()
    if not normalized:
        raise ValueError(f"无法将标签名转换为合法特征名: {label_name!r}")
    return normalized


def build_profile_probability_feature_names(
    label_names: tuple[str, ...] | list[str],
) -> tuple[str, ...]:
    return tuple(
        f"profile_prob_{_normalize_label_name(label_name)}"
        for label_name in label_names
    )


def get_all_feature_names(
    label_names: tuple[str, ...] | list[str],
) -> tuple[str, ...]:
    return BASE_FEATURE_NAMES + build_profile_probability_feature_names(label_names)

__all__ = [
    "BASE_FEATURE_NAMES",
    "build_profile_probability_feature_names",
    "get_all_feature_names",
    "STEPS_PER_DAY",
    "INPUT_DAYS",
    "TARGET_DAYS",
    "INPUT_LENGTH",
    "TARGET_LENGTH",
]
