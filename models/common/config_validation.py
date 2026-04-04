"""配置文件结构校验工具。"""

from __future__ import annotations

from collections.abc import Mapping
from difflib import get_close_matches
from pathlib import Path
from typing import Any


def _format_unknown_key_error(
    *,
    config_path: Path,
    scope: str,
    unknown_keys: set[str],
    allowed_keys: set[str],
) -> str:
    suggestions: list[str] = []
    for key in sorted(unknown_keys):
        matches = get_close_matches(key, sorted(allowed_keys), n=1, cutoff=0.6)
        if matches:
            suggestions.append(f"{key} -> {matches[0]}")

    error = (
        f"配置错误：{config_path} 的 {scope} 存在未识别字段: "
        f"{sorted(unknown_keys)}；允许字段为: {sorted(allowed_keys)}"
    )
    if suggestions:
        error += f"；可能想写的是: {suggestions}"
    return error


def require_mapping(
    raw_config: Mapping[str, Any],
    section_name: str,
    *,
    config_path: Path,
) -> Mapping[str, Any]:
    section = raw_config.get(section_name)
    if section is None:
        raise ValueError(f"配置错误：{config_path} 缺少 `{section_name}` 段")
    if not isinstance(section, Mapping):
        raise ValueError(f"配置错误：{config_path} 的 `{section_name}` 必须是键值映射")
    return section


def validate_config_schema(
    raw_config: Any,
    *,
    config_path: Path,
    allowed_top_level_keys: set[str],
    allowed_section_keys: dict[str, set[str]],
) -> dict[str, Mapping[str, Any]]:
    if not isinstance(raw_config, Mapping):
        raise ValueError(f"配置错误：{config_path} 顶层必须是键值映射")

    unknown_top_level_keys = set(raw_config).difference(allowed_top_level_keys)
    if unknown_top_level_keys:
        raise ValueError(
            _format_unknown_key_error(
                config_path=config_path,
                scope="顶层",
                unknown_keys=unknown_top_level_keys,
                allowed_keys=allowed_top_level_keys,
            )
        )

    sections: dict[str, Mapping[str, Any]] = {}
    for section_name, allowed_keys in allowed_section_keys.items():
        section = require_mapping(raw_config, section_name, config_path=config_path)
        unknown_section_keys = set(section).difference(allowed_keys)
        if unknown_section_keys:
            raise ValueError(
                _format_unknown_key_error(
                    config_path=config_path,
                    scope=f"`{section_name}` 段",
                    unknown_keys=unknown_section_keys,
                    allowed_keys=allowed_keys,
                )
            )
        sections[section_name] = section

    return sections
