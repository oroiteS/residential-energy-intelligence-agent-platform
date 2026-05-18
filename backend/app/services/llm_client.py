from __future__ import annotations

from flask import current_app


def normalize_openai_base_url(value: str | None) -> str:
    raw = (value or "").strip().rstrip("/")
    for suffix in ("/chat/completions", "/responses"):
        if raw.endswith(suffix):
            return raw[: -len(suffix)]
    return raw


def create_chat_model(*, timeout_seconds: int | None = None):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        base_url=normalize_openai_base_url(current_app.config["LLM_BASE_URL"]),
        api_key=current_app.config["LLM_API_KEY"],
        model=current_app.config["LLM_MODEL"],
        temperature=current_app.config["LLM_TEMPERATURE"],
        timeout=timeout_seconds or current_app.config["LLM_TIMEOUT_SECONDS"],
    )


def llm_log_target() -> dict[str, str]:
    return {
        "base_url": normalize_openai_base_url(current_app.config.get("LLM_BASE_URL")),
        "model": current_app.config.get("LLM_MODEL") or "",
    }


def can_use_llm() -> bool:
    if not (
        current_app.config.get("LLM_BASE_URL")
        and current_app.config.get("LLM_API_KEY")
        and current_app.config.get("LLM_MODEL")
    ):
        return False
    try:
        import langchain_core  # noqa: F401
        import langchain_openai  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


def unavailable_reason() -> str:
    if not current_app.config.get("LLM_BASE_URL"):
        return "LLM_BASE_URL_MISSING"
    if not current_app.config.get("LLM_API_KEY"):
        return "LLM_API_KEY_MISSING"
    if not current_app.config.get("LLM_MODEL"):
        return "LLM_MODEL_MISSING"
    try:
        import langchain_core  # noqa: F401
        import langchain_openai  # noqa: F401
    except ModuleNotFoundError:
        return "LANGCHAIN_UNAVAILABLE"
    return "LLM_REQUEST_FAILED"
