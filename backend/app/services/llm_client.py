from __future__ import annotations

from flask import current_app


def normalize_openai_base_url(value: str | None) -> str:
    """规范化 OpenAI 兼容接口地址。

    用户可能填写根地址，也可能填写 /chat/completions 或 /responses 端点；
    这里统一转换为 LangChain ChatOpenAI 需要的 base_url。
    """

    raw = (value or "").strip().rstrip("/")
    for suffix in ("/chat/completions", "/responses"):
        if raw.endswith(suffix):
            return raw[: -len(suffix)]
    return raw


def create_chat_model(*, timeout_seconds: int | None = None):
    """创建 LangChain ChatOpenAI 客户端。

    ChatOpenAI 是 LangChain 对 OpenAI 兼容聊天接口的封装。
    本项目不在这里直接拼 HTTP 请求，而是把 base_url、api_key、model、
    temperature 和 timeout 交给 ChatOpenAI，由它负责和大模型服务通信。
    """

    from langchain_openai import ChatOpenAI

    # temperature 越低，回复越稳定、越少发散。
    # 本项目用于报告和用电建议，更需要可控表达，所以默认配置较低。
    return ChatOpenAI(
        base_url=normalize_openai_base_url(current_app.config["LLM_BASE_URL"]),
        api_key=current_app.config["LLM_API_KEY"],
        model=current_app.config["LLM_MODEL"],
        temperature=current_app.config["LLM_TEMPERATURE"],
        timeout=timeout_seconds or current_app.config["LLM_TIMEOUT_SECONDS"],
    )


def llm_log_target() -> dict[str, str]:
    """返回日志中可安全记录的 LLM 目标信息。"""

    return {
        "base_url": normalize_openai_base_url(current_app.config.get("LLM_BASE_URL")),
        "model": current_app.config.get("LLM_MODEL") or "",
    }


def can_use_llm() -> bool:
    """判断当前环境是否可以调用 LLM。

    需要同时具备 base_url、api_key、model 配置和 LangChain 相关依赖。
    如果任一条件不满足，智能问答和报告摘要会走本地降级逻辑。
    """

    # 配置缺失时不尝试创建 ChatOpenAI。
    # 这样可以避免接口调用时才暴露大模型配置错误。
    if not (
        current_app.config.get("LLM_BASE_URL")
        and current_app.config.get("LLM_API_KEY")
        and current_app.config.get("LLM_MODEL")
    ):
        return False
    try:
        # 这里做轻量依赖探测。
        # langchain_core 提供 prompt/message/chain 等基础抽象；
        # langchain_openai 提供 OpenAI 兼容聊天模型封装。
        import langchain_core  # noqa: F401
        import langchain_openai  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


def unavailable_reason() -> str:
    """返回 LLM 不可用的主要原因。"""

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
