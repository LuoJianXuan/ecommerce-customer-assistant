"""LLM 接入：ChatOpenAI 直连上游（OpenAI 协议），单一激活配置。"""

from langchain_openai import ChatOpenAI

from app.config import Settings, get_settings


def _provider_kwargs(settings: Settings) -> dict:
    """返回供应商兼容参数；非 DeepSeek 接口不附加私有字段。"""
    if "api.deepseek.com" not in settings.llm_base_url.lower():
        return {}
    thinking_type = "enabled" if settings.llm_thinking_enabled else "disabled"
    return {"extra_body": {"thinking": {"type": thinking_type}}}


def build_chat_model(settings: Settings) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        temperature=settings.llm_temperature,
        streaming=True,
        **_provider_kwargs(settings),
    )


def get_chat_model() -> ChatOpenAI:
    """FastAPI 依赖入口；测试用 dependency_overrides 换成假模型。"""
    return build_chat_model(get_settings())


def build_intent_model(settings: Settings, *, small: bool = False) -> ChatOpenAI:
    """意图识别/指代消解专用模型：temperature=0（判类要稳定）。

    模型名单独配置：大模型默认沿用 llm_model，小模型用 intent_small_model
    （降级链路先行判断用，仅在 intent_escalation_enabled 时会被构建）。
    """
    model = settings.intent_small_model if small else (settings.intent_model or settings.llm_model)
    return ChatOpenAI(
        model=model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        temperature=0.0,
        **_provider_kwargs(settings),
    )


def build_summary_model(settings: Settings) -> ChatOpenAI:
    """ch07 摘要任务模型（拍板：主力模型 temperature=0——只提炼不编造，零温度防发挥）。"""
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        temperature=0.0,
        **_provider_kwargs(settings),
    )
