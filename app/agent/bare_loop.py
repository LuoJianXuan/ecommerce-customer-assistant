"""祛魅热身（ch05 M1）：不依赖任何编排框架，手写最裸的 Agent 循环。

看清它的本质：一个带工具的 while 循环——
  调 LLM → 有 tool_calls 就执行、把结果喂回去 → 没有 tool_calls 就收敛出答案。
加一个步数上限当保险丝，除此之外没有魔法。跑通之后再用 LangGraph 重构成图。

用法：python -m app.agent.bare_loop "订单 1001 的状态怎么样，它的物流到哪了"
"""

import asyncio
import sys

from langchain_core.messages import HumanMessage, SystemMessage

from app.config import get_settings
from app.db import get_session_factory
from app.llm import build_chat_model
from app.tools import build_default_registry
from app.tools.engine import execute as engine_execute

MAX_STEPS = 5

SYSTEM = "你是云优选电商平台的智能客服小云，用简体中文口语化回答，工具返回的数据如实转述、不编造。"


async def run_bare_agent(question: str, conversation_id: int = 1) -> str:
    settings = get_settings()
    llm = build_chat_model(settings)
    registry = build_default_registry(get_session_factory(), conversation_id, llm)
    bound = llm.bind_tools(registry.all())

    messages = [SystemMessage(content=SYSTEM), HumanMessage(content=question)]
    for step in range(1, MAX_STEPS + 1):
        ai = await bound.ainvoke(messages)
        messages.append(ai)
        if not ai.tool_calls:
            print(f"\n[step {step}] 模型不再调工具 → 收敛出答案：\n{ai.text}")
            return ai.text
        names = [tc["name"] for tc in ai.tool_calls]
        print(f"[step {step}] 模型请求工具: {names}")
        for tc in ai.tool_calls:
            out = await engine_execute(
                tc, registry,
                conversation_id=conversation_id,
                session_factory=get_session_factory(),
                default_timeout=settings.tool_timeout_seconds,
                max_retries=settings.tool_max_retries,
                allow_pending=False,  # 裸循环无确认 UI：写操作一律拒绝（ch08 引擎门）
            )
            result = out.message
            messages.append(result)
            preview = result.content.replace("\n", " ")[:90]
            print(f"         执行 {tc['name']} → {preview}")
    print(f"\n[stop ] 达到步数上限 {MAX_STEPS}，强制收敛")
    return ""


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "订单 1001 的状态怎么样，它的物流到哪了"
    print(f"问题：{q}\n" + "-" * 60)
    asyncio.run(run_bare_agent(q))
