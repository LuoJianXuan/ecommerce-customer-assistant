"""测试替身：可脚本化的聊天假 LLM 与桩工具注册表。

FakeChatLLM：bind_tools 记录绑定；ainvoke（第一调，工具决策）返回脚本化 AIMessage
并记录看到的 messages；astream（第二调，最终回答）按 2 字符/块流式吐 stream_text。
"""

from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.tools import tool

from app.tools.base import ToolRegistry


class FakeChatLLM:
    def __init__(self, first: AIMessage | None = None, stream_text: str = "这是流式回复内容"):
        self.first = first or AIMessage(content="")
        self.stream_text = stream_text
        self.retry_results: list[AIMessage] = []  # 第 2 次及以后的 ainvoke 按序消费（收敛重试测试用）
        self.seen: list[list] = []  # 每次 ainvoke 看到的 messages
        self.stream_seen: list[list] = []  # 每次 astream 看到的 messages
        self.bound_tools: list | None = None
        self.bind_kwargs: dict = {}

    def bind_tools(self, tools, **kwargs):
        self.bound_tools = list(tools)
        self.bind_kwargs = kwargs
        return self

    async def ainvoke(self, messages, config=None):
        self.seen.append(list(messages))
        if len(self.seen) > 1 and self.retry_results:
            return self.retry_results.pop(0)
        return self.first

    async def astream(self, messages):
        self.stream_seen.append(list(messages))
        for i in range(0, len(self.stream_text), 2):
            yield AIMessageChunk(content=self.stream_text[i : i + 2])


@tool
def stub_lookup(order_id: str) -> str:
    """桩查询工具（测试用），返回固定 JSON。"""
    return '{"stub": true, "order_id": "' + order_id + '"}'


def make_stub_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(stub_lookup)
    return reg


class FakeStructuredLLM:
    """结构化提取假模型：with_structured_output 返回固定结果的 runner，记录收到的 prompt。"""

    def __init__(self, result):
        self.result = result
        self.calls: list = []

    def with_structured_output(self, schema, **kwargs):
        fake = self

        class _Runner:
            async def ainvoke(self, prompt_value, config=None):
                fake.calls.append(prompt_value)
                return fake.result

        return _Runner()
