"""售后诉求提取服务：with_structured_output 把口语描述抽成固定字段。

method 显式用 "function_calling"：新版默认 json_schema 走 OpenAI 原生 Structured Outputs API，
百炼等第三方 OpenAI 兼容端点支持不一；tools 路线对 qwen 系最稳。本接口非流式，
不触犯百炼"tools 不能与 stream 同用"的限制。
"""

from langchain_core.prompts import ChatPromptTemplate

from app.schemas import AfterSaleInfo

EXTRACT_INSTRUCTIONS = """你是电商售后工单的信息抽取器。从下面的用户描述中抽取固定字段：
- order_id：订单号，用户提到才填，否则留空
- appeal_type：诉求类型，只能是 退款/退货退款/换货/补发/维修/降价补偿/投诉/其他 之一
- expected_solution：期望方案，用用户原话简短概括，没有则留空
只输出结构化字段，不要解释。

用户描述：
{text}"""

extract_prompt = ChatPromptTemplate.from_messages([("human", EXTRACT_INSTRUCTIONS)])


class ExtractService:
    def __init__(self, llm):
        self.structured = llm.with_structured_output(
            AfterSaleInfo, method="function_calling"
        )

    async def extract(self, text: str) -> AfterSaleInfo:
        messages = extract_prompt.invoke({"text": text}).to_messages()
        return await self.structured.ainvoke(messages)
