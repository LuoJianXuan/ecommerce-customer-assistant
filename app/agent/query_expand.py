"""ch06 Query 扩写（检索侧，现查现用）：一句问题 → 多条侧重不同的检索查询。

边界（需求原文）：
- 只对退款退货/售后这类高频、容错低的核心场景做（aftersale 子流程政策检索前触发），
  简单 FAQ 不扩写；
- 扩写只在查询时发生，入库侧不拆存多份（库里知识只留一份）。

强制 JSON 输出，字段只有 queries 一个数组；解析失败降级为 [原问题] 单查询。
"""

from langchain_core.prompts import ChatPromptTemplate

from app.agent.parsing import ai_text, extract_json_object

MAX_QUERIES = 4  # 原始问法 + 最多 3 个侧重变体

EXPAND_SYSTEM_PROMPT = """你是客服政策检索的查询扩写器。给你一个退款/售后类问题，生成
3-4 条侧重不同的检索查询，用于在知识库里检索政策条款。

规则：
- 第 1 条保留原问题（或其标准问法）；
- 其余各条侧重不同：政策条件侧重（可否退换、时间窗口）、流程时效侧重（怎么申请、多久到账）、
  费用侧重（运费谁出、扣除多少）等，只挑和问题相关的侧重；
- 只做查询扩写不回答问题；不引入用户没提到的信息；每条是一个独立短句。

只输出 JSON：{{"queries": ["查询1", "查询2", "查询3"]}}，queries 是字符串数组，
不要输出其他内容。"""


def parse_expand_json(raw: str) -> list[str] | None:
    """JSON 提取 → 校验非空字符串数组 → 去重保序 → 限量；非法返回 None。"""
    data = extract_json_object(raw)
    if data is None or not isinstance(data.get("queries"), list):
        return None
    seen, uniq = set(), []
    for item in data["queries"]:
        q = str(item).strip()
        if q and q not in seen:
            uniq.append(q)
            seen.add(q)
    uniq = uniq[:MAX_QUERIES]
    return uniq or None


async def aexpand(llm, query: str) -> list[str]:
    """扩写入口。空查询/解析失败一律降级为 [原问题]——多路检索退化为单路，永远可走。"""
    if not query.strip():
        return [query]
    prompt = ChatPromptTemplate.from_messages(
        [("system", EXPAND_SYSTEM_PROMPT), ("human", "{q}")]
    ).invoke({"q": query})
    try:
        resp = await llm.ainvoke(prompt.to_messages())
        parsed = parse_expand_json(ai_text(resp))
    except Exception:
        parsed = None
    return parsed if parsed else [query]
