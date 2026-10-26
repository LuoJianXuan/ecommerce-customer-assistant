"""Query 理解（ch04 M3）：口语模糊问法 → 标准问法 + 同义词扩展。

只在检索侧生效（改写词并集喂 dense/BM25），入库侧不拆存多份（需求原文）。
不做指代消解、多轮改写（本章边界）。
LLM 返回非法 JSON 时降级为原查询（检索永远有路可走）。
"""

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

QU_SYSTEM = """你是客服检索的查询改写器。把用户的口语化、模糊问法改写成一个标准检索问句，
并给出最多 3 个同义变体（换说法但意思相同的短查询）。只处理问法本身，不回答问题，
不引入用户没提到的新实体。

以 JSON 返回：{{"normalized": "标准问句", "synonyms": ["变体1", "变体2"]}}，不要输出其他内容。"""

MAX_SYNONYMS = 3


def understand_query(llm, raw_query: str) -> dict:
    """返回 {normalized, synonyms, all_texts}；all_texts=[normalized]+synonyms 供检索侧使用。"""
    prompt = ChatPromptTemplate.from_messages(
        [("system", QU_SYSTEM), ("human", "{q}")]
    ).invoke({"q": raw_query})
    try:
        resp = llm.invoke(prompt.to_messages())
        text = resp.text if hasattr(resp, "text") else str(resp)
        cleaned = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
        data = json.loads(cleaned)
        normalized = str(data["normalized"]).strip() or raw_query
        synonyms = [str(s).strip() for s in data.get("synonyms", []) if str(s).strip()]
    except Exception:
        normalized, synonyms = raw_query, []
    # 去重、限量、剔除与主查询相同的变体
    seen, uniq = {normalized}, []
    for s in synonyms:
        if s not in seen and len(uniq) < MAX_SYNONYMS:
            uniq.append(s)
            seen.add(s)
    return {"normalized": normalized, "synonyms": uniq, "all_texts": [normalized] + uniq}
