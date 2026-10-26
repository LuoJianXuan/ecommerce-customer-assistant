"""ch06 分流器共用解析小件：LLM 文本输出 → JSON 对象。

intent/coref/query_expand 三个 prompt 模块的围栏清洗+散文截取形状完全一致，
收敛到这里各自只留字段校验（code review 评注：Duplicated Code）。
"""

import json
import re


def ai_text(resp) -> str:
    """LLM 应答 → 文本（真 ChatOpenAI 与测试 fake 都兼容）。"""
    return resp.text if hasattr(resp, "text") else str(resp)


def extract_json_object(raw: str) -> dict | None:
    """``` 围栏清洗 → 散文包裹时截取 {..} 段 → json.loads；非法或非对象返回 None。"""
    text = str(raw or "").strip()
    if not text:
        return None
    cleaned = re.sub(r"^```(json)?|```$", "", text, flags=re.MULTILINE).strip()
    if not cleaned.startswith("{"):
        s, e = cleaned.find("{"), cleaned.rfind("}")
        if s == -1 or e <= s:
            return None
        cleaned = cleaned[s : e + 1]
    try:
        data = json.loads(cleaned)
    except Exception:
        return None
    return data if isinstance(data, dict) else None
