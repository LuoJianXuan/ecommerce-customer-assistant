"""指代消解 + Query 改写（ch06 M2）：带对话历史把「它能退吗」补全成独立完整问句。

规则（需求原文）：
- 指代/省略主语 → 结合历史补全成不依赖上下文也能看懂的完整问题；
- 问题本身已完整、指代已明确 → 原样透传，不强行改写；
- 口语模糊问法顺带归一成标准问法（Query 改写与指代消解一并处理）。

降级：无历史零调用直接透传；LLM 输出解析失败也透传原句——下游永远有路可走。
模型与意图识别共用 build_intent_model（temperature=0，判类要稳定）。
"""

from dataclasses import dataclass

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from app.agent.parsing import ai_text, extract_json_object

COREF_SYSTEM_PROMPT = """你是客服对话的指代消解与问句改写器。给你最近几轮对话和用户最新一句话，
把最新这句话处理成一个不依赖上下文也能独立看懂的完整问句。

规则：
- 句中有指代（它/这个/那单/他/上面那个/第一个）或省略了主语：结合对话历史补全成明确指代。
  例：历史「用户：我买的电饭煲什么时候到」+ 最新「它能退吗」→「我买的电饭煲能退货吗」
- 补全不能偷换所指：查包裹/快递语境里的「单号」是「快递单号」，不要改写成「订单单号」，
  也不要把用户没说的「订单」塞进问句。
- 问题本身已完整、指代已明确：原样返回这句话，不要改写、不要扩写、不要加语气词。
- 口语模糊问法顺带归一成标准问法：「咋还没到」→「我的订单为什么还没发货」。
- 只做改写不回答问题；不引入用户没提到的新信息（不编造订单号、商品名）。

只输出 JSON：{{"resolved_query": "补全或改写后的完整问句", "rewritten": true或false}}。
做了指代补全或改写为 true，原样透传为 false。不要输出其他内容。"""


@dataclass
class CorefResult:
    resolved_query: str
    rewritten: bool = False
    degraded: bool = False  # 解析失败降级透传（原句保底）


def parse_coref_json(raw: str) -> CorefResult | None:
    """JSON 提取 → 字段校验；非法返回 None。

    rewritten 容错布尔陷阱：模型回字符串 "false" 时按内容判，不能 bool() 强转。
    """
    data = extract_json_object(raw)
    if data is None:
        return None
    resolved = str(data.get("resolved_query", "")).strip()
    if not resolved:
        return None
    raw_rw = data.get("rewritten", False)
    if isinstance(raw_rw, str):
        rewritten = raw_rw.strip().lower() in ("true", "1", "yes", "是")
    else:
        rewritten = bool(raw_rw)
    return CorefResult(resolved, rewritten)


def _transcript(history: list) -> str:
    """历史消息 → 「用户：/客服：」转写；空内容与工具消息不进（对消解无用且占窗口）。

    ch07 起窗口大小由调用方控制（resolve_refs 传层1滑窗），这里不再自截条数。
    """
    lines = []
    for m in history:
        content = str(m.content or "").strip()
        if not content:
            continue
        if isinstance(m, HumanMessage):
            lines.append(f"用户：{content}")
        elif isinstance(m, AIMessage):
            lines.append(f"客服：{content}")
    return "\n".join(lines)


async def aresolve(llm, history: list, utterance: str, background: str = "") -> CorefResult:
    """指代消解+改写入口。无历史零调用透传；解析失败降级透传原句。

    background（ch07）：早期对话梗概一行，作为消解的远处背景（层3 内容的替身）。
    """
    transcript = _transcript(history)
    if background:
        transcript = (transcript + "\n" if transcript else "") + f"[早期对话梗概]{background}"
    if not transcript:
        return CorefResult(utterance)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", COREF_SYSTEM_PROMPT),
            ("human", "最近对话：\n{history}\n\n最新用户消息：{q}"),
        ]
    ).invoke({"history": transcript, "q": utterance})
    try:
        resp = await llm.ainvoke(prompt.to_messages())
        parsed = parse_coref_json(ai_text(resp))
    except Exception:
        parsed = None
    if parsed is None:
        return CorefResult(utterance, degraded=True)
    return parsed
