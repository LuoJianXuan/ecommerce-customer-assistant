"""意图识别正式版（ch06 M1）：四件套 prompt + JSON 容错重试 + 「其他」兜底 + 可选两级降级。

四件套（需求原文）：
① 七类业务意图 + 「其他」兜底，枚举成选择题让模型选；
② 强制 JSON 输出，字段只有 intent 和 confidence 两个；
③ 边界 few-shot 样例（ch05 错例集中处：售后↔退款退货、物流↔订单发货歧义）；
④ 「其他」兜底规则：拿不准就归它，不硬塞业务意图。

降级链路（INTENT_ESCALATION_ENABLED，默认关）：小模型给判断和置信度，
confidence 低于阈值或解析失败才升级大模型重判一次，以大模型为准（需求原文）。
单模型内解析失败同参重试 1 次；仍失败降级「其他」+ confidence 0——分流永远有路可走。
"""

import re
from dataclasses import dataclass

from langchain_core.prompts import ChatPromptTemplate

from app.agent.parsing import ai_text, extract_json_object

INTENTS = ("物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "闲聊", "其他")
FALLBACK_INTENT = "其他"

# 选择题字母 ↔ 类别名：模型偶尔回字母（实测 deepseek 温度 0 下确定性复读，重试救不回），
# 解析侧直接映射，prompt 侧同时要求写中文全称（双保险）。
INTENT_LETTER = {
    "A": "物流", "B": "订单", "C": "商品咨询", "D": "退款退货",
    "E": "售后", "F": "投诉", "G": "闲聊", "H": "其他",
}

INTENT_SYSTEM_PROMPT = """你是电商客服的意图分类器。判断用户这句话想干什么，从八个类别里选一个：

A. 物流——查包裹运输进度：到哪了、快递单号、什么时候能送到、卡在中转
B. 订单——订单本身的状态与操作：订单什么状态、还没发货、取消订单、改收货地址
C. 商品咨询——商品规格参数、库存、推荐对比、会员权益与规则说明
D. 退款退货——要退钱或退物：退款、退货、仅退款、退货退款、七天无理由、退运费谁出
E. 售后——不退钱退物的售后诉求：换货、维修、补发、降价补偿/价保、开发票
F. 投诉——表达不满、要求投诉、升级处理、投诉客服或物流
G. 闲聊——与购物业务无关的问候寒暄、随便聊聊
H. 其他——以上都不匹配，或拿不准属于哪一类

边界示例（先看这些再判断）：
- 「退货运费谁出」→ D（要退货，连带运费争议）
- 「换货流程」→ E（换货不退钱）
- 「发票怎么开」→ E（开票是售后事项）
- 「七天无理由是什么规则」→ D（问的是退货政策，不是商品知识）
- 「净水器滤芯坏了能换一个吗」→ E（换新是换货诉求）；「滤芯有单独卖的吗」→ C（问商品购买）
- 「我的订单到哪了」→ A（问包裹位置）
- 「订单还没发货什么情况」→ B（问订单发货状态）
- 「我投诉过的那笔订单后来处理得怎么样」→ B（问订单处理进展，不是在提新投诉）
- 「帮我建个售后工单」→ H（明确要求建工单的交主力 Agent 走建单流程，不进业务子流程）
- 「帮我写一篇作文」→ H（超出客服业务）
- 「地球为什么是圆的」→ H（百科知识问题超出业务范围，不算闲聊）

判断规则：
- 只看这句话表达的最主要意图；两类都说得通且没有更明确信号时选 H，不要硬猜。
- 用户明确要求「建工单／转人工／找人工客服」时，即使句中提到订单/售后/退款等
  业务字样，也选 H「其他」（由主力 Agent 处理建单与转接，不走业务分流）；
  但句子里带着明确投诉情绪或「投诉」字眼的，仍选 F「投诉」（投诉出口自带转人工/建工单按钮）。
- 「闲聊」仅限寒暄问候、情绪交流（不需要回答具体知识内容）；超出电商客服业务
  范围的实际问题（写作、股票行情、订票、百科知识等）一律归 H「其他」。

只输出 JSON：{{"intent": "类别名", "confidence": 0到1的小数}}，两个字段都必填。
intent 必须写中文类别全称（如「物流」），不要写字母代号，不要输出其他内容。
输出示例：{{"intent": "物流", "confidence": 0.92}}"""


@dataclass
class IntentResult:
    intent: str
    confidence: float
    escalated: bool = False
    parse_failed: bool = False


def parse_intent_json(raw: str) -> tuple[str, float] | None:
    """JSON 提取 → 校验；非法返回 None（由调用方重试/降级）。

    容错：模型偶尔回选择题字母（含「D. 退款退货」混合式，小写兼容）→ 映射回类别名；
    confidence >1 且 ≤100 视为百分数换算；其余越界值钳制到 [0,1]。
    """
    data = extract_json_object(raw)
    if data is None:
        return None
    intent = str(data.get("intent", "")).strip()
    if not intent:
        return None
    m = re.match(r"^([A-Ha-h])[.、．:：]?\s*(.*)$", intent)
    if m:
        # 字母代号（可带类别名尾巴）：有名用名，没名按字母映射
        rest = m.group(2).strip()
        intent = rest if rest in INTENTS else INTENT_LETTER[m.group(1).upper()]
    if intent not in INTENTS:
        return None
    try:
        conf = float(data.get("confidence"))
    except (TypeError, ValueError):
        return None
    if 1 < conf <= 100:
        conf = conf / 100
    conf = min(max(conf, 0.0), 1.0)
    return intent, round(conf, 4)


def _prompt_messages(query: str) -> list:
    return (
        ChatPromptTemplate.from_messages(
            [("system", INTENT_SYSTEM_PROMPT), ("human", "{q}")]
        )
        .invoke({"q": query})
        .to_messages()
    )


async def _classify_once(llm, query: str) -> IntentResult:
    """单模型判一次；解析失败同参重试一次；仍失败降级「其他」。"""
    messages = _prompt_messages(query)
    for _ in range(2):
        resp = await llm.ainvoke(messages)
        parsed = parse_intent_json(ai_text(resp))
        if parsed is not None:
            return IntentResult(intent=parsed[0], confidence=parsed[1])
    return IntentResult(intent=FALLBACK_INTENT, confidence=0.0, parse_failed=True)


async def aclassify(llm, query: str, *, small_llm=None, threshold: float = 0.6) -> IntentResult:
    """意图分类入口。

    small_llm 为空：单模型直判（默认）。给了：小模型先判，confidence 低于阈值
    或解析失败 → llm（大模型）重判一次，以大模型为准。
    """
    if small_llm is None:
        return await _classify_once(llm, query)
    first = await _classify_once(small_llm, query)
    if first.confidence >= threshold and not first.parse_failed:
        return first
    second = await _classify_once(llm, query)
    second.escalated = True
    return second
