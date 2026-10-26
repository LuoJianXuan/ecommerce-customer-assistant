"""工具路由评估集跑分：真实模型第一调（bind_tools）选中工具 vs 标注。

用法：
    python evals/run_routing_eval.py                                  # 当前生产配置
    python evals/run_routing_eval.py --prompt old --temperature 0.7   # 对照实验
    LLM_MODEL=qwen-plus python evals/run_routing_eval.py --repeats 3  # 多次跑看分布

只评路由（模型决定调什么工具），不执行工具、不写库。
用例间并发（信号量限 5），repeats 次重复用于观察温度带来的稳定性差异。
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.db import get_session_factory  # noqa: E402
from app.llm import build_chat_model  # noqa: E402
from app.prompts import CUSTOMER_SERVICE_SYSTEM  # noqa: E402
from app.tools import build_default_registry  # noqa: E402

EVAL_FILE = Path(__file__).parent / "tool_routing.jsonl"

# v1 旧版 prompt（ch02-M3 首版，qwen-plus 在其上跑出 6/9 的那次基线），存档供对照实验
CUSTOMER_SERVICE_SYSTEM_V1 = """你是{platform_name}电商平台的智能客服「小云」，只处理本平台购物相关咨询。

## 角色设定
- 用简体中文口语化回答，亲切、简洁，优先解决用户当前的问题。
- 默认回答不超过 3 句话，除非用户明确要求展开。

## 工具使用
- 订单状态/金额/商品价格库存/物流轨迹类问题，调用对应查询工具后再回答，不要编造。
- 政策、规则、流程类问题先查 FAQ；查不到相关内容就如实说明"知识库里没有找到"，不要瞎编答案。
- 用户情绪激烈、诉求超出你的能力、或明确要求人工时，创建工单转人工。
- 工具返回结果后用自然语言组织回答，别把原始 JSON 直接丢给用户。

## 行为约束
- 订单状态、物流细节、库存等实时信息以工具返回为准，工具查不到就直说，绝不编造。
- 不承诺任何具体赔付金额或比例，涉及赔付一律回复"将由专员跟进核实"。
- 用户提出退款、退货、换货等售后诉求时，先礼貌引导用户提供订单号。
- 遇到辱骂、人身安全、法律纠纷或超出能力范围的情况，建议转人工客服。
- 拒绝回答与平台购物无关的问题（如写作、代码、医疗建议），礼貌拉回正题。"""

PROMPT_VARIANTS = {"current": CUSTOMER_SERVICE_SYSTEM, "old": CUSTOMER_SERVICE_SYSTEM_V1}


async def eval_once(llm, system, cases, semaphore) -> list[str | None]:
    async def one(case) -> str | None:
        async with semaphore:
            ai = await llm.ainvoke([system, HumanMessage(content=case["text"])])
            return ai.tool_calls[0]["name"] if ai.tool_calls else None

    return list(await asyncio.gather(*(one(c) for c in cases)))


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", choices=PROMPT_VARIANTS, default="current")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()

    settings = get_settings()
    cases = [json.loads(line) for line in EVAL_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    registry = build_default_registry(get_session_factory(), conversation_id=0)
    # 与生产链路对齐：第一调带客服 system prompt（含工具使用指引）
    llm = build_chat_model(settings).bind_tools(registry.all(), temperature=args.temperature)
    system = SystemMessage(content=PROMPT_VARIANTS[args.prompt].format(platform_name=settings.platform_name))
    semaphore = asyncio.Semaphore(5)

    print(
        f"模型={settings.llm_model}  prompt={args.prompt}  "
        f"temperature={args.temperature}  repeats={args.repeats}"
    )

    runs = [await eval_once(llm, system, cases, semaphore) for _ in range(args.repeats)]
    accuracies = [
        sum(r[i] == cases[i]["expected_tool"] for i in range(len(cases))) / len(cases)
        for r in runs
    ]

    print(f"{'用户输入':<24} {'期望':<16} {'命中':<8} 出现过的实际值")
    print("-" * 80)
    for i, case in enumerate(cases):
        hits = sum(runs[j][i] == case["expected_tool"] for j in range(args.repeats))
        actuals = sorted({str(r[i]) for r in runs})
        print(
            f"{case['text']:<24} {str(case['expected_tool']):<16} "
            f"{hits}/{args.repeats:<6} {' | '.join(actuals)}"
        )
    print("-" * 80)
    acc_str = "  ".join(f"{a:.0%}" for a in accuracies)
    print(f"各次准确率: {acc_str}   均值 {sum(accuracies) / len(accuracies):.0%}")
    return 0 if all(a == 1.0 for a in accuracies) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
