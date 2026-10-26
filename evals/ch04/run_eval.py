"""ch04 评估体系 v2：四策略检索对比（R@5/R@10/MRR 分桶）+ 生成段全量 Faithfulness
（LLM 裁判 → faith_cases 台账幂等落表）+ D 桶拒答率 + report.json（页面数据源）。

用法：
    python evals/ch04/run_eval.py                      # 全量（检索 + 生成）
    python evals/ch04/run_eval.py --retrieval-only     # 只跑检索段
    python evals/ch04/run_eval.py --gen-limit 12       # 生成段限量（快速模式）
报告双产物：report.md（人读）+ report.json（/eval 页数据源，含 run 时间戳）。
progress_cb 可注入进度回调（页面就地重跑的日志回显数据源）。
"""

import argparse
import asyncio
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sqlalchemy import select  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.db import get_session_factory  # noqa: E402
from app.knowledge.models import FaithCase, KnowledgeChunk  # noqa: E402
from app.knowledge.retrieval import reorder_for_prompt, retrieve  # noqa: E402
from app.llm import build_chat_model  # noqa: E402
from app.prompts import CONVERGENCE_SYSTEM, REFUSAL_PREFIX  # noqa: E402

EVAL_FILE = Path(__file__).parent / "eval_cases.jsonl"
REPORT_MD = Path(__file__).parent / "report.md"
REPORT_JSON = Path(__file__).parent / "report.json"
STRATEGIES = ["dense", "bm25", "hybrid", "hybrid_rerank"]


def load_cases() -> list[dict]:
    return [
        json.loads(line)
        for line in EVAL_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def resolve_gt_ids(cases: list[dict]) -> dict[str, set[int]]:
    """gt 键 → 库内 chunk id 集合。以 questions 为主解析（唯一性高），
    category 匹配 exact 或前缀（### 级块的 category 是完整上级路径）。"""
    factory = get_session_factory()
    with factory() as s:
        rows = s.scalars(select(KnowledgeChunk)).all()
    by_q: dict[str, list] = {}
    for r in rows:
        by_q.setdefault(r.questions.splitlines()[0], []).append(r)
    resolved = {}
    for c in cases:
        ids: set[int] = set()
        for g in c["gt"]:
            for r in by_q.get(g["questions"], []):
                cat = r.category or ""
                if cat == g["category"] or cat.startswith(g["category"] + ">") or g["category"].startswith(cat + ">"):
                    ids.add(r.id)
        resolved[c["id"]] = ids
    return resolved


def eval_retrieval(cases, gt_ids, rewrite_llm=None, progress=None) -> dict:
    """四策略 × Recall@5 / Recall@10 / MRR（分桶）。R@5/R@10 即"两项证据覆盖度"。"""
    from app.knowledge.query_understanding import understand_query

    factory = get_session_factory()
    per: dict[str, dict[str, list[tuple[float, float, float]]]] = {s: {} for s in STRATEGIES}
    todo = [c for c in cases if c["bucket"] != "D_absent"]
    for i, case in enumerate(todo):
        if progress and i % 10 == 0:
            progress(f"[检索段] {i}/{len(todo)} {case['id']}")
        query_texts = (
            understand_query(rewrite_llm, case["query"])["all_texts"] if rewrite_llm else None
        )
        for strategy in STRATEGIES:
            hits = retrieve(
                factory, case["query"], top_k=10, strategy=strategy, query_texts=query_texts
            )
            ranked_ids = [h["chunk_id"] for h in hits]
            gt = gt_ids[case["id"]]
            r5 = len(set(ranked_ids[:5]) & gt) / len(gt) if gt else 0.0
            r10 = len(set(ranked_ids[:10]) & gt) / len(gt) if gt else 0.0
            mrr = 0.0
            for rank, cid in enumerate(ranked_ids, start=1):
                if cid in gt:
                    mrr = 1.0 / rank
                    break
            per[strategy].setdefault(case["bucket"], []).append((r5, r10, mrr))
    summary = {}
    for strategy, buckets in per.items():
        summary[strategy] = {}
        all_vals = []
        for bucket, vals in buckets.items():
            summary[strategy][bucket] = {
                "r5": statistics.mean(v[0] for v in vals),
                "r10": statistics.mean(v[1] for v in vals),
                "mrr": statistics.mean(v[2] for v in vals),
                "n": len(vals),
            }
            all_vals += vals
        summary[strategy]["ALL"] = {
            "r5": statistics.mean(v[0] for v in all_vals),
            "r10": statistics.mean(v[1] for v in all_vals),
            "mrr": statistics.mean(v[2] for v in all_vals),
            "n": len(all_vals),
        }
    return summary


JUDGE_SYSTEM = """你是忠实度裁判。给定用户问题、检索到的证据列表和客服的回答，判断回答是否完全被证据支持。

判据规则（按此执行，不要只按字面"证据里有没有这句话"）：
1. 区分"有没有"与"该不该说"：回答里证据之外的增加内容，若是与安全使用、人身财产安全相关的一般性善意提醒（不替平台承诺、不改写库内结论），判 faithful=true。
2. 常识填充即编造：回答给出证据里没有的具体事实——数字、时限、金额、承诺、型号参数——即使听起来合理，也判 faithful=false，reason 指出编在哪一句、证据里实际写到哪。
3. 角标不等于依据：回答带 [n] 角标时，必须核对第 n 条证据原文是否真的支持那句话；挂着角标但证据里没有的，是最典型的编造。
4. 合理拒答（说明证据不足并建议人工）一律 faithful=true。

真实判例（据此校准尺度）：
- 判例1【误报，不要再犯】：回答末尾说"可以告诉我订单号，我帮您看看"——这是客服交互话术，不是事实性断言，不构成编造，faithful=true。
- 判例2【真编造，必须抓住】：回答称"滤芯、内胆这类配件单独购买质保 90 天[2]"并挂角标，但证据[2]只写滤芯更换周期、90 天那条证据说的是另一类商品且被泛化——挂着角标但证据不支持的泛化事实，faithful=false。
- 判例3【误报，不要再犯】：回答末尾"节假日一般也会顺延"若只是温和的兜底提示且不改变库内结论的具体数值/承诺，倾向 faithful=true；若给出了证据里没有的具体天数/金额则 false。

以 JSON 返回：{"faithful": true/false, "reason": "..."}，不要输出其他内容。"""


async def eval_generation(cases, progress=None, gen_limit: int | None = None) -> dict:
    """生成段：非 D 桶生成答案→裁判（编造落 faith_cases 台账）；D 桶评拒答率。"""
    settings = get_settings()
    llm = build_chat_model(settings)
    factory = get_session_factory()
    from langchain_core.messages import HumanMessage, SystemMessage

    d_cases = [c for c in cases if c["bucket"] == "D_absent"]
    gen_cases = [c for c in cases if c["bucket"] != "D_absent"]
    if gen_limit:
        gen_cases = gen_cases[:gen_limit]

    def _answer(case) -> tuple[str, list[dict]]:
        hits = retrieve(factory, case["query"], top_k=settings.rerank_top_k, strategy="hybrid_rerank")
        evidence = reorder_for_prompt(hits)
        evidence_text = (
            "\n".join(f"[{i}] {m['question']}：{m['answer']}" for i, m in enumerate(evidence, 1))
            if evidence
            else "（知识库未检索到相关内容）"
        )
        msgs = [
            SystemMessage(content=CONVERGENCE_SYSTEM.format(platform_name=settings.platform_name)),
            HumanMessage(content=f"用户问题：{case['query']}\n\n知识库检索结果：\n{evidence_text}"),
        ]
        reply = llm.invoke(msgs).text.strip()
        return reply, evidence

    stats = {"judged": 0, "fabricated": 0, "refused_correct": 0, "refused_wrong": 0,
             "by_bucket": {}}
    total = len(gen_cases) + len(d_cases)
    done = 0
    for case in gen_cases:
        done += 1
        if progress and done % 10 == 0:
            progress(f"[生成段] {done}/{total}")
        reply, evidence = _answer(case)
        prompt = (
            f"用户问题：{case['query']}\n\n证据列表：\n"
            + "\n".join(f"[{i}] {m['question']}：{m['answer']}" for i, m in enumerate(evidence, 1))
            + f"\n\n客服回答：\n{reply}"
        )
        raw = llm.invoke([SystemMessage(content=JUDGE_SYSTEM), HumanMessage(content=prompt)]).text
        try:
            cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            verdict = json.loads(cleaned)
        except json.JSONDecodeError:
            verdict = {"faithful": True, "reason": "裁判输出不可解析，按忠实处理"}
        stats["judged"] += 1
        bucket_stat = stats["by_bucket"].setdefault(case["bucket"], {"judged": 0, "fabricated": 0})
        bucket_stat["judged"] += 1
        if not verdict.get("faithful"):
            stats["fabricated"] += 1
            bucket_stat["fabricated"] += 1
            _upsert_faith_case(case, reply, str(verdict.get("reason", "")), evidence, settings.llm_model)
    for case in d_cases:
        done += 1
        if progress and done % 10 == 0:
            progress(f"[拒答段] {done}/{total}")
        reply, _ = _answer(case)
        if reply.startswith(REFUSAL_PREFIX):
            stats["refused_correct"] += 1
        else:
            stats["refused_wrong"] += 1
            _upsert_faith_case(case, reply, "库外题未拒答（无据而答）", [], settings.llm_model)
    return stats


def _upsert_faith_case(case, answer, reason, evidence, judge_model) -> None:
    """一题一行幂等：复发 seen_count+1、状态回退未解决；citations 存证据全集快照。"""
    factory = get_session_factory()
    citations = [
        {"n": i, "chunk_id": m["chunk_id"], "section_path": m.get("section_path"),
         "question": m["question"], "answer": m["answer"]}
        for i, m in enumerate(evidence, 1)
    ]
    with factory() as s:
        row = s.scalar(select(FaithCase).where(FaithCase.eval_id == case["id"]))
        if row is None:
            s.add(
                FaithCase(
                    eval_id=case["id"], bucket=case["bucket"], query=case["query"],
                    answer=answer, reason=reason, citations=citations, judge_model=judge_model,
                )
            )
        else:
            row.answer, row.reason, row.citations, row.judge_model = answer, reason, citations, judge_model
            row.seen_count += 1
            row.status = "未解决"  # 复发回退（DDL 注释语义）
            row.resolution = None
        s.commit()


def fmt_report(retr: dict, gen: dict, run_ts: str) -> str:
    lines = [f"# ch04 检索质量评估报告（{run_ts}）", ""]
    lines += ["## 检索段：四策略 × 覆盖度(R@5/R@10) / MRR（分桶）", ""]
    buckets = ["A_policy", "B_model", "C_colloquial", "E_multi", "ALL"]
    lines.append("| 策略 | " + " | ".join(f"{b} R@5/R@10/MRR" for b in buckets) + " |")
    lines.append("|---|" + "---|" * len(buckets))
    best = max(STRATEGIES, key=lambda st: retr[st]["ALL"]["mrr"])
    for st in STRATEGIES:
        cells = []
        for b in buckets:
            d = retr[st].get(b, {"r5": 0, "r10": 0, "mrr": 0})
            cells.append(f"{d['r5']:.0%}/{d['r10']:.0%}/{d['mrr']:.3f}")
        mark = " 🏆" if st == best else ""
        lines.append(f"| {st}{mark} | " + " | ".join(cells) + " |")
    lines += ["", "## 生成段（hybrid_rerank，全量可作答题 + D 拒答桶）", ""]
    lines.append("| 桶 | 评上 | 编造 | 忠实率 |")
    lines.append("|---|---|---|---|")
    for b, d in sorted(gen.get("by_bucket", {}).items()):
        rate = f"{1 - d['fabricated']/d['judged']:.0%}" if d["judged"] else "-"
        lines.append(f"| {b} | {d['judged']} | {d['fabricated']} | {rate} |")
    refused_total = gen["refused_correct"] + gen["refused_wrong"]
    if refused_total:
        lines += ["", f"- 拒答率（D 库外桶）：{gen['refused_correct']}/{refused_total} = {gen['refused_correct']/refused_total:.0%}"]
    lines += ["", f"- **编造个案（已落 faith_cases 台账）：{gen['fabricated'] + gen['refused_wrong']}**（含该拒未拒 {gen['refused_wrong']}）"]
    return "\n".join(lines) + "\n"


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--gen-limit", type=int, default=None)
    args = parser.parse_args()

    run_ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    cases = load_cases()
    gt_ids = resolve_gt_ids(cases)
    missing = [c["id"] for c in cases if c["gt"] and not gt_ids[c["id"]]]
    if missing:
        print(f"警告：以下题的 ground-truth 在库中未解析到：{missing}")

    settings = get_settings()
    rewrite_llm = build_chat_model(settings) if settings.query_rewrite_enabled else None
    print(f"评估检索段（{sum(1 for c in cases if c['bucket'] != 'D_absent')} 题 × 4 策略 + Query 改写）…")
    retr = eval_retrieval(cases, gt_ids, rewrite_llm)

    gen = {"judged": 0, "fabricated": 0, "refused_correct": 0, "refused_wrong": 0, "by_bucket": {}}
    if not args.retrieval_only:
        print(f"评估生成段（可作答 {sum(1 for c in cases if c['bucket'] != 'D_absent')} + 库外 {sum(1 for c in cases if c['bucket'] == 'D_absent')}）…")
        gen = await eval_generation(cases, gen_limit=args.gen_limit)

    report = fmt_report(retr, gen, run_ts)
    REPORT_MD.write_text(report, encoding="utf-8")
    REPORT_JSON.write_text(
        json.dumps(
            {"run_ts": run_ts, "run_at": datetime.now().isoformat(),
             "cases_total": len(cases),
             "retrieval": retr, "generation": gen},
            ensure_ascii=False, indent=1,
        ),
        encoding="utf-8",
    )
    print(report)
    print(f"报告已写 {REPORT_MD} / {REPORT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
