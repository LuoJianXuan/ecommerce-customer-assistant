"""Read-only full RAG audit against the current project; outputs are versioned."""
import argparse
import asyncio
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
parser.add_argument("--cases", type=Path, default=Path(__file__).resolve().parent/"releases/2026-10-26/eval_cases.jsonl")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
ROOT = args.project.resolve()
OUT = args.output.resolve()
if OUT.is_relative_to(ROOT/"evals/releases"):
    parser.error("输出请放在 tmp/，不能覆盖已冻结的公开结果")
sys.path.insert(0, str(ROOT))
import os
os.chdir(ROOT)

from sqlalchemy import select
from langchain_core.messages import HumanMessage, SystemMessage
from app.config import get_settings
from app.db import get_session_factory
from app.knowledge.models import KnowledgeChunk
from app.knowledge.retrieval import retrieve, reorder_for_prompt
from app.knowledge.embeddings import BgeM3Embedder, build_vector_text
from app.knowledge.reranker import BgeReranker
from app.knowledge.vector_store import MilvusKnowledgeStore
from app.knowledge.query_understanding import understand_query
from app.agent.graph import evidence_confidence
from app.agent.parsing import extract_json_object
from app.llm import build_chat_model
from app.prompts import CONVERGENCE_SYSTEM, REFUSAL_PREFIX
from evals.ch04.run_eval import JUDGE_SYSTEM
from evals.ch09.calibrate_gate import best_threshold

def digest(data):
    return hashlib.sha256(data).hexdigest()

def dump(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

def now():
    return datetime.now().astimezone().isoformat()

def log(message):
    print(f"[{now()}] {message}", flush=True)

def append(name, value):
    with (OUT / name).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")

def gate_metrics(samples, threshold):
    answerable = [x for x in samples if x["answerable"]]
    absent = [x for x in samples if not x["answerable"]]
    passed = sum(x["confidence"] >= threshold for x in answerable)
    blocked = sum(x["confidence"] < threshold for x in absent)
    return {"threshold":threshold, "answerable_count":len(answerable),
            "answerable_passed":passed, "answerable_pass_rate":passed/len(answerable),
            "absent_count":len(absent), "absent_blocked":blocked,
            "correct_refusal":blocked/len(absent),
            "false_allow_rate":1-blocked/len(absent), "false_block_rate":1-passed/len(answerable)}

def resolve_gt(cases, rows):
    by_question = defaultdict(list)
    for row in rows:
        by_question[row["questions"].splitlines()[0]].append(row)
    out = {}
    for case in cases:
        ids = set()
        for expected in case["gt"]:
            matched = [r["id"] for r in by_question[expected["questions"]]
                if r["category"] == expected["category"]
                or r["category"].startswith(expected["category"]+">")
                or expected["category"].startswith(r["category"]+">")]
            if not matched:
                raise RuntimeError(f"Unresolved active ground truth: {case['id']} {expected}")
            ids.update(matched)
        out[case["id"]] = sorted(ids)
    return out

async def main():
    OUT.mkdir(parents=True, exist_ok=False)
    started = now()
    settings = get_settings()
    fields = ["llm_model","llm_temperature","llm_thinking_enabled","embed_model_name",
              "rerank_model_name","local_model_dtype","recall_top_k","rerank_candidates",
              "rerank_top_k","retrieval_top_k","retrieval_strategy","query_rewrite_enabled",
              "retrieval_excluded_categories","evidence_conf_threshold","agent_max_steps",
              "tool_timeout_seconds","tool_max_retries","intent_model","intent_small_model",
              "intent_escalation_enabled","intent_confidence_threshold"]
    config = {k:getattr(settings,k) for k in fields}
    config.update({"project_root":"${PROJECT_ROOT}", "evaluation":"all 100 questions; no generation limit",
                   "api_concurrency":3, "api_timeout_seconds":90,
                   "retrieval_metric_depth":10,
                   "generation_query_rewrite":False,
                   "gate_query_rewrite":False,
                   "judge_parse_failure":"unknown; never counted as faithful",
                   "retrieval_formula":"mean per-question fraction of GT blocks hit; MRR first GT rank",
                   "calibration_scope":"in-sample diagnostic; not held-out production estimate"})
    file_hashes = {p.relative_to(ROOT).as_posix():digest(p.read_bytes())
        for base in ("app","evals/ch04","evals/ch09")
        for p in (ROOT/base).rglob("*.py")}
    head = subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,text=True,capture_output=True)
    config["git_head"] = head.stdout.strip() if head.returncode == 0 else None
    config["source_working_tree_dirty"] = bool(subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip())
    config["source_sha256"] = file_hashes
    config["packages"] = {name:version(name) for name in
        ["torch","transformers","sentence-transformers","pymilvus","langchain-openai","pydantic-settings"]}
    config["runner_sha256"] = digest(Path(__file__).read_bytes())
    dump("run_config.json",config)
    shutil.copy2(__file__, OUT/"run_audit.py")
    raw = args.cases.resolve().read_bytes()
    cases = [json.loads(x) for x in raw.decode("utf-8").splitlines() if x.strip()]
    revisions_file = args.cases.resolve().parent/"label_revisions.json"
    revisions = json.loads(revisions_file.read_text(encoding="utf-8")) if revisions_file.exists() else []
    changes = {r["id"]:r for r in revisions}
    assert len(cases)==100 and len({c["id"] for c in cases})==100
    assert all(c.get("answerability") in {"full","partial","absent"} for c in cases)
    (OUT/"eval_cases.jsonl").write_bytes(raw)
    dump("label_revisions.json",revisions)
    factory = get_session_factory()
    with factory() as session:
        rows = session.scalars(select(KnowledgeChunk).order_by(KnowledgeChunk.id)).all()
        snapshot = [{k:getattr(r,k) for k in
            ["id","category","questions","answer","section_path","content_type",
             "is_key_clause","vector_id","vectorize_status","governance_status"]} for r in rows]
    active = [r for r in snapshot if r["governance_status"]=="active"]
    dump("knowledge_snapshot.json",active)
    store = MilvusKnowledgeStore()
    vectors = store.client.query(store.collection,filter="",output_fields=["vector_id","chunk_id","text","category"],limit=16384)
    active_ids = {r["id"] for r in active}
    vectors = [v for v in vectors if v["chunk_id"] in active_ids]
    dump("milvus_text_snapshot.json",vectors)
    vector_map = {v["chunk_id"]:v for v in vectors}
    mismatches = [r["id"] for r in active if r["vectorize_status"]!="done"
        or r["id"] not in vector_map
        or vector_map[r["id"]]["text"] != build_vector_text(r["category"],r["questions"],r["answer"])]
    if mismatches:
        dump("preflight_failure.json",{"knowledge_vector_mismatches":mismatches})
        raise RuntimeError(f"KB/Milvus mismatch, evaluation stopped: {mismatches}")
    gt_ids = resolve_gt(cases,active)
    dump("ground_truth_ids.json",gt_ids)
    manifest = {"started_at":started,"input_cases_sha256":digest(raw),"knowledge_snapshot_scope":"active_only",
                "corrected_cases_sha256":digest((OUT/"eval_cases.jsonl").read_bytes()),
                "knowledge_sha256":digest((OUT/"knowledge_snapshot.json").read_bytes()),
                "milvus_sha256":digest((OUT/"milvus_text_snapshot.json").read_bytes()),
                "governance_counts":dict(Counter(r["governance_status"] for r in snapshot)),
                "active_count":len(active),"milvus_count":len(vectors),
                "bucket_counts":dict(Counter(c["bucket"] for c in cases)),
                "answerability_counts":dict(Counter(c["answerability"] for c in cases)),
                "gate_positive_label":"full or partial evidence; not a guarantee of complete answers",
                "metric_limitations":["Chapter-based GT may include table introductory blocks.",
                    "E13 asks purifier accessories but legacy GT includes rice-cooker liner; no inference across products is allowed.",
                    "E15 asks e-/paper-invoice differences but KB documents only electronic issuance; unsupported portions must be declined.",
                    "Overlapping/contradictory membership, warranty, delivery and return-window texts remain in the KB; this run does not rewrite business policies."]}
    dump("manifest.json",manifest)
    log(f"Preflight OK: {len(active)} active chunks, {len(vectors)} vectors; buckets={manifest['bucket_counts']}")
    llm = build_chat_model(settings)
    llm.request_timeout = 90
    llm.max_retries = 1
    sem = asyncio.Semaphore(3)
    query_map = {}
    async def rewrite(case):
        async with sem:
            value = await asyncio.to_thread(understand_query,llm,case["query"]) if settings.query_rewrite_enabled else {"all_texts":[case["query"]]}
            query_map[case["id"]] = value
            append("query_rewrites.jsonl",{"id":case["id"],"query":case["query"],"result":value})
    answerable = [c for c in cases if c["bucket"]!="D_absent"]
    log(f"Query understanding: {len(answerable)} cases")
    await asyncio.gather(*(rewrite(c) for c in answerable))
    log("Loading current local embedding/reranker models")
    embedder = BgeM3Embedder.get()
    reranker = BgeReranker.get()
    from functools import lru_cache
    embedder.embed_query = lru_cache(maxsize=512)(embedder.embed_query)
    strategies = ["dense","bm25","hybrid","hybrid_rerank"]
    per = {s:defaultdict(list) for s in strategies}
    original70 = {s:[] for s in strategies}
    log(f"Retrieval: {len(answerable)} x 4 strategies")
    for index,case in enumerate(answerable,1):
        for strategy in strategies:
            hits = retrieve(factory,case["query"],top_k=10,strategy=strategy,
                 query_texts=query_map[case["id"]]["all_texts"],store=store,embedder=embedder,reranker=reranker)
            ranked = [h["chunk_id"] for h in hits]
            gt = set(gt_ids[case["id"]])
            values = (len(set(ranked[:5])&gt)/len(gt),len(set(ranked[:10])&gt)/len(gt),
                       next((1/i for i,cid in enumerate(ranked,1) if cid in gt),0.0))
            per[strategy][case["bucket"]].append(values)
            if case["id"] not in changes:
                original70[strategy].append(values)
            append("retrieval_cases.jsonl",{"id":case["id"],"strategy":strategy,"query":case["query"],
                   "query_texts":query_map[case["id"]]["all_texts"],"gt_ids":sorted(gt),
                   "r5":values[0],"r10":values[1],"mrr":values[2],"hits":hits})
        if index%10==0 or index==len(answerable):
            log(f"Retrieval {index}/{len(answerable)}")
    def summarize(values):
        return {"n":len(values),"r5":statistics.mean(x[0] for x in values),
                "r10":statistics.mean(x[1] for x in values),"mrr":statistics.mean(x[2] for x in values)}
    retrieval = {}
    for strategy in strategies:
        retrieval[strategy] = {bucket:summarize(values) for bucket,values in per[strategy].items()}
        retrieval[strategy]["ALL"] = summarize([x for xs in per[strategy].values() for x in xs])
    dump("retrieval_summary.json",retrieval)
    dump("unchanged_cases_retrieval_summary.json",{s:summarize(v) for s,v in original70.items()})
    log("Raw-query retrieval for generation and current gate")
    samples = []
    generation_hits = {}
    for case in cases:
        hits = retrieve(factory,case["query"],top_k=settings.rerank_top_k,strategy="hybrid_rerank",
                        store=store,embedder=embedder,reranker=reranker)
        generation_hits[case["id"]] = reorder_for_prompt(hits)
        samples.append({"id":case["id"],"answerable":case["bucket"]!="D_absent",**evidence_confidence(hits)})
    recommended = best_threshold(samples)
    gate = {"sample_count":len(samples),"current":gate_metrics(samples,settings.evidence_conf_threshold),
            "recommended":recommended,"samples":samples,
            "note":"Current formula; raw-query retrieval; in-sample calibration diagnostic; runtime threshold unchanged."}
    dump("gate_calibration.json",gate)
    gen_results = []
    judge_system = JUDGE_SYSTEM + """\n补充返回两个字段，不改变上述 faithful 的判据：
refused: 回答是否明确说明资料不足/无法可靠确认用户所问的核心事实，而不是自行作肯定结论。
answerable_from_evidence: 给定证据是否足以支持所问核心事实；“有明确不支持/不承诺规则”可以提供受限的否定答复，不等于资料缺失。
返回 JSON: {"faithful":true/false,"refused":true/false,"answerable_from_evidence":true/false,"reason":"..."}。"""
    async def generate(case):
        async with sem:
            began = time.perf_counter()
            evidence = generation_hits[case["id"]]
            evidence_text = "\n".join(f"[{i}] {h['question']}：{h['answer']}" for i,h in enumerate(evidence,1)) or "（知识库未检索到相关内容）"
            messages = [SystemMessage(content=CONVERGENCE_SYSTEM.format(platform_name=settings.platform_name)),
                        HumanMessage(content=f"用户问题：{case['query']}\n\n知识库检索结果：\n{evidence_text}")]
            row = {"id":case["id"],"bucket":case["bucket"],"answerability":case["answerability"],
                   "query":case["query"],"evidence":evidence}
            try:
                reply = (await llm.ainvoke(messages)).text.strip()
                row["answer"] = reply
                prompt = f"用户问题：{case['query']}\n\n证据列表：\n{evidence_text}\n\n客服回答：\n{reply}"
                raw_judge = (await llm.ainvoke([SystemMessage(content=judge_system),HumanMessage(content=prompt)])).text
                verdict = extract_json_object(raw_judge)
                row["judge_raw"] = raw_judge
                row["verdict"] = verdict
                row["judge_valid"] = isinstance(verdict,dict) and all(isinstance(verdict.get(k),bool) for k in ["faithful","refused","answerable_from_evidence"])
                row["prefix_refused"] = reply.startswith(REFUSAL_PREFIX)
            except Exception as exc:
                row["error"] = type(exc).__name__
                row["judge_valid"] = False
            row["duration_seconds"] = round(time.perf_counter()-began,3)
            gen_results.append(row)
            append("generation_cases.jsonl",row)
            log(f"Generation {len(gen_results)}/100 {case['id']} valid={row['judge_valid']}")
    log("Generation + judge: all 100 cases")
    await asyncio.gather(*(generate(c) for c in cases))
    arows = [r for r in gen_results if r["bucket"]!="D_absent"]
    drows = [r for r in gen_results if r["bucket"]=="D_absent"]
    judged = [r for r in arows if r["judge_valid"]]
    fabricated = [r for r in judged if not r["verdict"]["faithful"]]
    generation = {"answerable_total":len(arows),"judged":len(judged),"judge_unknown":len(arows)-len(judged),
        "fabricated":len(fabricated),"fabricated_ids":[r["id"] for r in fabricated],
        "faithfulness":1-len(fabricated)/len(judged) if judged else None,
        "answerable_refused":sum(r["judge_valid"] and r["verdict"]["refused"] for r in arows),
        "d_total":len(drows),"d_judge_unknown":sum(not r["judge_valid"] for r in drows),
        "refused_correct_prefix":sum(r.get("prefix_refused",False) for r in drows),
        "refused_correct_semantic":sum(r["judge_valid"] and r["verdict"]["refused"] for r in drows),
        "d_label_conflict_ids":[r["id"] for r in drows if r["judge_valid"] and r["verdict"]["answerable_from_evidence"]],
        "api_error_ids":[r["id"] for r in gen_results if "error" in r]}
    generation["reference_bearing_total"] = len(arows)
    generation["fully_answerable_total"] = sum(r["answerability"]=="full" for r in arows)
    generation["partial_support_total"] = sum(r["answerability"]=="partial" for r in arows)
    generation["reference_bearing_prefix_refused"] = sum(r.get("prefix_refused",False) for r in arows)
    generation["reference_bearing_semantic_refusal_ids"] = [r["id"] for r in arows if r["judge_valid"] and r["verdict"]["refused"]]
    generation["prefix_refusal_rate"] = generation["refused_correct_prefix"]/len(drows)
    generation["semantic_refusal_rate"] = generation["refused_correct_semantic"]/len(drows)
    report = {"run_started_at":started,"run_completed_at":now(),"cases_total":len(cases),
              "run_config":config,"manifest":manifest,"retrieval":retrieval,"generation":generation,
              "gate":{"current":gate["current"],"recommended":recommended}}
    dump("report.json",report)
    lines = ["# 当前项目全量 RAG 审计评估", "","- 项目：${PROJECT_ROOT}",f"- 开始：{started}",
        f"- 完成：{report['run_completed_at']}",f"- 分桶：{manifest['bucket_counts']}",
        f"- 当前重排序候选数：{settings.rerank_candidates}；改写：{settings.query_rewrite_enabled}",
        "", "| 策略 | R@5 | R@10 | MRR | n |","|---|---:|---:|---:|---:|"]
    for strategy in strategies:
        v = retrieval[strategy]["ALL"]
        lines.append(f"| {strategy} | {v['r5']:.4%} | {v['r10']:.4%} | {v['mrr']:.4%} | {v['n']} |")
    lines += ["","## 生成与拒答","",json.dumps(generation,ensure_ascii=False,indent=2),
        "","## 当前闸与同集重标定","",json.dumps(report["gate"],ensure_ascii=False,indent=2),
        "","## 口径限制",""] + ["- "+s for s in manifest["metric_limitations"]]
    (OUT/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    log("COMPLETE "+json.dumps(generation,ensure_ascii=False))

if __name__=="__main__":
    asyncio.run(main())

