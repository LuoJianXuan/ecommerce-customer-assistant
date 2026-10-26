"""Retry only malformed judge responses; keep the initial evidence/answer unchanged."""
import argparse, ast, asyncio, json, shutil, sys
from pathlib import Path
from datetime import datetime
parser=argparse.ArgumentParser(description="仅重试无法解析的模型裁判，不重新生成答案")
parser.add_argument("--run",type=Path,required=True)
args=parser.parse_args()
ROOT=Path(__file__).resolve().parents[1]
OUT=args.run.resolve()
if OUT.is_relative_to(ROOT/"evals/releases"):
    parser.error("不能修改已冻结结果；请选择 tmp/ 下的新运行目录")
sys.path.insert(0,str(ROOT))
from app.config import get_settings
from app.llm import build_chat_model
from app.agent.parsing import extract_json_object
from langchain_core.messages import SystemMessage,HumanMessage
from evals.ch04.run_eval import JUDGE_SYSTEM

async def main():
    records=[json.loads(x) for x in (OUT/"generation_cases.jsonl").read_text(encoding="utf-8").splitlines()]
    pending=[r for r in records if not r.get("judge_valid") and "answer" in r]
    if not pending:
        print("没有需要重试的已生成答案；未修改运行记录，也未调用模型。")
        return
    tree=ast.parse((OUT/"run_audit.py").read_text(encoding="utf-8"))
    extra=next(n.value.right.value for n in ast.walk(tree) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=="judge_system" for t in n.targets))
    llm=build_chat_model(get_settings());llm.request_timeout=90;llm.max_retries=1
    attempts=[]
    for row in records:
        if row not in pending:continue
        evidence="\n".join(f"[{i}] {h['question']}：{h['answer']}" for i,h in enumerate(row["evidence"],1))
        prompt=f"用户问题：{row['query']}\n\n证据列表：\n{evidence}\n\n客服回答：\n{row['answer']}"
        raw=(await llm.ainvoke([SystemMessage(content=JUDGE_SYSTEM+extra+
            "\n请输出可解析的合法JSON；reason中不要使用未转义的双引号。"),HumanMessage(content=prompt)])).text
        parsed=extract_json_object(raw)
        valid=isinstance(parsed,dict) and all(isinstance(parsed.get(k),bool)
            for k in ["faithful","refused","answerable_from_evidence"])
        attempts.append({"id":row["id"],"initial_raw":row["judge_raw"],"retry_raw":raw,"retry_valid":valid})
        if valid:
            row["judge_raw"],row["verdict"],row["judge_valid"]=raw,parsed,True
        print(json.dumps(attempts[-1],ensure_ascii=False),flush=True)
    if not (OUT/"generation_cases_before_judge_retry.jsonl").exists():
        shutil.copy2(OUT/"generation_cases.jsonl",OUT/"generation_cases_before_judge_retry.jsonl")
    (OUT/"generation_cases.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in records),encoding="utf-8")
    history=json.loads((OUT/"judge_retry.json").read_text(encoding="utf-8")) if (OUT/"judge_retry.json").exists() else []
    (OUT/"judge_retry.json").write_text(json.dumps(history+attempts,ensure_ascii=False,indent=2),encoding="utf-8")
    report=json.loads((OUT/"report.json").read_text(encoding="utf-8"));g=report["generation"];oldg=dict(g)
    a=[r for r in records if r["bucket"]!="D_absent"];d=[r for r in records if r["bucket"]=="D_absent"]
    judged=[r for r in a if r["judge_valid"]];fabricated=[r for r in judged if not r["verdict"]["faithful"]]
    g.update(judged=len(judged),judge_unknown=len(a)-len(judged),fabricated=len(fabricated),
        fabricated_ids=[r["id"] for r in fabricated],faithfulness=(1-len(fabricated)/len(judged) if judged else None),
        answerable_refused=sum(r["judge_valid"] and r["verdict"]["refused"] for r in a),
        reference_bearing_semantic_refusal_ids=[r["id"] for r in a if r["judge_valid"] and r["verdict"]["refused"]])
    report["judge_retry"]={"completed_at":datetime.now().astimezone().isoformat(),"cases":[x["id"] for x in attempts],
        "note":"Only malformed JSON retried; questions, evidence and answers unchanged. Initial output preserved."}
    (OUT/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    md=(OUT/"report.md").read_text(encoding="utf-8")
    md=md.replace(json.dumps(oldg,ensure_ascii=False,indent=2),json.dumps(g,ensure_ascii=False,indent=2))
    (OUT/"report.md").write_text(md,encoding="utf-8")
asyncio.run(main())

