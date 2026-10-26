"""ch09 飞轮验收脚本（需本地服务已启动）。

默认跑到“差评 → 标准化归并 → 审核详情可追溯”；传 --approve-answer 才会
临时发布知识并复问，验证后默认删除测试知识块。
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.agent.graph import GATE_FALLBACK_REPLY
from app.config import get_settings


def request_json(url: str, method: str = "GET", body: dict | None = None) -> dict:
    data = json.dumps(body, ensure_ascii=False).encode() if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"{method} {url}: HTTP {exc.code} {exc.read().decode()}") from exc


def chat(base_url: str, user_id: str, question: str) -> dict:
    req = urllib.request.Request(
        f"{base_url}/api/agent/chat",
        data=json.dumps({"user_id": user_id, "message": question}, ensure_ascii=False).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    frames = []
    with urllib.request.urlopen(req, timeout=180) as response:
        for raw in response:
            line = raw.decode().strip()
            if line.startswith("data: "):
                frames.append(json.loads(line[6:]))
    meta = next(item for item in frames if item["type"] == "meta")
    done = next(item for item in frames if item["type"] == "done")
    reply = "".join(item.get("content", "") for item in frames if item["type"] == "delta")
    return {"conversation_id": meta["conversation_id"], "message_id": done.get("assistant_message_id"), "reply": reply}


def find_review(base_url: str, question: str, source: str, timeout: int = 60) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        for item in request_json(f"{base_url}/api/review/items?status={quote('待审')}")["items"]:
            detail = request_json(f"{base_url}/api/review/items/{item['id']}")
            if any(
                raw["raw_question"] == question and raw["source"] == source
                for raw in detail["raw_questions"]
            ):
                return detail
        time.sleep(2)
    raise RuntimeError("60 秒内未在待审队列找到该问题")


def delete_test_chunk(chunk_id: int) -> None:
    from app.db import get_session_factory
    from app.knowledge.models import KnowledgeChunk
    from app.knowledge.vector_store import MilvusKnowledgeStore

    MilvusKnowledgeStore().delete_by_chunk_ids([chunk_id])
    factory = get_session_factory()
    with factory() as session:
        row = session.get(KnowledgeChunk, chunk_id)
        if row is not None:
            session.delete(row)
            session.commit()


def verify_langfuse(user_id: str, timeout: int = 60) -> tuple[str, int, dict[str, int]]:
    from langfuse import Langfuse

    settings = get_settings()
    no_proxy = os.environ.get("NO_PROXY", "")
    os.environ["NO_PROXY"] = ",".join(filter(None, [no_proxy, "127.0.0.1", "localhost"]))
    client = Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        base_url=settings.langfuse_host,
    )
    fields = "core,basic,io,metadata,usage,trace_context,model"
    deadline = time.time() + timeout
    root = None
    while time.time() < deadline:
        roots = client.api.observations.get_many(
            name="agent-chat",
            user_id=user_id,
            is_root_observation=True,
            fields=fields,
            limit=10,
        ).data
        if roots:
            root = roots[0]
            break
        time.sleep(2)
    assert root is not None, "Langfuse 60 秒内未查到本次 agent-chat 根观测"
    observations = client.api.observations.get_many(
        trace_id=root.trace_id,
        fields=fields,
        limit=100,
    ).data
    assert len(observations) > 1, "Langfuse trace 缺少节点/模型子观测"
    names = {item.name for item in observations}
    assert {"classify", "retrieve"} <= names, "Langfuse trace 缺少 classify/retrieve 节点"
    assert any(item.name == "retrieve" and item.output for item in observations), (
        "Langfuse retrieve 节点缺少检索输出"
    )
    assert any(item.type == "GENERATION" and item.input and item.output for item in observations), (
        "Langfuse trace 缺少带输入输出的模型调用"
    )
    tokens = sum((item.usage_details or {}).get("total", 0) for item in observations)
    assert tokens > 0, "Langfuse trace 缺少 token 用量"
    assert root.metadata and root.metadata.get("intent"), "Langfuse 根观测缺 intent metadata"
    assert root.metadata.get("entry") == "agent", "Langfuse 根观测缺 agent 入口 metadata"

    recent = client.api.observations.get_many(fields=fields, limit=1000).data
    roots_by_trace = {
        item.trace_id: item for item in recent
        if item.is_root_observation and item.name == "agent-chat"
        and item.metadata and item.metadata.get("intent")
    }
    by_intent: dict[str, int] = {}
    for item in recent:
        trace_root = roots_by_trace.get(item.trace_id)
        if trace_root is not None:
            intent = str(trace_root.metadata["intent"])
            by_intent[intent] = by_intent.get(intent, 0) + (
                (item.usage_details or {}).get("total", 0)
            )
    assert by_intent, "Langfuse 中没有可按 intent 汇总的 token 数据"
    assert len(by_intent) >= 2, "Langfuse 至少需要两个 intent 才能比较哪类最耗 token"
    return root.trace_id, tokens, by_intent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--question", default="玄武星会员的跨星球运费是多少？")
    parser.add_argument("--approve-answer")
    parser.add_argument("--keep-published", action="store_true")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    user_id = f"ch09-e2e-{uuid.uuid4().hex[:8]}"

    assert request_json(f"{base}/health")["status"] == "ok"
    first = chat(base, user_id, args.question)
    assert first["message_id"], "done 帧缺 assistant_message_id"
    assert first["reply"] == GATE_FALLBACK_REPLY, "库外问题未被置信度闸拦截"
    gate_review = find_review(base, args.question, "retrieval_low_conf")
    gate_raw = next(
        item for item in gate_review["raw_questions"]
        if item["raw_question"] == args.question and item["source"] == "retrieval_low_conf"
    )
    assert gate_raw["retrieval_snapshot"], "置信闸自动落池缺当轮召回快照"
    print(f"[OK] 库外问题被置信闸拦截并自动落池 #{gate_review['id']}")

    feedback = request_json(f"{base}/api/agent/feedback", "POST", {
        "conversation_id": first["conversation_id"],
        "message_id": first["message_id"],
        "helpful": False,
    })
    assert feedback["status"] == "queued"
    review = find_review(base, args.question, "user_feedback")
    raw = next(
        item for item in review["raw_questions"]
        if item["raw_question"] == args.question and item["source"] == "user_feedback"
    )
    assert raw["retrieval_snapshot"], "审核详情缺当轮召回快照"
    print(f"[OK] 差评落池并归并到待审 #{review['id']}：{review['std_question']}")

    if args.approve_answer:
        published = request_json(f"{base}/api/review/items/{review['id']}/approve", "POST", {
            "approved_answer": args.approve_answer,
        })
        second = chat(base, user_id, args.question)
        assert args.approve_answer[:8] in second["reply"], "复问未命中新发布知识"
        print(f"[OK] 审核通过发布知识块 #{published['chunk_id']}，复问已命中")
        if not args.keep_published:
            delete_test_chunk(published["chunk_id"])
            print(f"- 已清理测试知识块 #{published['chunk_id']}")
    else:
        print("- 未传 --approve-answer，跳过知识发布与复问")

    runs = request_json(f"{base}/api/eval/runs")["items"]
    assert len(runs) >= 2, f"评估轮次不足：{len(runs)} < 2"
    assert runs[0]["delta"], "最新评估轮次没有趋势差值"
    print(f"[OK] eval_runs={len(runs)}，最新轮次已有趋势差值")

    trace_id, tokens, by_intent = verify_langfuse(user_id)
    hottest = max(by_intent, key=by_intent.get)
    print(f"[OK] Langfuse trace={trace_id}，本次 {tokens} tokens")
    print(f"[OK] 按 intent 汇总 token：{by_intent}；当前最高={hottest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
