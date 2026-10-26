"""查看 checkpointer 里到底存了啥（可读视图）。

用法：python -m app.agent.inspect_state <conversation_id>
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from langchain_core.messages import HumanMessage

from app.agent.graph import AgentDeps, build_graph, close_checkpointer, get_checkpointer


async def main(conv_id: str) -> None:
    deps = AgentDeps(llm=None, retriever=lambda q: [], session_factory=None)
    checkpointer = await get_checkpointer()
    graph = build_graph(deps).compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": conv_id}}
    history = [snap async for snap in graph.aget_state_history(config)]
    if not history:
        print(f"thread {conv_id} 在 checkpointer 里没有状态")
        return
    print(f"=== thread {conv_id}：{len(history)} 个 superstep 快照（新→旧）===\n")
    for snap in history:
        v = snap.values or {}
        trace = " → ".join(v.get("node_trace") or [])
        print(f"[step {snap.metadata.get('step', '?')}] next={snap.next} | intent={v.get('intent')} | trace={trace}")
        if v.get("gate_passed") is not None:
            print(f"    gate_passed={v.get('gate_passed')}")
        if v.get("retrieval_evidence"):
            top = v["retrieval_evidence"][0]
            print(f"    evidence top1: {top.get('question', '')[:24]} score={top.get('score')}")
        if v.get("final_reply"):
            print(f"    final: {v['final_reply'][:60]}")
        msgs = v.get("messages") or []
        for m in msgs[-4:]:
            print(f"    [{m.type}] {str(m.content)[:70]}")
        print()
    cur = (await graph.aget_state(config)).values or {}
    print("=== 当前态 ===")
    print("messages:", len(cur.get("messages") or []), "| intent:", cur.get("intent"))
    print("suggestions:", cur.get("suggestions"))
    print("node_trace:", cur.get("node_trace"))
    await close_checkpointer()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "1"))
