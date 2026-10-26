"""ch06 M4 curl 级实证：物流 → 「这个能退吗」(指代消解+子流程+订单选择器) →
点选回填(扩写+多路检索+agent 终判+退款表单建议) → 提交退款单。

用法：服务起着时  python evals/ch06/e2e_aftersale_demo.py
"""

import json
import sys
import time
import uuid

import httpx

BASE = "http://127.0.0.1:8000"
USER = "u-e2e-" + uuid.uuid4().hex[:8]


def post_chat(client, payload):
    frames = []
    t0 = time.time()
    with client.stream("POST", f"{BASE}/api/agent/chat", json=payload, timeout=300) as r:
        for line in r.iter_lines():
            if line.startswith("data: "):
                try:
                    frames.append(json.loads(line[6:]))
                except json.JSONDecodeError:
                    pass
    return frames, time.time() - t0


def show(frames, elapsed):
    for f in frames:
        t = f.get("type")
        if t == "node":
            print("  node:", f["name"])
        elif t == "order_picker":
            print("  >>> 订单选择器弹出:")
            for o in f.get("orders", []):
                print(f"      {o['order_no']} | {o['product_name']} | {o['status']} | {o['amount']}元")
        elif t == "suggestions":
            print("  >>> 建议:", [s["label"] for s in f.get("items", [])])
        elif t == "citations":
            print("  >>> 引用:", [c.get("section_path") for c in f.get("items", [])][:5])
        elif t == "delta":
            continue
        elif t in ("meta", "tool"):
            if t == "tool":
                print("  tool:", f.get("name"))
        else:
            print(f"  [{t}]", json.dumps(f, ensure_ascii=False)[:200])
    reply = "".join(f.get("content", "") for f in frames if f.get("type") == "delta")
    if reply:
        print("  终答:", reply[:300])
    print(f"  ({elapsed:.1f}s)")


def main() -> int:
    ok = True
    with httpx.Client() as client:
        conv_id = None

        print(f"== 第1轮 物流(user={USER}) ==")
        frames, el = post_chat(client, {"user_id": USER, "message": "我买的空气净化器什么时候能到"})
        show(frames, el)
        conv_id = next((f["conversation_id"] for f in frames if f.get("type") == "meta"), None)
        assert conv_id, "未拿到 conversation_id"

        print("\n== 第2轮 「这个能退吗」:期待 消解→退款子流程→订单选择器中断 ==")
        frames, el = post_chat(client, {"conversation_id": conv_id, "user_id": USER, "message": "这个能退吗"})
        show(frames, el)
        picker = next((f for f in frames if f.get("type") == "order_picker"), None)
        nodes = [f["name"] for f in frames if f.get("type") == "node"]
        assert picker, "第2轮应弹订单选择器"
        assert nodes[:2] == ["resolve_refs", "classify"], f"节点顺序异常: {nodes}"
        order_id = picker["orders"][0]["id"]
        order_no = picker["orders"][0]["order_no"]
        has_reply = any(f.get("type") == "delta" and f.get("content") for f in frames)
        assert not has_reply, "中断轮不应有终答"

        print(f"\n== 第3轮 点选订单 {order_no} 回填:期待 扩写+多路检索+agent 终判+退款表单建议 ==")
        frames, el = post_chat(client, {"conversation_id": conv_id, "user_id": USER, "resume": {"order_id": order_id}})
        show(frames, el)
        nodes = [f["name"] for f in frames if f.get("type") == "node"]
        assert "aftersale_orders" in nodes and "aftersale_policy" in nodes and "agent" in nodes, nodes
        reply = "".join(f.get("content", "") for f in frames if f.get("type") == "delta")
        assert reply, "应出终答"
        sugg = next((f for f in frames if f.get("type") == "suggestions"), None)
        assert sugg and sugg["items"][0]["action"] == "refund_form", sugg

        print("\n== 第4轮 提交退款表单(固定类目) ==")
        r = client.post(f"{BASE}/api/agent/refund", json={
            "conversation_id": conv_id, "user_id": USER,
            "order_id": order_id, "reason": "七天无理由",
        }, timeout=30)
        print(" ", r.status_code, r.json())
        assert r.status_code == 200 and r.json()["refund_no"].startswith("RF-")

    print("\n全部断言通过 ✓" if ok else "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
