"""ch08 工具系统端到端验收：自起三进程（主服务 8012 + MCP 8101/8102）跑验收六条。

用法：python evals/ch08/e2e_tool_system.py
① 插件新工具：脚本写入 app/tools/plugins/check_membership_points.py → 只重启主服务 → 对话用上 → 删文件
② 双 MCP Server：问物流轨迹 → Agent 经 MCP 查到并回答（审计 source=mcp:logistics）
③ MCP 侧加工具：重启 8101（MCP_DEMO_EXTRA_TOOL=1）→ 主服务零动 → 问配送费用上新工具
④ 建工单：缺描述追问 → 补齐 → 预览帧 → resume 确认 → tickets 落表+回复带工单号
⑤ 取消：同④到预览 → resume 取消 → 零工单+审计「权限拒绝」
⑥ 超时：读工具超时重试（审计 retry>0）+ 写工具超时不重试（retry=0）——引擎级直调（人为注入）
"""

import asyncio
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
PORT = 8012
LOG_DIR = ROOT / "log" / "ch08-e2e"

PLUGIN_FILE = ROOT / "app" / "tools" / "plugins" / "check_delivery_coverage.py"
PLUGIN_CODE = '''"""验收①演示插件：配送范围查询（e2e 脚本动态写入,跑完即删）。"""
import json
from langchain_core.tools import tool

@tool
def check_delivery_coverage(region: str) -> str:
    """查询配送范围是否覆盖某地区及大致时效。用户问某地是否配送/送货到不到/覆盖范围时调用。"""
    covered = "新疆" not in region or True  # 演示 mock:全境覆盖
    return json.dumps({"region": region, "covered": covered, "eta": "3-5 个工作日"}, ensure_ascii=False)

TOOLS = [check_delivery_coverage]
'''


def start_proc(args, env_extra, log_name):
    log = open(LOG_DIR / log_name, "w", encoding="utf-8")
    return subprocess.Popen(
        [PY, *args], cwd=str(ROOT),
        env={**os.environ, **env_extra}, stdout=log, stderr=subprocess.STDOUT,
    )


def wait_health(port, proc, timeout=150, path="/health", accept_any=False):
    """accept_any=True 用于 MCP Server（FastMCP 无 /health,任意 HTTP 响应即视为就绪）。"""
    for _ in range(timeout):
        if proc.poll() is not None:
            raise RuntimeError(f"端口 {port} 进程退出 code={proc.returncode},看 {LOG_DIR}")
        try:
            r = httpx.get(f"http://127.0.0.1:{port}{path}", timeout=2, trust_env=False)
            if (accept_any and r.status_code < 500) or r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError(f"端口 {port} {timeout}s 未就绪")


def stop(proc):
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


def chat(conv_id, message, user, expect_confirm=False):
    frames = []
    with httpx.Client(timeout=300, trust_env=False) as c:
        with c.stream("POST", f"http://127.0.0.1:{PORT}/api/agent/chat",
                      json={"user_id": user, "conversation_id": conv_id, "message": message}) as r:
            for line in r.iter_lines():
                if line.startswith("data: "):
                    try:
                        frames.append(json.loads(line[6:]))
                    except json.JSONDecodeError:
                        pass
    conv = next((f["conversation_id"] for f in frames if f.get("type") == "meta"), conv_id)
    reply = "".join(f.get("content", "") for f in frames if f.get("type") == "delta")
    confirm = next((f for f in frames if f.get("type") == "ticket_confirm"), None)
    return conv, reply, confirm


def resume(conv_id, payload, user):
    frames = []
    with httpx.Client(timeout=300, trust_env=False) as c:
        with c.stream("POST", f"http://127.0.0.1:{PORT}/api/agent/chat",
                      json={"user_id": user, "conversation_id": conv_id, "resume": payload}) as r:
            for line in r.iter_lines():
                if line.startswith("data: "):
                    try:
                        frames.append(json.loads(line[6:]))
                    except json.JSONDecodeError:
                        pass
    return "".join(f.get("content", "") for f in frames if f.get("type") == "delta")


def audit_rows(conv_id, tool=None):
    from sqlalchemy import text

    from app.db import get_session_factory

    q = "SELECT tool_name, status, source, retry_count FROM tool_audit_logs WHERE conversation_id=:c"
    p = {"c": conv_id}
    if tool:
        q += " AND tool_name=:t"
        p["t"] = tool
    with get_session_factory()() as s:
        return s.execute(text(q + " ORDER BY id"), p).all()


def new_conv_row(user):
    from sqlalchemy import text

    from app.db import get_session_factory

    with get_session_factory()() as s:
        cid = s.execute(text("INSERT INTO conversations (user_id) VALUES (:u)"), {"u": user}).lastrowid
        s.commit()
    return cid


async def timeout_demos(user):
    """验收⑥：人为超时——读工具重试、写工具零重试（引擎级直调注入真审计表）。"""
    import time as _time

    from langchain_core.tools import tool

    from app.tools.base import ToolRegistry
    from app.tools.engine import execute

    @tool
    def slow_probe(order_id: str) -> str:
        """慢探测（验收⑥人为注入）。"""
        _time.sleep(1.5)
        return "never"

    @tool
    def slow_write(description: str) -> str:
        """慢写（验收⑥人为注入）。"""
        _time.sleep(1.5)
        return "T999"

    from app.db import get_session_factory

    factory = get_session_factory()
    cid = new_conv_row(user)
    reg = ToolRegistry()
    reg.register(slow_probe, timeout_seconds=0.5)
    reg.register(slow_write, permission="write")

    r1 = await execute({"name": "slow_probe", "args": {"order_id": "1"}, "id": "c6a"}, reg,
                       conversation_id=cid, session_factory=factory, max_retries=2)
    r2 = await execute({"name": "slow_write", "args": {"description": "x"}, "id": "c6b"}, reg,
                       conversation_id=cid, session_factory=factory, max_retries=2,
                       default_timeout=0.5, confirmed=True)
    return cid, r1, r2, audit_rows(cid)


def main() -> int:
    import shutil

    shutil.rmtree(LOG_DIR, ignore_errors=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    logi = after = main_proc = None
    try:
        print("== 启动三进程 ==")
        logi = start_proc(["mcp_servers/logistics_server.py"], {}, "mcp_logistics.log")
        after = start_proc(["mcp_servers/aftersale_server.py"], {}, "mcp_aftersale.log")
        wait_health(8101, logi, 60, path="/mcp", accept_any=True)
        wait_health(8102, after, 60, path="/mcp", accept_any=True)
        main_proc = start_proc(["-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(PORT)],
                               {"LOG_DIR": str(LOG_DIR)}, "main.log")
        wait_health(PORT, main_proc, 200)

        # ---- ② 双 MCP Server：物流轨迹走 MCP ----
        user2 = "u-ch08-e2e-" + uuid.uuid4().hex[:6]
        cid, reply, _ = chat(None, "我买的耳机到哪了", user2)
        rows = [r for r in audit_rows(cid) if r[0] == "query_logistics"]
        results["②物流走MCP"] = bool(rows) and rows[0][2] == "mcp:logistics" and rows[0][1] == "成功" and any(
            k in reply for k in ("运单", "承运", "快递", "物流"))
        print(f"  ② {reply[:60]!r} | {rows}")

        # ---- ③ MCP 侧加工具：只重启 8101(带 extra),主服务零动 ----
        stop(logi)
        logi = start_proc(["mcp_servers/logistics_server.py"], {"MCP_DEMO_EXTRA_TOOL": "1"}, "mcp_logistics2.log")
        wait_health(8101, logi, 60, path="/mcp", accept_any=True)
        cid3, reply3, _ = chat(None, "查一下订单 SO-1234-99 的配送费要多少", user2)
        rows3 = [r for r in audit_rows(cid3) if r[0] == "check_delivery_fee"]
        results["③Server侧加工具"] = bool(rows3) and rows3[0][2] == "mcp:logistics" and any(
            k in reply3 for k in ("元", "包邮", "配送费", "运费"))
        print(f"  ③ {reply3[:60]!r} | {rows3}")

        # ---- ④ 建工单：追问→补齐→预览→确认 ----
        user4 = "u-ch08-e2e-" + uuid.uuid4().hex[:6]
        cid4, ask, _ = chat(None, "帮我建个工单", user4)
        cid4, reply4, confirm = chat(cid4, "净水器出水有异味,建售后工单", user4)
        assert confirm is not None, f"未收到工单预览帧: {reply4[:80]}"
        final4 = resume(cid4, {"confirmed": True}, user4)
        from sqlalchemy import text

        from app.db import get_session_factory

        with get_session_factory()() as s:
            n4 = s.execute(text("SELECT COUNT(*) FROM tickets WHERE conversation_id=:c"), {"c": cid4}).scalar()
        a4 = audit_rows(cid4, "create_ticket")
        results["④确认建单"] = n4 == 1 and a4 and a4[-1][1] == "成功" and "T2" in final4
        print(f"  ④ 追问={ask[:30]!r} 终答={final4[:60]!r} 工单数={n4} 审计={a4}")

        # ---- ⑤ 取消：零工单+权限拒绝 ----
        user5 = "u-ch08-e2e-" + uuid.uuid4().hex[:6]
        cid5, _, _ = chat(None, "帮我建个工单", user5)
        cid5, reply5, confirm5 = chat(cid5, "买的净水器和页面描述不符,建售后单", user5)
        assert confirm5 is not None, f"未收到预览帧: {reply5[:80]}"
        final5 = resume(cid5, {"confirmed": False}, user5)
        with get_session_factory()() as s:
            n5 = s.execute(text("SELECT COUNT(*) FROM tickets WHERE conversation_id=:c"), {"c": cid5}).scalar()
        a5 = audit_rows(cid5, "create_ticket")
        results["⑤取消=权限拒绝"] = n5 == 0 and a5 and a5[-1][1] == "权限拒绝"
        print(f"  ⑤ 工单数={n5} 审计={a5} 终答={final5[:50]!r}")

        # ---- ① 插件新工具：写入插件文件→只重启主服务→用上→删 ----
        PLUGIN_FILE.write_text(PLUGIN_CODE, encoding="utf-8")
        stop(main_proc)
        main_proc = start_proc(["-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(PORT)],
                               {"LOG_DIR": str(LOG_DIR)}, "main2.log")
        wait_health(PORT, main_proc, 200)
        user1 = "u-ch08-e2e-" + uuid.uuid4().hex[:6]
        cid1, reply1, _ = chat(None, "你们配送范围覆盖新疆吗?大概几天能到", user1)
        a1 = audit_rows(cid1, "check_delivery_coverage")
        results["①插件新工具"] = bool(a1) and a1[0][2] == "builtin" and any(
            k in reply1 for k in ("新疆", "配送", "工作日"))
        print(f"  ① {reply1[:60]!r} | {a1}")
        PLUGIN_FILE.unlink(missing_ok=True)

        # ---- ⑥ 人为超时（引擎级注入） ----
        user6 = "u-ch08-e2e-" + uuid.uuid4().hex[:6]
        cid6, r1, r2, a6 = asyncio.run(timeout_demos(user6))
        read_row = next(r for r in a6 if r[0] == "slow_probe")
        write_row = next(r for r in a6 if r[0] == "slow_write")
        results["⑥超时重试策略"] = (
            r1.status == "超时" and read_row[3] >= 1 and read_row[1] == "超时"
            and r2.status == "超时" and write_row[3] == 0 and write_row[1] == "超时"
        )
        print(f"  ⑥ 读={read_row} 写={write_row}")
    finally:
        PLUGIN_FILE.unlink(missing_ok=True)
        stop(main_proc)
        stop(logi)
        stop(after)

    print("\n== 验收结果 ==")
    ok = True
    for name, passed in results.items():
        print(f"  {'✓' if passed else '✗'} {name}")
        ok = ok and passed
    print("-> " + ("六条全过" if ok else "未通过"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
