"""ch07 端到端验收脚本：自起独立服务（独立端口+独立日志目录）驱动多轮对话。

模式：
    python evals/ch07/e2e_context_demo.py --mode demo    # 演示配置(18000窗口):级联+梗概答题
    python evals/ch07/e2e_context_demo.py --mode default # 默认配置:20轮零压缩(装得下就不压)

demo 断言（验收 1/2/4）：
- 22 轮跑完服务不崩、无 error 帧；
- 日志出现完整级联：层1 降级 X→Y → summary trigger → summary done 第N段；
- 压缩后的「最开始那个订单后来怎么说」答对（回复含第 1 轮点名的 SO-1234-88）；
- grep model_ctx / history_ctx 每轮可见，摘要注入后 model_ctx 摘要含 SO-1234-88。
default 断言（验收 3）：20 轮轻对话零降级零摘要。
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent.parent
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")

DEMO_ENV = {
    "MODEL_CONTEXT_WINDOW": "18000",
    "MAX_OUTPUT_TOKENS": "2000",
    "MAX_USER_INPUT_TOKENS": "2000",
    "MAX_AGENT_STEPS": "3",
    "TOOL_RESULT_MAX_TOKENS": "1200",
    "RERANK_TOP_K": "5",
}

LONG_TAIL = (
    "当时下单用的是会员积分抵扣了一部分，收货人是家里老人，地址写的老小区东门代收点，"
    "物流显示到了之后一直没人通知我去拿，我在 App 上看不到更新的节点，客服电话也打不通，"
    "我想把整个过程都说明白，希望你们认真记录一下每一个细节，不要漏掉任何一条信息，"
    "这已经不是第一次出现这种问题了，上次也是拖了好几天才解决，体验非常不好，"
    "麻烦这次一次性给我一个清楚的说法和准确的进度，不要再用模板话术敷衍了事。"
) * 3  # ~600+ token 的长陈述


def long_msg(head: str) -> str:
    return head + LONG_TAIL


DEMO_TURNS = [
    ("订单", long_msg("我上周下的订单 SO-1234-88，买的空气净化器 KJ-450F，")),  # 第1轮:关键订单号
    ("物流", long_msg("SO-1234-88 这单物流到哪了，")),
    ("商品", long_msg("再问下你们家无线降噪耳机 E-3 的降噪参数，")),
    ("投诉", long_msg("你们这个物流太慢了我忍不了要投诉，")),
    ("订单", long_msg("另外我还想确认下我名下有几笔订单，")),
    ("物流", long_msg("还是查一下最早那台的配送进度，")),
    ("商品", long_msg("对了净水器 JR-600 的滤芯多久换一次，")),
    ("其他", long_msg("顺便说一句你们的 App 首页设计真的很难用，")),
    ("订单", long_msg("再帮我看看最近一笔的状态，")),
    ("商品", long_msg("电饭煲 CFXB-40A 有预约功能吗，")),
]
SHORT_TURNS = [
    "好的知道了",
    "你们会员有什么权益",
    "在吗",
    "今天天气不错",
    "耳机续航多久",
    "退款政策是什么",
    "嗯嗯",
    "帮我推荐个电饭煲",
    "下单后多久发货",
    "谢谢啦",
    "你们几点下班",
]

# 默认配置的「装得下就不压」验收用轻对话：闲聊+短商品问答为主，
# 刻意避开订单/物流/推荐类重轮（工具 JSON 与长答案会把层1 顶破预算——那是演示配置的剧本）
LIGHT_TURNS = [
    "在吗", "你好呀", "今天天气不错", "聊聊天呗",
    "耳机防水吗", "好的知道了", "嗯嗯", "谢谢啦",
    "你们几点下班", "在忙吗", "晚安", "机器人还是真人",
    "会员分几级", "明白了", "辛苦了", "再见",
    "对了会员有什么权益", "收到", "那没事了", "祝生意兴隆",
]


def start_server(port: int, log_dir: str, env_extra: dict) -> subprocess.Popen:
    log_path = ROOT / (log_dir + ".uvicorn.log")
    env = {**os.environ, **env_extra, "LOG_DIR": log_dir}
    proc = subprocess.Popen(
        [PY, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(ROOT), env=env,
        stdout=open(log_path, "w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
    )
    for _ in range(180):  # BGE 双模型预热 30-60s
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=2, trust_env=False).status_code == 200:
                return proc
        except Exception:
            pass
        time.sleep(1)
    proc.terminate()
    raise RuntimeError(f"服务 {port} 120s 未就绪,看 {log_path}")


def chat(port: int, conv_id, message: str):
    frames = []
    with httpx.Client(timeout=300, trust_env=False) as c:
        with c.stream(
            "POST", f"http://127.0.0.1:{port}/api/agent/chat",
            json={"user_id": "u-ch07-e2e", "conversation_id": conv_id, "message": message},
        ) as r:
            for line in r.iter_lines():
                if line.startswith("data: "):
                    try:
                        frames.append(json.loads(line[6:]))
                    except json.JSONDecodeError:
                        pass
    conv = next((f["conversation_id"] for f in frames if f.get("type") == "meta"), conv_id)
    reply = "".join(f.get("content", "") for f in frames if f.get("type") == "delta")
    error = next((f.get("message") for f in frames if f.get("type") == "error"), None)
    return conv, reply, error


def run(mode: str) -> int:
    import shutil

    port, log_dir = (8010, "log/ch07-demo") if mode == "demo" else (8011, "log/ch07-default")
    env_extra = DEMO_ENV if mode == "demo" else {}
    applog = ROOT / log_dir / "app.log"
    shutil.rmtree(ROOT / log_dir, ignore_errors=True)  # 断言只看本次运行的日志
    proc = start_server(port, log_dir, env_extra)
    print(f"[{mode}] 服务就绪 port={port} 日志={applog}")
    try:
        conv_id = None
        errors = []

        def send(msg, tag=""):
            nonlocal conv_id
            t0 = time.time()
            conv_id, reply, error = chat(port, conv_id, msg)
            if error:
                errors.append(f"{tag or msg[:12]}: {error}")
            print(f"  [{time.time()-t0:4.1f}s] {msg[:18]:<20} -> {reply[:40]!r}")
            return reply

        if mode == "demo":
            for i, (tag, msg) in enumerate(DEMO_TURNS, 1):
                send(msg, tag)
            time.sleep(12)  # 等后台摘要任务收尾（触发在上一轮回复之后异步跑）
            recall_reply = send("最开始那个订单后来怎么说", "recall")
            for msg in SHORT_TURNS:
                send(msg)
            log_text = applog.read_text(encoding="utf-8")
            idx_degrade = log_text.find("层1 降级 msg_id")
            idx_trigger = log_text.find("summary trigger")
            idx_done = log_text.find("summary done 第1段")
            checks = {
                "级联顺序 降级→触发→完成": 0 <= idx_degrade < idx_trigger < idx_done,
                "层1 降级痕迹": idx_degrade >= 0,
                "summary trigger": idx_trigger >= 0,
                "summary done": idx_done >= 0,
                "梗概答题含订单号": "SO-1234-88" in recall_reply,
                "梗概答题含诉求": ("物流" in recall_reply) or ("投诉" in recall_reply),
                "model_ctx 摘要含订单号": any(
                    "model_ctx 摘要=" in line and "SO-1234-88" in line
                    for line in log_text.splitlines()
                ),
                "history_ctx 每轮可见": log_text.count("history_ctx conv=") >= 22,
                "无 error 帧": not errors,
            }
        else:
            for msg in LIGHT_TURNS:
                send(msg)
            log_text = applog.read_text(encoding="utf-8")
            checks = {
                "零降级": "层1 降级 msg_id" not in log_text,
                "零摘要触发": "summary trigger" not in log_text and "summary done" not in log_text,
                "history_ctx 每轮可见": log_text.count("history_ctx conv=") >= 20,
                "无 error 帧": not errors,
            }

        print(f"\n[{mode}] 断言：")
        ok = True
        for name, passed in checks.items():
            print(f"  {'✓' if passed else '✗'} {name}")
            ok = ok and passed
        if errors:
            print("  错误帧：", errors[:3])
        print(f"[{mode}] -> {'通过' if ok else '未通过'}")
        return 0 if ok else 1
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["demo", "default"], required=True)
    sys.exit(run(parser.parse_args().mode))
