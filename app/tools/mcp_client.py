"""ch08 MCP Client：进程级 MultiServerMCPClient，每轮现问现拿（拍板②）。

- Server 配置来自 settings（mcp_logistics_url / mcp_aftersale_url）;
- 每轮调用 fetch_mcp_tools() 分 Server 拉清单（分组拉是为了给 ToolRecord 标 source=mcp:<name>）;
- 任一 Server 拉取失败降级为空+warn，不阻塞聊天（Server 不在线=工具缺席，Agent 诚实说查不到）;
- 合并进注册中心用 on_duplicate="skip"：与内置重名的 MCP 工具跳过，别留重名工具。
"""

import logging
import os

logger = logging.getLogger("mcp_client")

# 本机 MCP 不许走系统代理（httpx 默认 trust_env,代理劫持 127.0.0.1 会 502——实测翻车）
for _host in ("127.0.0.1", "localhost"):
    if _host not in (os.environ.get("NO_PROXY") or ""):
        os.environ["NO_PROXY"] = (os.environ.get("NO_PROXY") + "," if os.environ.get("NO_PROXY") else "") + _host

_client = None  # 进程级单例（连接配置不变；adapters 每次调用自建会话，单例无状态风险）


def _get_client():
    global _client
    if _client is None:
        from langchain_mcp_adapters.client import MultiServerMCPClient

        from app.config import get_settings

        s = get_settings()
        _client = MultiServerMCPClient(
            {
                "logistics": {"transport": "http", "url": s.mcp_logistics_url},
                "aftersale": {"transport": "http", "url": s.mcp_aftersale_url},
            }
        )
    return _client


async def fetch_mcp_tools() -> list[tuple[str, list]]:
    """返回 [(server_name, tools)]；失败降级空清单。每轮调用=真拉 tools/list（现问现拿）。"""
    client = _get_client()
    out: list[tuple[str, list]] = []
    for server in ("logistics", "aftersale"):
        try:
            tools = await client.get_tools(server_name=server)
            out.append((server, tools))
        except Exception:
            logger.warning("MCP Server %s 工具拉取失败（降级为空,不影响聊天）", server, exc_info=True)
    return out


async def merge_mcp_tools(registry) -> int:
    """把 MCP 工具合并进注册中心（source=mcp:<server>,重名跳过）。返回合并成功数。"""
    merged = 0
    for server, tools in await fetch_mcp_tools():
        for t in tools:
            if registry.register(t, source=f"mcp:{server}", on_duplicate="skip") is not None:
                merged += 1
    return merged
