"""ch08 插件目录：约定优于配置的自动发现（拍板①）。

任何 .py 模块暴露以下任一模块级变量即被启动扫描登记：
- TOOLS: list[BaseTool]——现成工具（不依赖会话上下文）
- FACTORIES: list[(session_factory, conversation_id, llm) -> BaseTool]——需要会话/模型注入的工厂

验收①口径：新工具=在本目录放一个文件+重启服务，零核心代码改动；
免重启的热更新是 MCP 侧的事（验收③），两机制各管一段。
单个插件加载失败只告警，不影响其余注册。
"""

import importlib
import logging
import pkgutil

logger = logging.getLogger(__name__)


def discover_plugins(package: str = "app.tools.plugins"):
    """扫描插件包，返回 (TOOLS 现成工具, FACTORIES 工厂列表)。"""
    tools: list = []
    factories: list = []
    try:
        module = importlib.import_module(package)
    except ModuleNotFoundError:
        return tools, factories
    for mod in pkgutil.iter_modules(module.__path__):
        try:
            m = importlib.import_module(f"{package}.{mod.name}")
            tools.extend(getattr(m, "TOOLS", None) or [])
            factories.extend(getattr(m, "FACTORIES", None) or [])
        except Exception:
            logger.exception("插件加载失败（跳过）: %s", mod.name)
    return tools, factories
