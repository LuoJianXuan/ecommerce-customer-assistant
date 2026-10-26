"""工具包：内置件套 + 插件自动发现（ch08），经 build_default_registry 每轮按会话构建。

注册表收录 ToolRecord（三要素+来源+权限）；MCP 工具由 mcp_client.merge_mcp_tools
每轮现问现拿后合并（react 侧 await 调用,拍板②）。
create_ticket 等会话工厂经闭包注入；query_faq 按工具覆盖执行超时。
"""

from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings
from app.tools.base import ToolRegistry
from app.tools.ecommerce import make_list_orders, make_query_order, query_product
from app.tools.faq import make_query_faq
from app.tools.plugins import discover_plugins
from app.tools.tickets import make_create_ticket


def build_default_registry(
    session_factory: sessionmaker[Session] | None, conversation_id: int, llm=None
) -> ToolRegistry:
    settings = get_settings()
    registry = ToolRegistry()
    # ch06 修复轮：订单类工具按会话定位用户、查 orders 表（统一订单源，不再随机造数）
    registry.register(make_list_orders(session_factory, conversation_id))
    registry.register(make_query_order(session_factory, conversation_id))
    registry.register(query_product)
    registry.register(
        make_query_faq(session_factory, llm=llm, conversation_id=conversation_id),
        timeout_seconds=settings.faq_tool_timeout_seconds,
    )
    registry.register(make_create_ticket(session_factory, conversation_id))
    # ch08 M3：物流查询移交 MCP Server（内置 query_logistics 下线，重名由注册中心兜底拒绝）
    # ch08：插件目录自动扫描（拍板①）——TOOLS 现成工具 + FACTORIES 会话工厂
    plugin_tools, plugin_factories = discover_plugins()
    for t in plugin_tools:
        registry.register(t)
    for factory in plugin_factories:
        registry.register(factory(session_factory, conversation_id, llm))
    return registry
