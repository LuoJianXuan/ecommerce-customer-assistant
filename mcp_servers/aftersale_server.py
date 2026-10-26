"""ch08 售后 MCP Server（独立进程，端口 8102）：查在保、查退货进度。

- MCP 官方 Python SDK，Streamable HTTP 传输;
- mock 口径同第 2 章（订单号做种子）;
- 验收③的 extra 工具开关在物流 Server（logistics_server.py）——物流/订单意图直达
  Agent，新工具调用确定性高（评审改定，原 spec 写在售后侧）;
- 启动：python mcp_servers/aftersale_server.py
"""

import json
import random

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("aftersale", host="127.0.0.1", port=8102)


@mcp.tool()
def check_warranty(order_id: str, product_name: str = "") -> str:
    """查询商品是否在保修期内：保修状态、保修期长度、到期日。查不到如实返回未找到。

    Args:
        order_id: 订单号
        product_name: 商品名称（可选，用于核对）
    """
    rng = random.Random(f"warranty:{order_id}")
    in_warranty = rng.random() > 0.3
    years = rng.choice([1, 2, 3])
    return json.dumps(
        {
            "order_id": order_id,
            "product_name": product_name or "未提供",
            "warranty_status": "在保" if in_warranty else "已过保",
            "warranty_period": f"{years} 年",
            "expire_date": f"2027-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}" if in_warranty else "已过期",
        },
        ensure_ascii=False,
    )


@mcp.tool()
def query_refund_progress(order_id: str) -> str:
    """按订单号查询退货退款进度：当前环节、进入该环节的时间、预计时效。查不到如实返回未找到。

    Args:
        order_id: 订单号
    """
    rng = random.Random(f"refund:{order_id}")
    stage = rng.choice(["已申请待审核", "审核中", "待寄回", "仓库验收中", "退款中", "已退款"])
    return json.dumps(
        {
            "order_id": order_id,
            "stage": stage,
            "updated_at": f"2026-10-{rng.randint(1, 12):02d} {rng.randint(8, 20):02d}:00",
            "eta": "1-3 个工作日" if stage in ("仓库验收中", "退款中") else "以审核结果为准",
        },
        ensure_ascii=False,
    )


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
