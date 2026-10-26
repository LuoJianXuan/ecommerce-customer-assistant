"""ch08 物流 MCP Server（独立进程，端口 8101）：查物流轨迹。

- MCP 官方 Python SDK，Streamable HTTP 传输（默认 /mcp 路径）;
- 内部照第 2 章工具的做法随机 mock（订单号做种子，同单稳定），不接真实系统、不建表;
- 验收③机关：环境变量 MCP_DEMO_EXTRA_TOOL=1 时额外暴露 check_delivery_fee——
  演示「Server 侧加工具只重启本进程，客服系统零改动零重启，下一轮即用」
  （放物流 Server 而非售后：物流/订单意图直达 Agent，新工具调用确定性高——评审改定）;
- 启动：python mcp_servers/logistics_server.py
"""

import json
import os
import random

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("logistics", host="127.0.0.1", port=8101)

CARRIERS = ["顺丰速运", "中通快递", "圆通速递", "京东物流"]
NODES = ["杭州转运中心", "上海浦东分拨中心", "广州白云集散点", "北京顺义集散中心", "成都分拨中心"]


@mcp.tool()
def query_logistics(order_id: str) -> str:
    """按订单号查询物流轨迹：承运公司、运单号、当前状态、最新节点与时间。查不到会如实返回未找到。

    Args:
        order_id: 订单号（完整单号如 SO-XXXXXX-02，或末尾短号如 02）
    """
    if not order_id.strip():
        return json.dumps({"error": "订单号不能为空"}, ensure_ascii=False)
    rng = random.Random(f"logistics:{order_id}")
    status = rng.choice(["运输中", "派送中", "已签收"])
    return json.dumps(
        {
            "order_id": order_id,
            "carrier": rng.choice(CARRIERS),
            "tracking_no": f"SF{rng.randint(10**13, 10**14 - 1)}",
            "status": status,
            "latest_node": f"已到达【{rng.choice(NODES)}】",
        },
        ensure_ascii=False,
    )


if os.environ.get("MCP_DEMO_EXTRA_TOOL") == "1":

    @mcp.tool()
    def check_delivery_fee(order_id: str) -> str:
        """查询订单的配送费/运费金额与包邮规则。验收③演示：Server 侧新加的工具。

        Args:
            order_id: 订单号
        """
        rng = random.Random(f"fee:{order_id}")
        return json.dumps(
            {
                "order_id": order_id,
                "delivery_fee": rng.choice([0, 0, 6]),
                "rule": "满99元包邮,不满99元收6元基础运费;大件家电免配送费",
            },
            ensure_ascii=False,
        )


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
