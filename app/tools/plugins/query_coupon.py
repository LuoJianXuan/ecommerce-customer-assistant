"""示例插件（验收①蓝本）：优惠券查询——纯静态 mock，演示「放一个文件即被登记」。

约定：模块级 TOOLS=现成工具；FACTORIES=会话工厂。两类可同时暴露。
"""

import json

from langchain_core.tools import tool

COUPONS = [
    {"coupon": "满99包邮券", "condition": "实物订单无门槛", "expires": "2026-12-31"},
    {"coupon": "耳机满300减50券", "condition": "无线降噪耳机 E-3 专享", "expires": "2026-11-30"},
    {"coupon": "会员积分翻倍券", "condition": "黄金会员及以上", "expires": "2026-10-31"},
]


@tool
def query_coupon(keyword: str = "") -> str:
    """查询当前可领取的优惠券信息，可按商品关键词筛选。用户问优惠券/优惠/省钱/有没有券时调用。

    Args:
        keyword: 商品关键词，如"耳机"；空字符串列出全部
    """
    hits = [c for c in COUPONS if not keyword.strip() or keyword in c["coupon"]] or COUPONS
    return json.dumps({"coupons": hits}, ensure_ascii=False)


TOOLS = [query_coupon]
