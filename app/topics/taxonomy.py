"""17 类主题术语表:全系统单一权威源(spec: dev-notes/ch10-spec.md §2)。

造数 prompt、预标 prompt、训练标签空间、推理输出映射、前端展示五处共用本表;
顺序即权威顺序(退换货/物流/尺码/发票领头,其他殿后),禁止在别处复制类目清单。
"""

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Topic:
    name: str
    boundary: str  # 一句边界说明,近邻类目靠它划开


TOPICS: tuple[Topic, ...] = (
    Topic("退换货", "退货、换货、退款申请与进度等售后诉求;「修」字维修诉求归保修维修,「退」归退换货。"),
    Topic("物流", "发货、在途、签收、慢、丢件等货物流转问题;只管货,运费收不收/多少钱归运费。"),
    Topic("尺码", "尺码/大小/偏大偏小怎么选;若同句还表达了退换诉求,则同时打退换货(多标签)。"),
    Topic("发票", "开票、抬头、税点、电子发票补开重开等发票事宜。"),
    Topic("质量问题", "商品到手后的破损、故障、与描述不符等缺陷;想「修」归保修维修,想退换归退换货。"),
    Topic("运费", "运费收不收、多少钱、谁承担、免运费门槛;运费管钱,物流管货——货在哪/多久到归物流。"),
    Topic("优惠活动", "优惠券、满减、折扣活动的规则与领取使用;券和满减归优惠活动,买完降价补差价归价保。"),
    Topic("价保", "价保是补差价:成交后降价申请补差;券和满减的规则问题归优惠活动。"),
    Topic("支付", "支付方式、支付失败、重复扣款、分期等付钱环节。"),
    Topic("订单修改", "改地址、改电话、改数量规格、取消订单等订单信息变更。"),
    Topic("库存补货", "缺货、何时补货、到货提醒等货源问题。"),
    Topic("商品信息", "售前商品参数、材质、用法、适配等咨询,不含质量缺陷与售后诉求。"),
    Topic("保修维修", "「修」字诉求:保修期、维修进度、送修;单纯退换退款归退换货。"),
    Topic("账号", "登录、注册、密码、绑定手机/第三方等账号问题。"),
    Topic("会员积分", "会员等级权益、积分获取/兑换/余额。"),
    Topic("评价", "评价晒单、评价有礼、改评删评、无法评价。"),
    Topic("其他", "闲聊问候与无法归入前 16 类的超范围问题;能归类的先归类,不硬塞其他。"),
)

TOPIC_NAMES: tuple[str, ...] = tuple(t.name for t in TOPICS)
_TOPIC_INDEX: dict[str, int] = {name: i for i, name in enumerate(TOPIC_NAMES)}
N_TOPICS: int = len(TOPICS)


def validate_labels(labels: Sequence[str]) -> list[str]:
    """校验标签集:非空、全部在权威表内;返回按权威顺序排序去重后的列表。"""
    seen = set()
    for lab in labels:
        if lab not in _TOPIC_INDEX:
            raise ValueError(f"未知主题标签「{lab}」,合法标签以 app/topics/taxonomy.py 为准")
        seen.add(lab)
    if not seen:
        raise ValueError("标签集不能为空:无法归类时应打「其他」")
    return [name for name in TOPIC_NAMES if name in seen]


def labels_to_vector(labels: Sequence[str]) -> list[int]:
    """标签集 → 17 维 multi-hot 向量(训练/推理共用口径)。"""
    valid = validate_labels(labels)
    vec = [0] * N_TOPICS
    for lab in valid:
        vec[_TOPIC_INDEX[lab]] = 1
    return vec


def vector_to_labels(vec: Sequence[int]) -> list[str]:
    """multi-hot 向量 → 按权威顺序的标签名列表。"""
    return [TOPIC_NAMES[i] for i, v in enumerate(vec) if v]


def taxonomy_prompt_block() -> str:
    """LLM 共用的类目清单文本(造数/预标/复判 prompt 同源,杜绝第二份定义)。

    多标签口径(2026-10-20 抽审门修订):按「独立诉求」打标,不按关键词命中打标。
    """
    lines = ["## 主题类目表(共 17 类,按权威顺序)"]
    for i, t in enumerate(TOPICS, 1):
        lines.append(f"{i}. {t.name}:{t.boundary}")
    lines.append(
        "## 多标签口径:每个需要客服独立回答或处理的明确诉求打一个标签;"
        "背景信息(订单经过、支付经历、商品来源、身份限定词、假设性后续动作)不单独打标签。规则:\n"
        "1.「其他」是独占兜底:仅当无法归入前 16 类时单独使用,绝不与可归类标签共存;"
        "寒暄、语气词、「顺便问」不构成独立的「其他」诉求;\n"
        "2. 商品来源与经历是背景:「积分兑换的礼品发什么快递」→ [物流];「积分兑换的东西能退吗」→ [退换货];\n"
        "3. 会员身份只有决定所问权益时才打会员积分:「玄武星会员的跨星球运费」→ [运费, 会员积分];"
        "「这个折扣是只针对会员吗」→ [优惠活动, 会员积分];普通身份修饰词不标;\n"
        "4. 红包是优惠不是支付:「用了红包付款,退款的话红包退不退」→ [退换货, 优惠活动];\n"
        "5. 质量现象与售后方式按当前诉求分记:「快递到货了但外包装破了,我想换一个」→ [退换货, 质量问题];"
        "「这机器修过一次又坏了,能不能直接退」→ [退换货, 质量问题](「修过」是历史,当前要退不再记保修维修);\n"
        "6. 独立诉求示例:「买大了想退」→ [退换货, 尺码];「这个插座插不进去,是不是不配套」→ [商品信息]"
    )
    return "\n".join(lines)
