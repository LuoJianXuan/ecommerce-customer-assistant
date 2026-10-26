"""切分器接缝（TDD）：五条切分规则 + 政策手册映射规则。

规则出处=ch03 需求原文：
1. 按标题层级做结构感知切分
2. 超长内容递归切
3. 块间加重叠且裁到最近句号，不留半截话
4. 大表格按行切时每块都复制表头
5. 映射：无天然问题文档 questions=所在最深层标题，category=上级标题路径
6. is_key_clause：标题含【关键条款】或正文以 **关键条款：** 开头 → 1
"""

from app.knowledge.splitter import MarkdownKnowledgeSplitter, SplitChunk

CHUNK_SIZE = 500
OVERLAP = 80


def make_splitter():
    return MarkdownKnowledgeSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP)


def test_rule1_split_by_header_levels():
    md = "# 退货政策\n七天无理由。\n## 运费说明\n质量问题平台承担运费。\n## 时效\n签收后7天内。"
    chunks = make_splitter().split(md)
    sections = [c.section_path for c in chunks]
    assert "退货政策>运费说明" in sections
    assert "退货政策>时效" in sections
    # 每块正文不含其他小节内容（结构感知：内容归属正确小节）
    yf = next(c for c in chunks if "运费说明" in c.section_path)
    assert "平台承担运费" in yf.answer
    assert "签收后" not in yf.answer


def test_rule2_long_content_recursively_split():
    para = "这一段讲退货运费的承担规则。" * 40  # ~440 字，远超 500 上限的一半
    md = f"# 手册\n## 长节\n{para}{para}"  # ~880 字正文 → 必须切成多块
    chunks = make_splitter().split(md)
    assert len([c for c in chunks if c.section_path == "手册>长节"]) >= 2


def test_rule3_overlap_and_sentence_boundary():
    para = "第一句讲时效。第二句讲运费。第三句讲凭证。第四句讲退款。第五句讲完结。"
    md = "# 手册\n## 长\n" + para * 20  # ~700 字
    chunks = [c for c in make_splitter().split(md) if c.section_path == "手册>长"]
    assert len(chunks) >= 2
    # 不留半截话：每块以句末标点收尾（。！？；）
    for c in chunks:
        assert c.answer.rstrip()[-1] in "。！？；", f"块尾半截话: ...{c.answer.rstrip()[-8:]}"
    # 块间有重叠：后块开头的内容在前块末尾出现过
    if len(chunks) >= 2:
        assert chunks[1].answer[:6] in chunks[0].answer or chunks[0].answer[-30:] in chunks[1].answer


def test_rule4_table_split_copies_header_to_every_chunk():
    rows = "\n".join(f"| 商品{i} | {i * 10}元 | 现货 |" for i in range(30))
    md = (
        "# 售后手册\n## 价目表\n| 商品 | 价格 | 库存 |\n|---|---|---|\n" + rows
    )
    chunks = [c for c in make_splitter().split(md) if "价目表" in c.section_path]
    assert len(chunks) >= 2  # 大表被按行切了
    for c in chunks:  # 每块都带表头与分隔行
        assert "| 商品 | 价格 | 库存 |" in c.answer
        assert "|---|---|---|" in c.answer


def test_rule5_mapping_questions_and_category():
    md = "# 退货政策\n## 运费说明\n质量问题平台承担运费。"
    chunk = make_splitter().split(md)[0]
    assert chunk.questions == "运费说明"          # questions=最深层标题
    assert chunk.category == "退货政策"            # category=上级标题路径
    assert chunk.content_type == "policy"          # 政策/手册类


def test_rule6_key_clause_detection():
    md = (
        "# 退货政策\n## 普通\n普通内容。\n"
        "## 【关键条款】七天时限\n七天无理由是法定底线。\n"
        "## 另一节\n**关键条款：**赔付需专员核实。"
    )
    chunks = {c.section_path.split(">")[-1]: c for c in make_splitter().split(md)}
    assert chunks["普通"].is_key_clause == 0
    assert chunks["【关键条款】七天时限"].is_key_clause == 1
    assert chunks["另一节"].is_key_clause == 1  # 正文以 **关键条款：** 开头
