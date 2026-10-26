"""Markdown 结构感知切分器（ch03 需求五规则 + 映射规则）。

流程：MarkdownHeaderTextSplitter 按标题层级切（携带标题路径 metadata）→
每个小节内部：表格块按行切（每块复制表头）+ 文本块递归切（句号进分隔符、带重叠）→
块尾裁到最近句末标点，不留半截话。

映射规则（政策/手册类，无天然问题）：
- questions = 所在最深层标题
- category  = 上级标题路径（如"退货政策"；单级标题时用自身）
- is_key_clause = 标题含【关键条款】或正文以 **关键条款：** 开头

FAQ 导入与对话挖矿不走本切分器（questions=真实问法）。
"""

import re
from dataclasses import dataclass

SENTENCE_END = "。！？；"
KEY_TITLE_MARK = "【关键条款】"
KEY_BODY_PREFIX = "**关键条款：**"


@dataclass
class SplitChunk:
    category: str
    questions: str
    answer: str
    section_path: str
    content_type: str
    is_key_clause: int
    source_order: int  # 文档内顺序，落库后用于回填 prev/next 指针


def _trim_to_sentence_end(text: str) -> str:
    """块尾裁到最近句末标点；全文无句末标点则原样返回（无从裁起）。"""
    stripped = text.rstrip()
    if not stripped or stripped[-1] in SENTENCE_END:
        return stripped
    for i in range(len(stripped) - 1, -1, -1):
        if stripped[i] in SENTENCE_END:
            return stripped[: i + 1]
    return stripped


def _split_table_block(lines: list[str], chunk_size: int) -> list[list[str]]:
    """表格按行切：表头两行（列名+分隔行）复制到每个块。"""
    if len(lines) < 3:  # 表头+分隔+至少一行数据才算表
        return [lines] if lines else []
    header = lines[:2]
    body = lines[2:]
    header_len = sum(len(l) + 1 for l in header)
    chunks: list[list[str]] = []
    current: list[str] = []
    current_len = header_len
    for row in body:
        cost = len(row) + 1
        if current and current_len + cost > chunk_size:
            chunks.append(header + current)
            current = []
            current_len = header_len
        current.append(row)
        current_len += cost
    if current:
        chunks.append(header + current)
    return chunks


class MarkdownKnowledgeSplitter:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 80, content_type: str = "policy") -> None:
        # 懒加载：SplitChunk 数据类被 repository/tests 复用，不该被重依赖拖住
        from langchain_text_splitters import (
            MarkdownHeaderTextSplitter,
            RecursiveCharacterTextSplitter,
        )

        self.chunk_size = chunk_size
        self.content_type = content_type
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
            strip_headers=True,
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "；", "，", " ", ""],
            keep_separator=True,  # 分隔符归属前段：句号留在句尾
            length_function=len,
        )

    def split(self, markdown: str) -> list[SplitChunk]:
        docs = self.header_splitter.split_text(markdown)
        chunks: list[SplitChunk] = []
        order = 0
        for doc in docs:
            path = [doc.metadata[k] for k in ("h1", "h2", "h3") if doc.metadata.get(k)]
            if not path:  # 无标题的散落正文：挂到"未分类"
                path = ["未分类"]
            section_path = ">".join(path)
            deepest = path[-1]
            category = ">".join(path[:-1]) or path[0]
            is_key = 1 if (KEY_TITLE_MARK in deepest or doc.page_content.lstrip().startswith(KEY_BODY_PREFIX)) else 0

            for piece in self._split_section_body(doc.page_content):
                chunks.append(
                    SplitChunk(
                        category=category,
                        questions=deepest,
                        answer=piece,
                        section_path=section_path,
                        content_type=self.content_type,
                        is_key_clause=is_key,
                        source_order=order,
                    )
                )
                order += 1
        return chunks

    def _split_section_body(self, body: str) -> list[str]:
        """小节正文切块：表格行块与文本块分别处理，保持出现顺序。"""
        pieces: list[str] = []
        lines = body.splitlines()
        buf: list[str] = []
        table_buf: list[str] = []

        def flush_text() -> None:
            if buf:
                text = "\n".join(buf).strip()
                if text:
                    for seg in self.text_splitter.split_text(text):
                        trimmed = _trim_to_sentence_end(seg)
                        if trimmed:
                            pieces.append(trimmed)
                buf.clear()

        def flush_table() -> None:
            if table_buf:
                for tbl in _split_table_block(table_buf, self.chunk_size):
                    pieces.append("\n".join(tbl))
                table_buf.clear()

        for line in lines:
            if line.lstrip().startswith("|"):
                flush_text()
                table_buf.append(line)
            else:
                flush_table()
                buf.append(line)
        flush_table()
        flush_text()
        return pieces
