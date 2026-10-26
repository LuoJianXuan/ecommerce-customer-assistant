"""双写状态机接缝（TDD）：MySQL pending → Milvus upsert → 回填 done；两层幂等。

用 FakeVectorStore / FakeEmbedder（确定性向量），不依赖真 Milvus / 真模型。
"""

import pytest
from sqlalchemy import select

from app.knowledge.models import KnowledgeChunk
from app.knowledge.repository import (
    insert_chunks,
    replace_source,
    vectorize_pending,
)
from app.knowledge.splitter import SplitChunk


def chunk(n: int, path: str = "测试文档>第一节") -> SplitChunk:
    return SplitChunk(
        category="测试文档",
        questions=f"第{n}节标题",
        answer=f"这是第 {n} 块的正文内容。",
        section_path=path,
        content_type="policy",
        is_key_clause=0,
        source_order=n,
    )


class FakeEmbedder:
    def __init__(self):
        self.calls: list[str] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.extend(texts)
        return [[float(len(t)), 1.0] for t in texts]


class FakeVectorStore:
    def __init__(self, fail_on: set[str] | None = None):
        self.rows: dict[str, dict] = {}  # vector_id -> record
        self.fail_on = fail_on or set()

    def upsert(self, vector_id: str, chunk_id: int, text: str, category: str, vector: list[float]) -> None:
        if str(chunk_id) in self.fail_on:
            raise RuntimeError(f"injected failure for chunk {chunk_id}")
        self.rows[vector_id] = {"chunk_id": chunk_id, "vector": vector}

    def delete_by_chunk_ids(self, chunk_ids: list[int]) -> None:
        doomed = {v for v, r in self.rows.items() if r["chunk_id"] in chunk_ids}
        for v in doomed:
            self.rows.pop(v)


@pytest.fixture
def db(require_db):
    from app.db import get_session_factory

    factory = get_session_factory()
    with factory() as s:
        yield factory, s
        s.query(KnowledgeChunk).filter(KnowledgeChunk.category == "测试文档").delete()
        s.commit()


def test_insert_chunks_pending_with_two_phase_pointers(db):
    factory, s = db
    ids = insert_chunks(s, [chunk(0), chunk(1), chunk(2)])
    s.commit()
    rows = s.scalars(
        select(KnowledgeChunk).where(KnowledgeChunk.id.in_(ids)).order_by(KnowledgeChunk.id)
    ).all()
    assert all(r.vectorize_status == "pending" for r in rows)
    assert rows[0].prev_chunk_id is None and rows[0].next_chunk_id == rows[1].id
    assert rows[1].prev_chunk_id == rows[0].id and rows[1].next_chunk_id == rows[2].id
    assert rows[2].prev_chunk_id == rows[1].id and rows[2].next_chunk_id is None


def test_vectorize_pending_marks_done_and_backfills(db):
    factory, s = db
    ids = insert_chunks(s, [chunk(0), chunk(1)])
    s.commit()
    store, embedder = FakeVectorStore(), FakeEmbedder()

    vectorize_pending(factory, embedder, store)

    rows = s.scalars(select(KnowledgeChunk).where(KnowledgeChunk.id.in_(ids))).all()
    assert all(r.vectorize_status == "done" for r in rows)
    assert all(r.vector_id == f"kb-{r.id}" for r in rows)  # 确定性主键=幂等根基
    assert set(store.rows) == {f"kb-{i}" for i in ids}
    # 向量化文本 = category\nquestions\nanswer 三行拼接
    assert any("测试文档\n第0节标题\n这是第 0 块的正文内容。" == t for t in embedder.calls)


def test_vectorize_pending_skips_quarantined_chunks(db):
    factory, s = db
    ids = insert_chunks(s, [chunk(0), chunk(1)])
    row = s.get(KnowledgeChunk, ids[1])
    row.governance_status = "quarantined"
    row.governance_reason = "test"
    s.commit()

    store, embedder = FakeVectorStore(), FakeEmbedder()
    vectorize_pending(factory, embedder, store)

    assert set(store.rows) == {f"kb-{ids[0]}"}
    with factory() as s2:
        quarantined = s2.get(KnowledgeChunk, ids[1])
        assert quarantined.vectorize_status == "pending"
        assert quarantined.vector_id is None


def test_interrupt_then_resume_picks_up_missing(db):
    """验收2核心：中途挂掉 → 已完成的不动、pending 的重跑补齐、无孤儿重复。"""
    factory, s = db
    ids = insert_chunks(s, [chunk(0), chunk(1), chunk(2)])
    s.commit()

    broken = FakeVectorStore(fail_on={str(ids[1])})  # 第二块 upsert 失败
    with pytest.raises(RuntimeError):
        vectorize_pending(factory, FakeEmbedder(), broken)
    assert set(broken.rows) == {f"kb-{ids[0]}"}  # 失败点前只灌了第一块

    with factory() as s2:
        statuses = {
            r.id: r.vectorize_status
            for r in s2.scalars(select(KnowledgeChunk).where(KnowledgeChunk.id.in_(ids)))
        }
    assert statuses[ids[0]] == "done"      # 失败点之前已回填
    assert statuses[ids[1]] == "pending"   # 失败块保持 pending
    assert statuses[ids[2]] == "pending"

    # 修好后重跑：只补 pending 两块；已 done 的不重灌（真幂等）
    healthy = FakeVectorStore()
    vectorize_pending(factory, FakeEmbedder(), healthy)
    with factory() as s3:
        rows = s3.scalars(select(KnowledgeChunk).where(KnowledgeChunk.id.in_(ids))).all()
    assert all(r.vectorize_status == "done" for r in rows)
    assert set(healthy.rows) == {f"kb-{ids[1]}", f"kb-{ids[2]}"}


def test_replace_source_rebuild_is_idempotent(db):
    """同一文档重复导入：旧块+旧向量整体替换，不堆积。"""
    factory, s = db
    store = FakeVectorStore()
    ids1 = replace_source(s, "测试文档", [chunk(0), chunk(1)], store=store)
    s.commit()
    vectorize_pending(factory, FakeEmbedder(), store)
    assert len(store.rows) == 2

    # 文档改版多了一块；同一个 store 延续（等价于真 Milvus 未清库）
    ids2 = replace_source(s, "测试文档", [chunk(0), chunk(1), chunk(2)], store=store)
    s.commit()
    vectorize_pending(factory, FakeEmbedder(), store)

    assert set(ids1).isdisjoint(set(ids2))  # 新行全新 id
    assert set(store.rows) == {f"kb-{i}" for i in ids2}  # 旧向量被删，只剩新块
    with factory() as s4:
        remain = s4.scalars(
            select(KnowledgeChunk).where(KnowledgeChunk.id.in_(ids1))
        ).all()
        assert remain == []                  # 旧行已删
