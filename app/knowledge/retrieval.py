"""在线检索（ch04：混合检索 + 重排 + 元数据过滤 + Query 理解挂点）。

策略：
- dense：仅向量
- bm25：仅全文
- hybrid：两路各 recall 条 → RRF 融合
- hybrid_rerank：hybrid 取 rerank 候选 → bge-reranker-v2-m3 精排 → top_k

Milvus 直出 text+category；结构化三格与溯源字段（section_path 等）拿 chunk_id 回
MySQL 权威源补全。query_texts 支持"改写+同义词"多查询（Query 理解在检索侧生效）：
dense 用归一后主查询，BM25 用全部变体拼接（词面并集）。
"""

from collections.abc import Sequence
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.knowledge.models import KnowledgeChunk


def _escape_expr_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def build_category_expr(
    filter_category: str | None,
    excluded_categories: str | None,
) -> str:
    """构建 Milvus 类别过滤式；默认排除未审核的历史挖掘知识。"""
    parts: list[str] = []
    if filter_category:
        parts.append(f'category == "{_escape_expr_value(filter_category)}"')
    excluded = {
        item.strip()
        for item in (excluded_categories or "").split(",")
        if item.strip()
    }
    for category in sorted(excluded):
        parts.append(f'category != "{_escape_expr_value(category)}"')
    return " and ".join(f"({part})" for part in parts)


def reorder_for_prompt(items: list[Any]) -> list[Any]:
    """首尾组装：最相关的放头、次相关放尾（lost-in-the-middle 缓解）。"""
    if len(items) < 3:
        return list(items)
    return [items[0], *items[2:], items[1]]


def retrieve_multi_query(
    session_factory: sessionmaker[Session],
    queries: Sequence[str],
    final_k: int = 5,
    per_query_k: int = 10,
    merge_candidates: int = 12,
    filter_category: str | None = None,
    store=None,
    embedder=None,
    reranker=None,
) -> list[dict]:
    """ch06 多路检索合并（查询扩写的消费端）。

    与 retrieve(query_texts=...) 的区别：那里是「一次 hybrid、变体并进 BM25 词面」；
    这里每条查询独立 hybrid 召回（各带自己的 dense 向量与排序，侧重不互相稀释），
    再按 chunk_id 去重合并（同块保留最高分），最后**只精排一次**取 final_k——
    控制 reranker 开销（CPU 上 CrossEncoder 是大头）。
    """
    from app.config import get_settings

    settings = get_settings()
    queries = [q for q in (str(x).strip() for x in queries) if q]
    if not queries:
        return []
    if store is None:
        from app.knowledge.vector_store import MilvusKnowledgeStore

        store = MilvusKnowledgeStore()
    if embedder is None:
        from app.knowledge.embeddings import BgeM3Embedder

        embedder = BgeM3Embedder.get()
    if reranker is None:
        from app.knowledge.reranker import BgeReranker

        reranker = BgeReranker.get()

    expr = build_category_expr(
        filter_category,
        settings.retrieval_excluded_categories,
    )

    best: dict[int, dict] = {}
    for q in queries:
        hits = store.hybrid_search(
            embedder.embed_query(q), q,
            recall=settings.recall_top_k, limit=per_query_k, expr=expr,
        )
        for h in hits:
            cid = h["chunk_id"]
            if cid not in best or (h.get("score") or 0) > (best[cid].get("score") or 0):
                best[cid] = h
    merged = sorted(best.values(), key=lambda h: h.get("score") or 0, reverse=True)
    merged = merged[:merge_candidates]
    reranked = reranker.rerank(queries[0], merged, final_k) if merged else []
    return _enrich_from_mysql(session_factory, reranked)


def retrieve(
    session_factory: sessionmaker[Session],
    query: str,
    top_k: int | None = None,
    strategy: str | None = None,
    filter_category: str | None = None,
    query_texts: Sequence[str] | None = None,
    store=None,
    embedder=None,
    reranker=None,
) -> list[dict]:
    from app.config import get_settings

    settings = get_settings()
    strategy = strategy or settings.retrieval_strategy
    top_k = top_k or settings.retrieval_top_k
    query_texts = list(query_texts) if query_texts else [query]

    if store is None:
        from app.knowledge.vector_store import MilvusKnowledgeStore

        store = MilvusKnowledgeStore()
    if embedder is None:
        from app.knowledge.embeddings import BgeM3Embedder

        embedder = BgeM3Embedder.get()

    expr = build_category_expr(
        filter_category,
        settings.retrieval_excluded_categories,
    )

    if strategy == "dense":
        hits = store.search_dense(embedder.embed_query(query_texts[0]), top_k, expr)
    elif strategy == "bm25":
        hits = store.search_bm25(" ".join(query_texts), top_k, expr)
    elif strategy in ("hybrid", "hybrid_rerank"):
        if reranker is None and strategy == "hybrid_rerank":
            from app.knowledge.reranker import BgeReranker

            reranker = BgeReranker.get()
        limit = (
            settings.rerank_candidates if strategy == "hybrid_rerank" else top_k
        )
        hits = store.hybrid_search(
            embedder.embed_query(query_texts[0]),
            " ".join(query_texts),
            recall=settings.recall_top_k,
            limit=limit,
            expr=expr,
        )
        if strategy == "hybrid_rerank":
            hits = reranker.rerank(query_texts[0], hits, top_k)
    else:
        raise ValueError(f"未知检索策略: {strategy}")

    return _enrich_from_mysql(session_factory, hits)


def _enrich_from_mysql(session_factory, hits: list[dict]) -> list[dict]:
    """MySQL 权威源补全结构化三格与溯源元数据。"""
    if not hits:
        return []
    with session_factory() as session:
        rows = session.scalars(
            select(KnowledgeChunk).where(
                KnowledgeChunk.id.in_([h["chunk_id"] for h in hits])
            )
        ).all()
    by_id = {r.id: r for r in rows}
    results = []
    for h in hits:
        row = by_id.get(h["chunk_id"])
        if row is None or getattr(row, "governance_status", "active") != "active":
            continue  # 向量指向的块已被替换删除（重导窗口期）
        results.append(
            {
                "chunk_id": h["chunk_id"],
                "question": row.questions,
                "answer": row.answer,
                "category": row.category,
                "section_path": row.section_path,
                "score": h.get("score"),
                "recall_score": h.get("recall_score"),
                "text": h.get("text", ""),
            }
        )
    return results
