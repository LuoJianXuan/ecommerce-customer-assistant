"""ch06 多路检索合并（TDD）：逐查询 hybrid 召回 → chunk_id 去重保最高分 → 单次精排。

全 fake（store/embedder/reranker/mysql 富集），不碰 Milvus 与 MySQL。
"""

import pytest

from app.knowledge.retrieval import retrieve_multi_query


class FakeStore:
    """按查询文本返回脚本化的 hybrid 命中。"""

    def __init__(self, hits_by_query: dict[str, list[dict]]):
        self.hits_by_query = hits_by_query
        self.seen_queries: list[str] = []

    def hybrid_search(self, dense_vec, text, recall, limit, expr=""):
        self.seen_queries.append(text)
        return list(self.hits_by_query.get(text, []))


class FakeEmbedder:
    def embed_query(self, q):
        return q  # 直接拿查询文本当「向量」，FakeStore 按文本回放


class FakeReranker:
    def __init__(self):
        self.calls: list[tuple] = []

    def rerank(self, query, hits, top_k):
        self.calls.append((query, [h["chunk_id"] for h in hits], top_k))
        return sorted(hits, key=lambda h: h["score"], reverse=True)[:top_k]


@pytest.fixture
def enrich_stub(monkeypatch):
    """富集直接透传（不查 MySQL）。"""
    monkeypatch.setattr("app.knowledge.retrieval._enrich_from_mysql", lambda sf, hits: hits)


def _hit(cid: int, score: float) -> dict:
    return {"chunk_id": cid, "score": score, "text": f"chunk-{cid}"}


def test_multi_query_merges_and_dedups_keep_best_score(enrich_stub):
    store = FakeStore({
        "耳机能退吗": [_hit(1, 0.9), _hit(2, 0.8)],
        "退货政策条件": [_hit(2, 0.95), _hit(3, 0.7)],  # chunk 2 两路都召回，保留 0.95
    })
    rr = FakeReranker()
    got = retrieve_multi_query(
        None, ["耳机能退吗", "退货政策条件"],
        final_k=3, store=store, embedder=FakeEmbedder(), reranker=rr,
    )
    assert store.seen_queries == ["耳机能退吗", "退货政策条件"]  # 每条查询独立召回
    ids = [h["chunk_id"] for h in got]
    assert sorted(ids) == [1, 2, 3]  # 去重合并
    chunk2 = next(h for h in got if h["chunk_id"] == 2)
    assert chunk2["score"] == 0.95  # 同块取最高分
    assert len(rr.calls) == 1  # 只精排一次


def test_multi_query_rerank_input_capped_and_final_k(enrich_stub):
    hits = [_hit(i, 0.5 + i / 100) for i in range(1, 21)]  # 20 块
    store = FakeStore({"q": hits})
    rr = FakeReranker()
    got = retrieve_multi_query(
        None, ["q"], final_k=5, merge_candidates=8,
        store=store, embedder=FakeEmbedder(), reranker=rr,
    )
    assert len(got) == 5
    rerank_input_ids = rr.calls[0][1]
    assert len(rerank_input_ids) == 8  # 合并集先截断到 merge_candidates 再进精排


def test_multi_query_empty_queries_returns_empty(enrich_stub):
    store = FakeStore({})
    rr = FakeReranker()
    got = retrieve_multi_query(None, ["  ", ""], store=store, embedder=FakeEmbedder(), reranker=rr)
    assert got == []
    assert rr.calls == []
