"""ch04 检索策略接缝（TDD）：首尾组装纯函数 + 策略分发（fake store/reranker）。"""

from app.knowledge.retrieval import build_category_expr, reorder_for_prompt, retrieve


def test_reorder_puts_best_at_head_and_second_at_tail():
    items = ["A", "B", "C", "D", "E"]  # 分数降序
    out = reorder_for_prompt(items)
    assert out[0] == "A"      # 最相关放头
    assert out[-1] == "B"     # 次相关放尾
    assert sorted(out) == sorted(items)  # 无丢失


def test_reorder_short_lists_identity():
    assert reorder_for_prompt(["A"]) == ["A"]
    assert reorder_for_prompt(["A", "B"]) == ["A", "B"]
    assert reorder_for_prompt([]) == []


class FakeStrategyStore:
    """记录三路调用并返回可区分结果的假向量库。"""

    def __init__(self):
        self.calls: list[str] = []

    def search_dense(self, vec, top_k, expr=""):
        self.calls.append(f"dense:{top_k}:{expr}")
        return [{"chunk_id": 1, "text": "t1", "category": "c", "score": 0.9}]

    def search_bm25(self, text, top_k, expr=""):
        self.calls.append(f"bm25:{top_k}:{text}:{expr}")
        return [{"chunk_id": 2, "text": "t2", "category": "c", "score": 8.0}]

    def hybrid_search(self, vec, text, recall=50, limit=10, expr=""):
        self.calls.append(f"hybrid:{recall}->{limit}")
        return [
            {"chunk_id": i, "text": f"t{i}", "category": "c", "score": 1.0 / i}
            for i in range(1, 6)
        ]


class FakeReranker:
    def rerank(self, query, candidates, top_k):
        # 反转顺序以证明 rerank 生效
        return list(reversed(candidates))[:top_k]


class FakeEmbedder:
    def embed_query(self, text):
        return [0.1]


def _session_factory():
    from types import SimpleNamespace

    class _S:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def scalars(self, *a, **k):
            class _R:
                def all(self):
                    return [
                        SimpleNamespace(
                            id=i, questions=f"q{i}", answer=f"a{i}",
                            category="c", section_path=f"s>{i}",
                        )
                        for i in range(1, 6)
                    ]

            return _R()

    return _S


def test_strategy_dispatch_dense_vs_bm25():
    s1, s2 = FakeStrategyStore(), FakeStrategyStore()
    retrieve(_session_factory(), "q", strategy="dense", store=s1, embedder=FakeEmbedder())
    retrieve(_session_factory(), "q", strategy="bm25", store=s2, embedder=FakeEmbedder())
    assert s1.calls[0].startswith("dense:3:")
    assert s2.calls[0].startswith("bm25:3:q")
    assert 'category != "对话挖掘"' in s1.calls[0]
    assert 'category != "对话挖掘"' in s2.calls[0]


def test_strategy_hybrid_uses_recall_then_limit():
    s = FakeStrategyStore()
    retrieve(_session_factory(), "q", strategy="hybrid", store=s, embedder=FakeEmbedder())
    assert s.calls == ["hybrid:50->3"]  # 召回 50 融合，非重排默认留 3


def test_strategy_hybrid_rerank_reorders():
    s = FakeStrategyStore()
    out = retrieve(
        _session_factory(), "q", strategy="hybrid_rerank",
        store=s, embedder=FakeEmbedder(), reranker=FakeReranker(),
    )
    from app.config import get_settings
    assert s.calls == [f"hybrid:50->{get_settings().rerank_candidates}"]  # 融合取 rerank 候选数
    assert out[0]["chunk_id"] == 5       # FakeReranker 反转 → 原第 5 名到头
    assert len(out) <= 10


def test_filter_category_passed_as_expr():
    s = FakeStrategyStore()
    retrieve(
        _session_factory(), "q", strategy="dense",
        store=s, embedder=FakeEmbedder(), filter_category="退货政策",
    )
    assert "退货政策" in s.calls[0]


def test_default_governance_filter_excludes_mined_knowledge():
    s = FakeStrategyStore()
    retrieve(_session_factory(), "q", strategy="dense", store=s, embedder=FakeEmbedder())
    assert 'category != "对话挖掘"' in s.calls[0]


def test_category_expr_escapes_values_and_combines_filters():
    expr = build_category_expr('退货"政策', "对话挖掘, 面试资料")
    assert 'category == "退货\\"政策"' in expr
    assert 'category != "对话挖掘"' in expr
    assert 'category != "面试资料"' in expr
