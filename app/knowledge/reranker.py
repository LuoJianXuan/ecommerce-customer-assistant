"""bge-reranker-v2-m3 重排器：CrossEncoder 懒加载单例 + 线程锁（同 Embedder 模式）。

输入 [query, passage] 对，输出相关性分（越高越相关）。
CPU 上 0.6B 模型按候选量控制延迟（默认 Top-20 进、Top-10 出）。
"""

import threading

from app.config import get_settings


class BgeReranker:
    _instance: "BgeReranker | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        import os

        from app.knowledge.embeddings import _prepare_hf_env

        _prepare_hf_env()
        from sentence_transformers import CrossEncoder
        import torch

        settings = get_settings()
        model_kwargs = {"torch_dtype": getattr(torch, settings.local_model_dtype)}
        try:
            self._model = CrossEncoder(
                settings.rerank_model_name, model_kwargs=model_kwargs
            )
        except Exception:
            os.environ.pop("HF_HUB_OFFLINE", None)  # 缓存缺失回退联网下载
            self._model = CrossEncoder(
                settings.rerank_model_name, model_kwargs=model_kwargs
            )

    @classmethod
    def get(cls) -> "BgeReranker":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        """按 query 对候选（含 text 字段）重打分排序，取 top_k。

        rerank 分覆盖 score，原始检索分保留为 recall_score。
        """
        if not candidates:
            return []
        pairs = [[query, c.get("text") or ""] for c in candidates]
        scores = self._model.predict(pairs)
        ranked = sorted(
            zip(candidates, scores), key=lambda cs: float(cs[1]), reverse=True
        )[:top_k]
        out = []
        for cand, score in ranked:
            item = dict(cand)
            item["recall_score"] = item.get("score")
            item["score"] = float(score)
            out.append(item)
        return out
