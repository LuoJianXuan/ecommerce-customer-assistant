"""BGE-M3 嵌入封装：懒加载单例，encode + L2 归一化（配合 Milvus COSINE）。

模型缓存走项目内 .hf_cache（HF_HOME，D 盘），下载经 hf-mirror（HF_ENDPOINT）。
文档核实（模型卡）：1024 维、sentence-transformers 默认输出 dense、无需 query 指令。
"""

import os
import threading
from pathlib import Path

from app.config import get_settings


def _prepare_hf_env() -> None:
    """HF 环境已在 app.config 模块级就位（必须早于 huggingface_hub 首次导入）；
    这里只负责确保缓存目录存在。幂等。"""
    settings = get_settings()
    cache_dir = (Path(__file__).resolve().parent.parent.parent / settings.hf_home).resolve()
    cache_dir.mkdir(exist_ok=True)


class BgeM3Embedder:
    """进程内懒加载单例：首次 embed 时才载模型（~2.3GB，避免拖慢测试启动）。

    get() 加锁防并发双载（lifespan 预热线程与工具执行线程可能同时首次调用）。
    """

    _instance: "BgeM3Embedder | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        _prepare_hf_env()
        from sentence_transformers import SentenceTransformer
        import torch

        settings = get_settings()
        model_kwargs = {"torch_dtype": getattr(torch, settings.local_model_dtype)}
        try:
            self._model = SentenceTransformer(
                settings.embed_model_name, model_kwargs=model_kwargs
            )
        except Exception:
            # 缓存缺失（首次建库/换模型）：回退联网经 hf-mirror 下载
            os.environ.pop("HF_HUB_OFFLINE", None)
            self._model = SentenceTransformer(
                settings.embed_model_name, model_kwargs=model_kwargs
            )
        self.dim = 1024

    @classmethod
    def get(cls) -> "BgeM3Embedder":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """批量编码并 L2 归一化（COSINE 度量下的标准做法）。"""
        vecs = self._model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vecs]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]


def build_vector_text(category: str, questions: str, answer: str) -> str:
    """向量化文本统一拼装：category\\nquestions\\nanswer 三行（计划 v4 缺口1）。"""
    return f"{category}\n{questions}\n{answer}"
