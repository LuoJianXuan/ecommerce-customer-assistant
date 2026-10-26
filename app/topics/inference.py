"""ch10 本地多标签主题推理；模型只在批量归类首次调用时按需加载。"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

from app.config import Settings, get_settings
from app.topics.taxonomy import TOPIC_NAMES

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class TopicPrediction:
    labels: list[str]
    scores: dict[str, float]


def decode_probabilities(
    rows: Sequence[Sequence[float]], threshold: float
) -> list[TopicPrediction]:
    """概率矩阵转权威顺序标签；允许空标签，供页面显示为“未识别”。"""
    if not 0 < threshold < 1:
        raise ValueError("主题分类阈值必须在 0 和 1 之间")
    out = []
    for values in rows:
        if len(values) != len(TOPIC_NAMES):
            raise ValueError(f"模型输出维度应为 {len(TOPIC_NAMES)},实际 {len(values)}")
        scores = {name: round(float(values[i]), 6) for i, name in enumerate(TOPIC_NAMES)}
        labels = [name for name in TOPIC_NAMES if scores[name] > threshold]
        out.append(TopicPrediction(labels=labels, scores=scores))
    return out


def validate_model_label_order(id2label: dict) -> None:
    """阻止同为 17 维但标签顺序不同的模型被静默接入。"""
    actual = [str(id2label.get(i, id2label.get(str(i), ""))) for i in range(len(TOPIC_NAMES))]
    if actual != list(TOPIC_NAMES):
        raise RuntimeError("主题模型 id2label 与权威术语表顺序不一致")


class LocalTopicClassifier:
    """Transformers 本地分类器；线程安全、延迟加载、批量推理。"""

    def __init__(self, settings: Settings):
        raw_path = Path(settings.topic_model_path)
        self.model_path = raw_path if raw_path.is_absolute() else PROJECT_ROOT / raw_path
        self.requested_device = settings.topic_device.lower()
        self.batch_size = settings.topic_batch_size
        self.fallback_threshold = settings.topic_threshold_fallback
        self._load_lock = threading.Lock()
        self._predict_lock = threading.Lock()
        self._model = None
        self._tokenizer = None
        self._device = None
        self.threshold = self.fallback_threshold
        self.model_version = self.model_path.name
        self._read_metadata()

    def _read_metadata(self) -> None:
        """只读轻量阈值元数据；查询已归类数据时不加载模型权重。"""
        metadata_path = self.model_path / "topic_threshold.json"
        if not metadata_path.exists():
            return
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            threshold = float(metadata["threshold"])
        except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"模型阈值文件无法读取:{metadata_path}") from exc
        if not 0 < threshold < 1:
            raise RuntimeError(f"模型阈值文件非法:{threshold}")
        self.threshold = threshold
        self.model_version = str(metadata.get("model_version") or self.model_path.name)

    def _load(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            if not self.model_path.is_dir():
                raise FileNotFoundError(f"主题模型目录不存在:{self.model_path}")

            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            if self.requested_device not in {"auto", "cuda", "cpu"}:
                raise ValueError("topic_device 仅支持 auto/cuda/cpu")
            if self.requested_device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("topic_device=cuda,但当前 CUDA 不可用")
            device = (
                "cuda"
                if self.requested_device == "cuda"
                or (self.requested_device == "auto" and torch.cuda.is_available())
                else "cpu"
            )

            tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path, local_files_only=True
            )
            validate_model_label_order(model.config.id2label)
            model = model.to(device)
            model.eval()
            self._tokenizer, self._model, self._device = tokenizer, model, device

    @property
    def device(self) -> str:
        self._load()
        return str(self._device)

    def predict(self, texts: Sequence[str]) -> list[TopicPrediction]:
        if not texts:
            return []
        self._load()
        import torch

        all_probs: list[list[float]] = []
        with self._predict_lock, torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                batch = list(texts[start : start + self.batch_size])
                encoded = self._tokenizer(
                    batch,
                    truncation=True,
                    max_length=256,
                    padding=True,
                    return_tensors="pt",
                ).to(self._device)
                probs = torch.sigmoid(self._model(**encoded).logits).cpu().tolist()
                all_probs.extend(probs)
        return decode_probabilities(all_probs, self.threshold)


@lru_cache
def get_topic_classifier() -> LocalTopicClassifier:
    return LocalTopicClassifier(get_settings())
