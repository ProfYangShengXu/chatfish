"""L2 实现：``Reranker`` 的实现 —— cross-encoder。

设计文档第八节。

不做什么：
  ❌ 不用于召回（必须成对，无法预计算）
  ❌ 不做批量缓存（候选每轮都变）
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sentence_transformers import CrossEncoder

from ..domain.models import Memory
from ..ports import Embedder, Reranker, UsageAttributor
from .hybrid import cosine_similarity


class CrossEncoderReranker(Reranker):
    """``BAAI/bge-reranker-base`` 的封装。

    ⚠️ CPU 上是延迟大头（占 60-80%）→ 候选数受延迟预算约束。
    """

    def __init__(
        self,
        model_dir: Path,
        max_candidates: int = 50,
        *,
        max_length: int = 512,
        batch_size: int = 16,
    ) -> None:
        """``max_candidates`` 是【硬上限】—— 超过就截断，
        因为 cross-encoder 是 O(n) 次前向（设计文档第五节）。

        ⚠️ ``batch_size`` 默认取 16（而不是官方实现的 128）：
           本机是 16GB / 无独显的 CPU，128 会让峰值内存和单次延迟都很难看。
           已被列入汇报的待确认项（实测延迟见汇报 F 节）。
        """
        if not model_dir.exists():
            raise FileNotFoundError(
                f"找不到 reranker 模型目录：{model_dir}\n"
                "请先跑 `python scripts/setup_models.py` 下载模型。"
            )
        if max_candidates <= 0:
            raise ValueError("CrossEncoderReranker: max_candidates 必须为正")
        self._model_dir = model_dir
        self._max_candidates = max_candidates
        self._batch_size = batch_size
        self._model: Any = CrossEncoder(
            str(model_dir),
            max_length=max_length,
            device="cpu",
            local_files_only=True,
        )

    def score(self, query: str, candidates: list[Memory]) -> list[float]:
        """cross-encoder 的原始 logit 分（sigmoid 之前）。

        ⚠️ 不做归一化：同一模型的分数量纲是一致的，
           阈值应当按**实测分布**来定（见汇报 F 节的分数分布）。
        """
        if not candidates:
            return []
        pairs = [(query, memory.what) for memory in candidates]
        raw = self._model.predict(
            pairs, batch_size=self._batch_size, show_progress_bar=False
        )
        return [float(s) for s in raw]

    def rerank(self, query: str, candidates: list[Memory], k: int) -> list[Memory]:
        """对 ``candidates`` 打分并返回前 ``k`` 条。

        ⚠️ 候选数超过 ``max_candidates`` 时**先截断**再打分 ——
           截断前的顺序是上游（RRF + 时间衰减）给的，已经是有意义的序。
        """
        if k <= 0 or not candidates:
            return []

        limited = candidates[: self._max_candidates]
        scores = self.score(query, limited)

        ranked = sorted(
            zip(limited, scores, strict=True),
            key=lambda item: (-item[1], item[0].id),
        )
        return [memory for memory, _ in ranked[:k]]


class LexicalAttributor(UsageAttributor):
    """退化路径：用**字符串重合**判定「哪些记忆被用上了」。

    ⚠️ 规格明确写：「不要用 ①（模型归因）之外的启发式当主路径 ——
       会引入噪声，污染升档判据」。所以：
         - 它**不是**默认实现（见 ``cli.py`` 的装配）；
         - 它是唯一可用的实现，因为规格第〇节的依赖清单里**没有任何 LLM 客户端**
           （openai / anthropic 都不在）。这条死结已列入汇报的待确认项。
         - 阈值调得**很高**（默认 0.8），宁可漏报也不误报：
           漏报只让升档慢一点，误报会让噪声条目升进常驻档。
    """

    def __init__(self, threshold: float = 0.8) -> None:
        self._threshold = threshold

    def attribute(self, *, memories: list[Memory], answer: str) -> list[str]:
        from ..domain.policy import text_overlap

        if not answer.strip():
            return []
        used: list[str] = []
        for memory in memories:
            if text_overlap(memory.what, answer) >= self._threshold:
                used.append(memory.id)
        return used


class EmbeddingAttributor(UsageAttributor):
    """用 embedding 余弦判定「被用上」。

    比 ``LexicalAttributor`` 强一点（能认换一种说法），
    但仍然不是规格要求的「模型归因」，所以同样**不是默认实现**。
    """

    def __init__(self, embedder: Embedder, threshold: float = 0.75) -> None:
        self._embedder = embedder
        self._threshold = threshold

    def attribute(self, *, memories: list[Memory], answer: str) -> list[str]:
        if not answer.strip() or not memories:
            return []
        vectors = self._embedder.encode([answer, *[m.what for m in memories]])
        answer_vector, memory_vectors = vectors[0], vectors[1:]
        used: list[str] = []
        for memory, vector in zip(memories, memory_vectors, strict=True):
            if cosine_similarity(answer_vector, vector) >= self._threshold:
                used.append(memory.id)
        return used


class NoopAttributor(UsageAttributor):
    """什么都不归因（``used_count`` 保持 0）。

    ⚠️ **这是默认值**。理由：让 ``used_count`` 停在 0 是**诚实的**，
       而用启发式填它会让 ``should_promote`` 基于噪声升档。
       代价是 ``scoring.is_low_value`` 的 ``used_count == 0`` 条件恒真 ——
       所以 ``is_low_value`` 额外要求 ``hit_count >= 10``，
       否则「淘汰」会退化成「优先杀最常被召回的记忆」。
    """

    def attribute(self, *, memories: list[Memory], answer: str) -> list[str]:
        del memories, answer
        return []


__all__ = [
    "CrossEncoderReranker",
    "LexicalAttributor",
    "EmbeddingAttributor",
    "NoopAttributor",
]
