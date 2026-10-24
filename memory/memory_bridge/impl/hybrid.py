"""L2 实现：``Retriever`` 的实现 —— BM25 + 向量 + RRF。

设计文档第八节。

不做什么：
  ❌ 不用分数加权（量纲不同，归一化也救不了）
  ❌ 不做 rerank（那是下一层）
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import math

import jieba
import numpy as np
from rank_bm25 import BM25Okapi

from ..domain.clock import to_naive_utc
from ..domain.models import Document, HybridConfig, Memory
from ..ports import Embedder, Retriever
from .pg_store import PgMemoryStore

_STOPWORDS = frozenset(
    {
        "的",
        "了",
        "是",
        "在",
        "我",
        "有",
        "和",
        "就",
        "不",
        "人",
        "都",
        "一",
        "一个",
        "上",
        "也",
        "很",
        "到",
        "说",
        "要",
        "去",
        "你",
        "会",
        "着",
        "没有",
        "看",
        "好",
        "自己",
        "这",
        "那",
        "它",
        "他",
        "她",
        "们",
        "把",
        "被",
        "让",
        "给",
        "对",
        "从",
        "以",
        "为",
        "与",
        "或",
        "而",
        "但",
        "所以",
        "因为",
        "如果",
        "这个",
        "那个",
        "什么",
        "怎么",
        "以及",
    }
)


def tokenize(text: str) -> list[str]:
    """jieba 分词 + 停用词过滤。

    ⚠️ 索引与查询**必须用同一个函数、同一份词典**，否则 term 对不上。
    ⚠️ jieba 的第一次分词要建前缀词典（官方说 1-3 秒）→ 在 ``Bm25Index.__init__``
       里显式 ``jieba.initialize()`` 预热，避免第一次召回卡住。
    """
    tokens: list[str] = []
    for token in jieba.cut(text):
        stripped = token.strip()
        if not stripped or stripped in _STOPWORDS:
            continue
        # 丢掉纯标点
        if all(not ch.isalnum() for ch in stripped):
            continue
        tokens.append(stripped)
    return tokens


class _SmoothedBM25(BM25Okapi):  # type: ignore[misc]  # rank_bm25 无 stub，基类是 Any
    """把语料规模当「大语料的一个样本」来算 IDF。

    为什么需要它（实测踩到的真问题）：
      ``BM25Okapi`` 的 IDF 是 ``log(N - n + 0.5) - log(n + 0.5)``，
      当某个 term 出现在不到一半文档里、而 **N 很小**时，
      原始 IDF 为负，rank-bm25 会把它截断成 ``epsilon × 平均 IDF``。
      记忆库在早期只有几十条时，几乎所有 term 都属于「稀有 term」——
      于是**最相关的文档得分恰好是 0**，BM25 这一路完全失效，
      而 RRF 融合又只看排名，结果就是向量路独自决定排序。

      把 N 视为 ``N + _PRIOR``（先验 50 篇文档）后 IDF 永远为正，
      大库上这个偏移可以忽略，小库上不再把相关文档打成 0 分。
    """

    _PRIOR = 50

    def _calc_idf(self, nd: dict[str, int]) -> None:
        """直接算 IDF，**不调用父类实现**。

        ⚠️ 不能先调 ``super()._calc_idf()`` 再覆盖：父类会拿原始 ``nd``
           做 ``log(corpus_size - freq + 0.5)``，N 很小时直接 ``math domain error``。
           这里完全接管计算。
        """
        self.idf = {}
        # 把语料看成「N + PRIOR 篇文档里的一个样本」
        corpus_size = len(self.doc_freqs) + self._PRIOR
        for term, freq in nd.items():
            self.idf[term] = math.log(
                (corpus_size - freq + 0.5) / (freq + 0.5) + 1.0
            )
        self.average_idf = (
            sum(self.idf.values()) / len(self.idf) if self.idf else 0.0
        )


class Bm25Index:
    """``rank-bm25`` 的封装。

    ⚠️ ``avgdl`` 是全局统计量：批量导入短文本会让所有旧条目分数集体下降
       （设计文档第十一节）。所以每次大批量写入后要 ``rebuild``。
    """

    def __init__(self) -> None:
        jieba.initialize()
        self._ids: list[str] = []
        self._bm25: BM25Okapi | None = None
        self._owner_of: dict[str, str] = {}

    def __len__(self) -> int:
        return len(self._ids)

    def rebuild(self, memories: list[Memory]) -> None:
        """用当前 active 记忆整体重建索引。

        ★ 同时建立 ``id → owner`` 映射：``rank_bm25`` 的 ``search(query, k)``
          **没有 owner 参数**，所以「隔离先于 ANN」在 BM25 分支上只能靠
          「先划子集再检索」来表达 —— 这个映射就是划子集用的。
        """
        self._ids = [m.id for m in memories]
        self._owner_of = {m.id: m.owner for m in memories}
        if not self._ids:
            self._bm25 = None
            return
        corpus = [tokenize(m.what) for m in memories]
        # 语料里可能出现空 token 列表（纯符号），BM25Okapi 会除零
        corpus = [tokens or ["\u3000"] for tokens in corpus]
        self._bm25 = _SmoothedBM25(corpus)

    def search(
        self,
        query: str,
        k: int,
        *,
        owner: str | None = None,
        allowed_ids: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        """返回 ``[(memory_id, score), ...]``。

        ★ ``owner`` 过滤在这里发生：**先划子集再排名**，
          而不是让别人的记忆先进排名再被扔掉
          （后者会让 RRF 的名次被污染）。
        """
        if self._bm25 is None or not self._ids or k <= 0:
            return []

        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)

        candidates: list[tuple[str, float]] = []
        for index, memory_id in enumerate(self._ids):
            if owner is not None and self._owner_of.get(memory_id) != owner:
                continue
            if allowed_ids is not None and memory_id not in allowed_ids:
                continue
            score = float(scores[index])
            if score <= 0:
                continue
            candidates.append((memory_id, score))

        # 按分数降序；分数相同用 id 兜底，保证**跨进程可复现**
        candidates.sort(key=lambda item: (-item[1], item[0]))
        return candidates[:k]


def rrf_fuse(rank_lists: list[list[str]], k: int = 60) -> list[str]:
    """Reciprocal Rank Fusion —— 纯函数，不引库。

    ``score = Σ 1 / (k + rank)``，**只用排名，不用分数**。

    ⚠️ ``rank`` 从 **0** 开始（原式如此）。pgvector 官方 SQL 示例里的 ``RANK()``
       从 1 开始，两者差一个常数；这里固定用 0-based，并在测试里钉住。
    """
    if k <= 0:
        raise ValueError(f"rrf_fuse: k 必须为正，收到 {k}")

    scores: dict[str, float] = {}
    for ranks in rank_lists:
        for rank, doc_id in enumerate(ranks):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    # 分数降序；同分用 id 兜底，保证确定性
    return sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))


class HybridRetriever(Retriever):
    """BM25 + 向量 + RRF。"""

    def __init__(
        self,
        store: PgMemoryStore,
        embedder: Embedder,
        bm25_index: Bm25Index,
        cfg: HybridConfig,
        *,
        now: datetime | None = None,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._bm25 = bm25_index
        self._cfg = cfg
        self._rebuilt_for: int = -1

    def ensure_index(self, owner: str) -> None:
        """索引与库同步。

        ⚠️ 这里用「条数变了就重建」做**廉价变更检测**。
           真正的触发点是「大批量写入之后」（规格明说 avgdl 是全局统计量），
           逐条写入就重建会让每次 remember 都付一次全量分词的代价。
        """
        memories = self._store.active_memories(owner)
        if len(memories) != self._rebuilt_for:
            self._bm25.rebuild(memories)
            self._rebuilt_for = len(memories)

    def rebuild(self, owner: str) -> int:
        memories = self._store.active_memories(owner)
        self._bm25.rebuild(memories)
        self._rebuilt_for = len(memories)
        return len(memories)

    def retrieve(
        self,
        query: str,
        owner: str,
        k: int,
        *,
        scope: str | None = None,
        pool: list[Memory] | None = None,
    ) -> list[Memory]:
        """① 前置隔离：先按 owner 划子集（★ 必须在 ANN 之前）
        ② BM25 与 向量跑
        ③ RRF 融合：``score = Σ 1/(60 + rank)``
        ④ 返回 top k
        """
        del scope  # owner 隔离在 SQL 与 Bm25Index.search 里做；scope 由 recall 层过滤
        if k <= 0 or not query.strip():
            return []

        # ★ pool 给了就**只**在里面挑：调用方已经做完可见性 + 分档 + 常驻预算退回。
        #   重新去库里查会让「退回检索档」的条目凭空消失。
        if pool is not None:
            by_id = {m.id: m for m in pool}
        else:
            self.ensure_index(owner)
            by_id = {m.id: m for m in self._store.active_memories(owner)}
        if not by_id:
            return []

        allowed = set(by_id)

        # ② 两路召回
        #    向量路：查询侧带 bge 检索指令前缀
        query_vector = self._embedder.encode([query])[0]
        vector_hits = self._store.vector_search(
            owner, query_vector, max(k, len(allowed)), allowed_ids=allowed
        )
        vector_ids = [memory_id for memory_id, _ in vector_hits if memory_id in allowed]

        bm25_hits = self._bm25.search(
            query, self._cfg.bm25_k, owner=owner, allowed_ids=allowed
        )
        bm25_ids = [memory_id for memory_id, _ in bm25_hits if memory_id in allowed]

        # ③ RRF 融合 —— 只用排名
        fused = rrf_fuse([vector_ids, bm25_ids], k=self._cfg.rrf_k)

        # ④ 回填
        out: list[Memory] = []
        for memory_id in fused[:k]:
            memory = by_id.get(memory_id)
            if memory is not None:
                out.append(memory)
        return out

    def score(self, query: str, candidates: list[Memory]) -> list[float]:
        """给候选打「融合分」—— 只在没有精排时用作相关性代理量。

        量纲说明：两路都命中 ≈ ``2/(k+rank)``，只命中一路 ≈ ``1/(k+rank)``。
        所以它衡量的是「多少条检索路径都认为它相关」，不是语义相似度。
        """
        by_id = {m.id: m for m in candidates}
        owner_of = {m.id: m.owner for m in candidates.values()} if False else {
            m.id: m.owner for m in candidates
        }
        owner = next(iter(owner_of.values()), "")
        query_vector = self._embedder.encode([query])[0]
        allowed = set(by_id)
        vector_hits = self._store.vector_search(
            owner, query_vector, max(len(allowed), 1), allowed_ids=allowed
        )
        bm25_hits = self._bm25.search(
            query, self._cfg.bm25_k, owner=owner, allowed_ids=allowed
        )
        scores: dict[str, float] = {}
        for rank, (memory_id, _) in enumerate(vector_hits):
            scores[memory_id] = scores.get(memory_id, 0.0) + 1.0 / (self._cfg.rrf_k + rank)
        for rank, (memory_id, _) in enumerate(bm25_hits):
            scores[memory_id] = scores.get(memory_id, 0.0) + 1.0 / (self._cfg.rrf_k + rank)
        return [scores.get(m.id, 0.0) for m in candidates]

    def retrieve_docs(
        self, query: str, collection: str, k: int, *, acl: list[str]
    ) -> list[Document]:
        """知识域检索：★ 父子两步 —— 子块检索、父块喂模型。"""
        if k <= 0 or not query.strip() or not acl:
            return []

        query_vector = self._embedder.encode([query])[0]
        child_hits = self._store.chunk_search(
            collection=collection, acl=acl, emb=query_vector, k=k
        )
        child_ids = [chunk_id for chunk_id, _ in child_hits]
        # ★ 按父块去重（在 pg_store.fetch_parents 里做，且**保序**）
        return self._store.fetch_parents(child_ids)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """给 ``EmbeddingAttributor`` 用的小工具。"""
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    denominator = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(va, vb) / denominator)


def normalize_now(now: datetime) -> datetime:  # pragma: no cover - 兼容用
    return to_naive_utc(now)


__all__ = [
    "Bm25Index",
    "HybridRetriever",
    "rrf_fuse",
    "tokenize",
    "cosine_similarity",
]
