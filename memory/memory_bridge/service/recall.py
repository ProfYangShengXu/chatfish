"""L2 服务：召回管道。

职责：给定 query，产出分好档的上下文（设计文档第八节）。

不做什么：
  ❌ 不渲染 prompt（归 ``inject.py``）
  ❌ 不跨域混排（记忆和知识**分别限流**，见设计文档第九节）
"""

from __future__ import annotations

from datetime import datetime

from ..domain import policy, scoring
from ..domain.clock import to_naive_utc
from ..domain.models import (
    Document,
    Memory,
    RecallConfig,
    RecalledContext,
    Tier,
    TierConfig,
)
from ..ports import MemoryStore, Reranker, Retriever


class RecallPipeline:
    def __init__(
        self,
        store: MemoryStore,
        retriever: Retriever,
        reranker: Reranker,
        cfg: RecallConfig,
        tier_cfg: TierConfig | None = None,
    ) -> None:
        self._store = store
        self._retriever = retriever
        self._reranker = reranker
        self._cfg = cfg
        self._tier_cfg = tier_cfg or TierConfig()

    def recall(
        self,
        *,
        owner: str,
        query: str,
        scope: str | None = None,
        cwd: str | None = None,
        now: datetime,
    ) -> RecalledContext:
        """完整召回流程。"""
        current = to_naive_utc(now)

        # ⓪ 取候选
        memories = self._store.active_memories(owner)

        # ① 可见性过滤 —— ★ 必须先过这关：跨 owner 是安全边界，不能靠排序
        visible = [
            m
            for m in memories
            if policy.visible_to(m, owner=owner, scope=scope, cwd=cwd, now=current)
        ]

        # ② 分档 —— ★ 按 ``m.tier``，不是按 ``type``（档位是自适应的）
        resident = [m for m in visible if m.tier is Tier.RESIDENT]
        state = [m for m in visible if m.tier is Tier.STATE]
        retrieved_pool = [m for m in visible if m.tier is Tier.RETRIEVED]

        # ③ 常驻档预算（token，不是条数）；超预算的【退回检索档】
        resident, demoted = policy.rank_for_resident(
            resident, self._tier_cfg, _count_tokens
        )
        retrieved_pool = [*retrieved_pool, *demoted]

        # ④ 检索档：混合检索 → 时间衰减加权 → 精排
        retrieved = self._retrieve_tier(
            owner=owner, query=query, scope=scope, pool=retrieved_pool, now=current
        )

        # ⑤ 反馈闭环：记 hit_count
        #    ⚠️ used_count 不在这里加 —— 它由 session.mark_used() 负责。
        hit_ids = [m.id for m in retrieved]
        if hit_ids:
            self._store.touch(hit_ids, current)

        return RecalledContext(
            resident=resident,
            state=state,
            retrieved=retrieved,
            documents=[],
            demoted=demoted,
        )

    def _retrieve_tier(
        self,
        *,
        owner: str,
        query: str,
        scope: str | None,
        pool: list[Memory],
        now: datetime,
    ) -> list[Memory]:
        if not query.strip():
            # 没有 query 就没有「检索档」可言（不是「返回全部」）
            return []

        candidates = self._retriever.retrieve(
            query, owner, self._cfg.recall_k, scope=scope, pool=pool
        )
        if not candidates:
            return []

        # ★ 隔离再兜一次：retriever 的实现（含 BM25 分支）不能把别人的记忆带进来。
        #   这一层过滤是纵深防御 —— 真正的隔离在 SQL 的 WHERE 里。
        pool_ids = {m.id for m in pool}
        candidates = [m for m in candidates if m.id in pool_ids]

        # ④a 时间衰减加权（廉价重排，先做以缩减精排候选）
        #     按 type 分档衰减；乘一个 [0.05, 1.0] 的因子，保序但能重排
        weighted = sorted(
            candidates,
            key=lambda m: scoring.decay_factor(m, now, scoring.half_life_for(m.type)),
            reverse=True,
        )

        # ④b 精排（延迟大头，候选数受 recall_max_candidates 硬上限约束）
        limited = weighted[: self._cfg.rerank_max_candidates]
        ranked = self._reranker.rerank(query, limited, self._cfg.recall_final_k)
        top = list(ranked[: self._cfg.recall_final_k])

        # ④c ★ 相关性下限 —— 这是「拒答」唯一可能的实现位置。
        #     没有它，ANN 的 top-k 永远返回 k 条（余弦相似度对任何输入都有值），
        #     「库里根本没有」与「库里有一条勉强相关」在结果上完全一样。
        #     返回**空**是有意义的信号：上层据此拒答，而不是硬凑几条塞进上下文。
        return self._apply_relevance_floor(query, top)

    def _apply_relevance_floor(self, query: str, candidates: list[Memory]) -> list[Memory]:
        """按 ``min_relevance`` 裁掉不相关的候选。

        ★ 分数来自 ``reranker.score()``（**不是** retriever 的融合分）。
          ``NoopReranker`` 一律返回 1.0，也就是「不设下限」——
          这是诚实的：没有精排就没有相关性判据，硬设阈值只会制造噪声。
          所以「拒答」能力**依赖 ``rerank_enabled: true``**，
          这一点已写入汇报的待确认项。
        """
        floor = self._cfg.min_relevance
        if floor <= 0 or not candidates:
            return candidates
        scores = self._reranker.score(query, candidates)
        return [
            memory
            for memory, score in zip(candidates, scores, strict=True)
            if score >= floor
        ]

    def recall_knowledge(
        self,
        *,
        query: str,
        collection: str,
        acl: list[str],
        k: int,
    ) -> list[Document]:
        """知识域召回（多对多语料，**独立限流，不跟记忆混排**）。

        ① 前置过滤：按 ``collection`` + ``acl`` 划子集 —— 在 ANN 之前（SQL 里）
        ② 只查【子块】：``WHERE is_parent = false``
        ③ 命中 → 取 ``parent_id`` → 回填【父块】
        ④ ★ 按父块去重
        ⑤ 返回父块列表

        ⚠️ 规格的签名是 ``recall_knowledge(..., conn)`` —— 裸露一个数据库连接。
           这里**故意不照抄**：``service/`` 不认识 psycopg，也就不能持有连接；
           连接的所有权在 ``impl/pg_store.py``。签名改为由 ``Retriever`` 端口承载。
           已列入汇报的待确认项。
        """
        if not query.strip() or not acl:
            return []
        return self._retriever.retrieve_docs(query, collection, k, acl=acl)


def _count_tokens(text: str) -> int:
    """常驻档预算的 token 计数。

    ⚠️ 这里用「CJK 字符 ≈ 1 token，其它按 4 字符 ≈ 1 token」的**估算**，
      不引入 tokenizer 依赖（规格第〇节：不引入新依赖）。
      保守偏高 —— 宁可少塞常驻档，也不要撑爆前缀。
    """
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - cjk
    return cjk + max(1, other // 4)
