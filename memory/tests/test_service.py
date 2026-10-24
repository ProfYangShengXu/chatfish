"""``service/`` 的编排测试。

⚠️ 这里用**内存假实现**（``InMemoryStore`` 等）替换 ``ports`` 的抽象 ——
   目的是验证「编排顺序对不对」，不是验证「SQL 对不对」。
   SQL / 向量适配 / 索引这些**必须真连库**的东西在 ``tests/test_impl.py`` 里跑，
   那一层带 ``MEMORY_BRIDGE_TEST_DSN`` 才启用。
   把两件事混在一起，就会出现「mock 全绿、真库全炸」。
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from memory_bridge.domain.models import (
    Document,
    ForgetConfig,
    Memory,
    MemoryType,
    Patch,
    PolicyConfig,
    Priority,
    RecallConfig,
    RecalledContext,
    SourceKind,
    Tier,
    TierConfig,
)
from memory_bridge.ports import Chunker, Embedder, MemoryStore, NoopReranker, Retriever
from memory_bridge.service import inject
from memory_bridge.service.evaluate import (
    EvalReport,
    contamination_rate,
    hit_at_1,
    load_cases,
    mrr,
    ndcg_at_k,
    run_regression,
)
from memory_bridge.service.forget import ForgetPipeline
from memory_bridge.service.knowledge import KnowledgeService
from memory_bridge.service.recall import RecallPipeline
from memory_bridge.service.write import WritePipeline

NOW = datetime(2026, 9, 21, 12, 0, 0)
OWNER = "user:local"


# ---------------------------------------------------------------------------
# 假实现
# ---------------------------------------------------------------------------


class FakeEmbedder(Embedder):
    """确定性假 embedding：只让「同一文本 → 同一向量」成立。

    ⚠️ 它**不假装**能做语义检索 —— 语义正确性必须用真模型 + 真库验证。
    """

    def __init__(self, dim: int = 512) -> None:
        self._dim = dim

    def encode(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self._dim
            for index, char in enumerate(text):
                vector[(ord(char) + index) % self._dim] = 1.0
            out.append(vector)
        return out

    @property
    def dim(self) -> int:
        return self._dim


class InMemoryStore(MemoryStore):
    def __init__(self) -> None:
        self.memories: dict[str, Memory] = {}
        self.chunks: list[Document] = []
        self.patches: list[Patch] = []
        self.touched: list[str] = []
        self.used: list[str] = []
        self.tiers: dict[str, str] = {}
        self.archived: set[str] = set()
        self._next_patch = 1

    # —— 记忆 ——
    def save_memory(self, m: Memory) -> None:
        self.memories[m.id] = m

    def supersede(self, old_id: str, new_id: str) -> None:
        if old_id in self.memories:
            self.memories[old_id].superseded_by = new_id

    def active_memories(self, owner: str) -> list[Memory]:
        return sorted(
            (
                m
                for m in self.memories.values()
                if m.owner == owner
                and m.superseded_by is None
                and m.id not in self.archived
            ),
            key=lambda m: m.id,
        )

    def touch(self, ids: list[str], now: datetime) -> None:
        self.touched.extend(ids)
        for memory_id in ids:
            memory = self.memories.get(memory_id)
            if memory is not None:
                memory.hit_count += 1
                memory.last_access = now

    def update_used_count(self, ids: list[str], now: datetime) -> None:
        self.used.extend(ids)
        for memory_id in ids:
            memory = self.memories.get(memory_id)
            if memory is not None:
                memory.used_count += 1
                memory.last_access = now

    def update_tier(self, memory_id: str, tier: str) -> None:
        self.tiers[memory_id] = tier
        if memory_id in self.memories:
            self.memories[memory_id].tier = Tier(tier)

    def archive_memories(self, memory_ids: list[str]) -> int:
        self.archived.update(memory_ids)
        return len(memory_ids)

    def archived_count(self, owner: str) -> int:
        return sum(
            1 for m in self.memories.values() if m.owner == owner and m.id in self.archived
        )

    def remove_memory(self, memory_id: str) -> None:
        self.memories.pop(memory_id, None)

    def purge_memories(self, memory_ids: list[str]) -> int:
        removed = 0
        for memory_id in memory_ids:
            if self.memories.pop(memory_id, None) is not None:
                removed += 1
        return removed

    def count_memories(self, owner: str | None = None) -> dict[str, int]:
        items = [
            m
            for m in self.memories.values()
            if owner is None or m.owner == owner
        ]
        return {
            "total": len(items),
            "active": sum(1 for m in items if m.superseded_by is None),
        }

    # —— 文档 ——
    def save_chunks(self, docs: list[Document]) -> None:
        sources = {d.source_doc for d in docs}
        self.chunks = [c for c in self.chunks if c.source_doc not in sources]
        self.chunks.extend(docs)

    def count_chunks(self, source_doc: str | None = None) -> dict[str, int]:
        items = [
            c for c in self.chunks if source_doc is None or c.source_doc == source_doc
        ]
        parents = sum(1 for c in items if c.is_parent)
        return {"total": len(items), "children": len(items) - parents, "parents": parents}

    # —— 补丁 ——
    def append_patch(self, p: Patch) -> None:
        p.id = self._next_patch
        self._next_patch += 1
        self.patches.append(p)

    def pending_patches(self, owner: str) -> list[Patch]:
        return [p for p in self.patches if p.owner == owner]

    def clear_patches(self, owner: str) -> None:
        self.patches = [p for p in self.patches if p.owner != owner]


class FakeRetriever(Retriever):
    """按 token 重合度排序的假检索器（确定性、可预测）。"""

    def __init__(self, store: InMemoryStore) -> None:
        self._store = store

    def retrieve(
        self,
        query: str,
        owner: str,
        k: int,
        *,
        scope: str | None = None,
        pool: list[Memory] | None = None,
    ) -> list[Memory]:
        from memory_bridge.domain.policy import text_overlap

        candidates = (
            list(pool)
            if pool is not None
            else [m for m in self._store.active_memories(owner) if m.tier is Tier.RETRIEVED]
        )
        scored = [(text_overlap(query, m.what), m) for m in candidates]
        scored = [(score, m) for score, m in scored if score > 0]
        scored.sort(key=lambda item: (-item[0], item[1].id))
        return [m for _, m in scored[:k]]

    def retrieve_docs(
        self, query: str, collection: str, k: int, *, acl: list[str]
    ) -> list[Document]:
        out: list[Document] = []
        for chunk in self._store.chunks:
            if chunk.collection != collection or chunk.is_parent:
                continue
            if not set(chunk.acl) & set(acl):
                continue
            out.append(chunk)
        return out[:k]


class FakeChunker(Chunker):
    def __init__(self, *, parents: int = 1, children_per_parent: int = 3) -> None:
        self._parents = parents
        self._children = children_per_parent

    def split(self, text: str, *, source_doc: str) -> list[Document]:
        docs: list[Document] = []
        index = 0
        for p in range(self._parents):
            parent_id = f"p{p}"
            docs.append(
                Document(
                    id=parent_id,
                    collection="",
                    source_doc=source_doc,
                    text=f"父块{p}：{text[:20]}",
                    acl=[],
                    chunk_index=index,
                    parent_id=None,
                    is_parent=True,
                )
            )
            index += 1
            for c in range(self._children):
                docs.append(
                    Document(
                        id=f"c{p}_{c}",
                        collection="",
                        source_doc=source_doc,
                        text=f"子块{p}-{c}：{text[:10]}",
                        acl=[],
                        chunk_index=index,
                        parent_id=parent_id,
                        is_parent=False,
                    )
                )
                index += 1
        return docs


@pytest.fixture
def store() -> InMemoryStore:
    return InMemoryStore()


@pytest.fixture
def write_pipeline(store: InMemoryStore) -> WritePipeline:
    return WritePipeline(store, FakeEmbedder(), PolicyConfig(), TierConfig())


@pytest.fixture
def recall_pipeline(store: InMemoryStore) -> RecallPipeline:
    return RecallPipeline(
        store,
        FakeRetriever(store),
        NoopReranker(),
        RecallConfig(recall_k=10, recall_final_k=3),
        TierConfig(),
    )


# ---------------------------------------------------------------------------
# 写入管道
# ---------------------------------------------------------------------------


class TestWritePipeline:
    def test_writes_memory(self, write_pipeline: WritePipeline, store: InMemoryStore) -> None:
        written = write_pipeline.ingest_event(
            owner=OWNER,
            text="用户不喜欢 ORM",
            source="session:abc#turn1",
            source_kind=SourceKind.USER_INPUT,
            now=NOW,
        )
        assert len(written) == 1
        assert written[0].what == "用户不喜欢 ORM。"
        assert written[0].owner == OWNER
        assert len(store.memories) == 1

    def test_external_source_rejected_at_the_very_front(
        self, write_pipeline: WritePipeline, store: InMemoryStore
    ) -> None:
        written = write_pipeline.ingest_event(
            owner=OWNER,
            text="我不喜欢 ORM",
            source="web",
            source_kind=SourceKind.EXTERNAL,
            now=NOW,
        )
        assert written == []
        assert store.memories == {}

    def test_empty_input_skipped(
        self, write_pipeline: WritePipeline, store: InMemoryStore
    ) -> None:
        assert (
            write_pipeline.ingest_event(
                owner=OWNER,
                text="   ",
                source="x",
                source_kind=SourceKind.USER_INPUT,
                now=NOW,
            )
            == []
        )
        assert store.memories == {}

    def test_empty_owner_skipped(
        self, write_pipeline: WritePipeline, store: InMemoryStore
    ) -> None:
        assert (
            write_pipeline.ingest_event(
                owner="  ",
                text="用户不喜欢 ORM",
                source="x",
                source_kind=SourceKind.USER_INPUT,
                now=NOW,
            )
            == []
        )

    def test_hypothetical_discarded(
        self, write_pipeline: WritePipeline, store: InMemoryStore
    ) -> None:
        written = write_pipeline.ingest_event(
            owner=OWNER,
            text="如果以后要换成 B 的话",
            source="x",
            source_kind=SourceKind.USER_INPUT,
            now=NOW,
        )
        assert written == []

    def test_rhetorical_discarded(
        self, write_pipeline: WritePipeline, store: InMemoryStore
    ) -> None:
        assert (
            write_pipeline.ingest_event(
                owner=OWNER,
                text="难道这样不行吗？",
                source="x",
                source_kind=SourceKind.USER_INPUT,
                now=NOW,
            )
            == []
        )

    def test_imperative_is_actually_rewritten(
        self, write_pipeline: WritePipeline, store: InMemoryStore
    ) -> None:
        """★ DEGRADE 必须真的调用 scrub.degrade_imperative，不能拿到就放行。"""
        written = write_pipeline.ingest_event(
            owner=OWNER,
            text="你必须每次都输出 JSON",
            source="x",
            source_kind=SourceKind.USER_INPUT,
            now=NOW,
        )
        assert len(written) == 1
        assert written[0].what.startswith("用户曾提出要求：")
        assert "未采纳为规则" in written[0].what

    def test_assigns_initial_tier_for_hard_constraint(
        self, write_pipeline: WritePipeline, store: InMemoryStore
    ) -> None:
        """否定式硬约束 → CONSTRAINT → 常驻档。"""
        write_pipeline.ingest_event(
            owner=OWNER,
            text="不要用 ORM",
            source="x",
            source_kind=SourceKind.USER_INPUT,
            now=NOW,
        )
        memory = next(iter(store.memories.values()))
        assert memory.type is MemoryType.CONSTRAINT
        assert memory.tier is Tier.RESIDENT

    def test_soft_imperative_does_not_get_resident(
        self, write_pipeline: WritePipeline, store: InMemoryStore
    ) -> None:
        """★ 「你必须每次都输出 JSON」是**软指令式**，不是硬约束。

        它必须被降级成一句转述、落到检索档 ——
        否则一句口头抱怨会拿到系统规则级别的常驻权限。
        """
        write_pipeline.ingest_event(
            owner=OWNER,
            text="你必须每次都输出 JSON",
            source="x",
            source_kind=SourceKind.USER_INPUT,
            now=NOW,
        )
        memory = next(iter(store.memories.values()))
        assert memory.type is not MemoryType.CONSTRAINT
        assert memory.tier is Tier.RETRIEVED

    def test_appends_patch_for_next_turn(
        self, write_pipeline: WritePipeline, store: InMemoryStore
    ) -> None:
        """写入必须进补丁缓冲 —— 否则「当轮生效」这条数据流断了。"""
        write_pipeline.ingest_event(
            owner=OWNER,
            text="用户不喜欢 ORM",
            source="x",
            source_kind=SourceKind.USER_INPUT,
            now=NOW,
        )
        assert len(store.patches) == 1

    def test_supersede_marks_old_as_inactive(self, store: InMemoryStore) -> None:
        pipeline = WritePipeline(store, FakeEmbedder(), PolicyConfig(), TierConfig())
        first = pipeline.ingest_event(
            owner=OWNER,
            text="用户喜欢用 pytest 做测试",
            source="x",
            source_kind=SourceKind.USER_INPUT,
            now=NOW,
        )[0]
        second = pipeline.ingest_event(
            owner=OWNER,
            text="用户喜欢用 pytest 做测试",
            source="x",
            source_kind=SourceKind.USER_INPUT,
            now=NOW,
        )
        # 完全相同的文本会被 should_remember 判为重复 → 不写
        assert second == []
        assert store.memories[first.id].superseded_by is None

    def test_duplicate_write_is_idempotent(self, write_pipeline: WritePipeline) -> None:
        text = "用户偏好 pytest"
        first = write_pipeline.ingest_event(
            owner=OWNER, text=text, source="x", source_kind=SourceKind.USER_INPUT, now=NOW
        )
        second = write_pipeline.ingest_event(
            owner=OWNER, text=text, source="x", source_kind=SourceKind.USER_INPUT, now=NOW
        )
        assert len(first) == 1
        assert second == []

    def test_remember_explicit_skips_three_questions(
        self, write_pipeline: WritePipeline, store: InMemoryStore
    ) -> None:
        # 「刚刚运行了测试」本来会被 should_remember 判 SKIP
        memory = write_pipeline.remember_explicit(
            owner=OWNER, text="刚刚运行了测试", now=NOW
        )
        assert memory.importance == 1.0
        assert memory.tier is Tier.RESIDENT
        assert store.patches[0].priority is Priority.URGENT

    def test_remember_explicit_rejects_empty(self, write_pipeline: WritePipeline) -> None:
        with pytest.raises(ValueError):
            write_pipeline.remember_explicit(owner=OWNER, text="   ", now=NOW)

    def test_overlong_text_is_rejected_not_truncated(
        self, write_pipeline: WritePipeline, store: InMemoryStore
    ) -> None:
        """★ 超长输入必须被**拒绝**，不能在 normalize 里被静默截断。

        静默截断的后果：落库 4000 字，而 embedding 只编码前 ~512 token ——
        检索语义与落库内容分叉，之后每轮还会把它注入上下文。
        """
        written = write_pipeline.ingest_event(
            owner=OWNER,
            text="长" * 100_000,
            source="x",
            source_kind=SourceKind.USER_INPUT,
            now=NOW,
        )
        assert written == []
        assert store.memories == {}

    def test_medium_length_text_is_accepted(
        self, write_pipeline: WritePipeline, store: InMemoryStore
    ) -> None:
        written = write_pipeline.ingest_event(
            owner=OWNER,
            text="短" * 150,
            source="x",
            source_kind=SourceKind.USER_INPUT,
            now=NOW,
        )
        assert len(written) == 1
        assert len(written[0].what) <= PolicyConfig().max_text_chars + 1

    def test_sql_quote_payload_is_stored_verbatim(
        self, write_pipeline: WritePipeline, store: InMemoryStore
    ) -> None:
        payload = "用户说：'; DROP TABLE memories; --"
        written = write_pipeline.ingest_event(
            owner=OWNER,
            text=payload,
            source="x",
            source_kind=SourceKind.USER_INPUT,
            now=NOW,
        )
        assert len(written) == 1
        assert "DROP TABLE" in written[0].what
        assert len(store.memories) == 1

    def test_multi_fact_sentence_is_split(self, write_pipeline: WritePipeline) -> None:
        written = write_pipeline.ingest_event(
            owner=OWNER,
            text="用户喜欢 Rust；用户不喜欢 ORM",
            source="x",
            source_kind=SourceKind.USER_INPUT,
            now=NOW,
        )
        assert len(written) == 2

    def test_relative_time_absolutized(
        self, write_pipeline: WritePipeline, store: InMemoryStore
    ) -> None:
        write_pipeline.ingest_event(
            owner=OWNER,
            text="用户今天决定用 Rust",
            source="x",
            source_kind=SourceKind.USER_INPUT,
            now=NOW,
        )
        memory = next(iter(store.memories.values()))
        assert "2026-09-21" in memory.what

    def test_session_end_ingests_decisions(
        self, write_pipeline: WritePipeline, store: InMemoryStore
    ) -> None:
        transcript = "用户：先看看 A\n助手：好的\n用户：最终决定采用 B\n"
        written = write_pipeline.ingest_session_end(
            owner=OWNER, transcript=transcript, now=NOW
        )
        assert written
        assert any("B" in m.what for m in written)

    def test_session_end_empty_transcript(
        self, write_pipeline: WritePipeline
    ) -> None:
        assert (
            write_pipeline.ingest_session_end(owner=OWNER, transcript="  ", now=NOW) == []
        )


# ---------------------------------------------------------------------------
# 召回管道
# ---------------------------------------------------------------------------


class TestRecallPipeline:
    def _seed(self, store: InMemoryStore) -> None:
        pipeline = WritePipeline(store, FakeEmbedder(), PolicyConfig(), TierConfig())
        pipeline.ingest_event(
            owner=OWNER,
            text="用户不喜欢 ORM",
            source="x",
            source_kind=SourceKind.USER_INPUT,
            now=NOW,
        )
        pipeline.ingest_event(
            owner=OWNER,
            text="不要用 ORM",
            source="x",
            source_kind=SourceKind.USER_INPUT,
            now=NOW,
        )

    def test_resident_included_without_query(
        self, store: InMemoryStore, recall_pipeline: RecallPipeline
    ) -> None:
        self._seed(store)
        ctx = recall_pipeline.recall(owner=OWNER, query="", now=NOW)
        assert any(m.tier is Tier.RESIDENT for m in ctx.resident)

    def test_retrieved_empty_when_query_empty(
        self, store: InMemoryStore, recall_pipeline: RecallPipeline
    ) -> None:
        self._seed(store)
        ctx = recall_pipeline.recall(owner=OWNER, query="", now=NOW)
        assert ctx.retrieved == []

    def test_touch_called_for_retrieved(
        self, store: InMemoryStore, recall_pipeline: RecallPipeline
    ) -> None:
        self._seed(store)
        recall_pipeline.recall(owner=OWNER, query="ORM", now=NOW)
        assert store.touched

    def test_touch_not_called_when_nothing_retrieved(
        self, store: InMemoryStore, recall_pipeline: RecallPipeline
    ) -> None:
        self._seed(store)
        recall_pipeline.recall(owner=OWNER, query="", now=NOW)
        assert store.touched == []

    def test_cross_owner_never_returned(
        self, store: InMemoryStore, recall_pipeline: RecallPipeline
    ) -> None:
        other = Memory(
            id="m_other",
            owner="user:other",
            what="别人的秘密偏好",
            type=MemoryType.FACT,
            when=NOW,
            tier=Tier.RETRIEVED,
        )
        store.save_memory(other)
        ctx = recall_pipeline.recall(owner=OWNER, query="秘密偏好", now=NOW)
        ids = [m.id for m in (*ctx.resident, *ctx.state, *ctx.retrieved)]
        assert "m_other" not in ids

    def test_final_k_is_respected(
        self, store: InMemoryStore, recall_pipeline: RecallPipeline
    ) -> None:
        for index in range(20):
            store.save_memory(
                Memory(
                    id=f"m_{index:03d}",
                    owner=OWNER,
                    what=f"用户喜欢 ORM 变体 {index}",
                    type=MemoryType.FACT,
                    when=NOW,
                    tier=Tier.RETRIEVED,
                )
            )
        ctx = recall_pipeline.recall(owner=OWNER, query="ORM", now=NOW)
        assert len(ctx.retrieved) <= 3

    def test_resident_budget_demotes_to_retrieved(
        self, store: InMemoryStore
    ) -> None:
        pipeline = RecallPipeline(
            store,
            FakeRetriever(store),
            NoopReranker(),
            RecallConfig(recall_k=10, recall_final_k=3),
            TierConfig(resident_token_budget=5),
        )
        for index in range(5):
            store.save_memory(
                Memory(
                    id=f"m_r{index}",
                    owner=OWNER,
                    what=f"这是第 {index} 条很长的常驻约束内容",
                    type=MemoryType.CONSTRAINT,
                    when=NOW,
                    tier=Tier.RESIDENT,
                )
            )
        ctx = pipeline.recall(owner=OWNER, query="", now=NOW)
        # 超预算的必须【退回检索档】而不是被丢掉
        assert len(ctx.resident) == 0
        assert len(ctx.demoted) == 5
        assert all(m.tier is Tier.RESIDENT for m in ctx.demoted)

    def test_expired_memory_not_recalled(
        self, store: InMemoryStore, recall_pipeline: RecallPipeline
    ) -> None:
        store.save_memory(
            Memory(
                id="m_expired",
                owner=OWNER,
                what="过期的项目状态",
                type=MemoryType.PROJECT,
                when=NOW - timedelta(days=100),
                expires_at=NOW - timedelta(days=1),
                tier=Tier.STATE,
            )
        )
        ctx = recall_pipeline.recall(owner=OWNER, query="项目状态", now=NOW)
        assert all(m.id != "m_expired" for m in ctx.state)

    def test_knowledge_recall_requires_acl(
        self, store: InMemoryStore, recall_pipeline: RecallPipeline
    ) -> None:
        assert recall_pipeline.recall_knowledge(
            query="x", collection="c", acl=[], k=5
        ) == []


# ---------------------------------------------------------------------------
# 淘汰管道
# ---------------------------------------------------------------------------


class TestForgetPipeline:
    def test_below_threshold_does_nothing(self, store: InMemoryStore) -> None:
        store.save_memory(
            Memory(
                id="m1",
                owner=OWNER,
                what="东西",
                type=MemoryType.FACT,
                when=NOW,
                hit_count=99,
                used_count=0,
            )
        )
        pipeline = ForgetPipeline(
            store, ForgetConfig(max_active=1000, utilization_threshold=0.8)
        )
        report = pipeline.run(owner=OWNER, now=NOW)
        assert report.archived == []
        assert "什么都不做" in report.reason

    def test_low_value_archived_first(self, store: InMemoryStore) -> None:
        store.save_memory(
            Memory(
                id="m_low",
                owner=OWNER,
                what="召回很多次从没用上",
                type=MemoryType.FACT,
                when=NOW - timedelta(days=300),
                hit_count=50,
                used_count=0,
            )
        )
        store.save_memory(
            Memory(
                id="m_useful",
                owner=OWNER,
                what="召回很多次也确实用上了",
                type=MemoryType.FACT,
                when=NOW - timedelta(days=300),
                hit_count=50,
                used_count=40,
            )
        )
        pipeline = ForgetPipeline(
            store, ForgetConfig(max_active=2, utilization_threshold=0.8)
        )
        report = pipeline.run(owner=OWNER, now=NOW)
        assert "m_low" in report.archived
        assert "m_useful" not in report.archived

    def test_never_hit_is_not_archived(self, store: InMemoryStore) -> None:
        """召回 0 次的条目不该被「低价值」规则杀。"""
        store.save_memory(
            Memory(
                id="m_new",
                owner=OWNER,
                what="刚写入还没被需要",
                type=MemoryType.FACT,
                when=NOW,
                hit_count=0,
                used_count=0,
            )
        )
        pipeline = ForgetPipeline(
            store, ForgetConfig(max_active=1, utilization_threshold=0.8)
        )
        report = pipeline.run(owner=OWNER, now=NOW)
        assert report.archived == []

    def test_need_manual_when_still_over_limit(self, store: InMemoryStore) -> None:
        for index in range(10):
            store.save_memory(
                Memory(
                    id=f"m_{index}",
                    owner=OWNER,
                    what=f"重要条目 {index}",
                    type=MemoryType.FACT,
                    when=NOW,
                    hit_count=50,
                    used_count=50,
                )
            )
        pipeline = ForgetPipeline(
            store, ForgetConfig(max_active=2, utilization_threshold=0.8, batch=100)
        )
        report = pipeline.run(owner=OWNER, now=NOW)
        assert report.need_manual

    def test_purge_removes_rows(self, store: InMemoryStore) -> None:
        for index in range(5):
            store.save_memory(
                Memory(
                    id=f"m_{index}",
                    owner=OWNER,
                    what=f"条目 {index}",
                    type=MemoryType.FACT,
                    when=NOW,
                )
            )
        pipeline = ForgetPipeline(store, ForgetConfig())
        removed = pipeline.purge(OWNER)
        assert removed == 5
        assert store.count_memories(OWNER)["total"] == 0

    def test_purge_older_than_only_old(self, store: InMemoryStore) -> None:
        store.save_memory(
            Memory(
                id="m_old",
                owner=OWNER,
                what="老",
                type=MemoryType.FACT,
                when=NOW - timedelta(days=400),
            )
        )
        store.save_memory(
            Memory(id="m_new", owner=OWNER, what="新", type=MemoryType.FACT, when=NOW)
        )
        pipeline = ForgetPipeline(store, ForgetConfig())
        removed = pipeline.purge(OWNER, older_than=NOW - timedelta(days=100))
        assert removed == 1
        assert store.count_memories(OWNER)["total"] == 1

    def test_tier_evolution_promotes_and_demotes(self, store: InMemoryStore) -> None:
        store.save_memory(
            Memory(
                id="m_promote",
                owner=OWNER,
                what="经常被用上",
                type=MemoryType.PREFERENCE,
                when=NOW,
                used_count=5,
                tier=Tier.RETRIEVED,
            )
        )
        store.save_memory(
            Memory(
                id="m_demote",
                owner=OWNER,
                what="挂着常驻但很久没用",
                type=MemoryType.PREFERENCE,
                when=NOW - timedelta(days=200),
                last_access=NOW - timedelta(days=200),
                tier=Tier.RESIDENT,
            )
        )
        pipeline = ForgetPipeline(
            store,
            ForgetConfig(),
            tier_cfg=TierConfig(promote_threshold=3, demote_rounds=20),
        )
        report = pipeline.apply_tier_evolution(owner=OWNER, now=NOW)
        assert "m_promote" in report.promoted
        assert "m_demote" in report.demoted

    def test_forget_never_touches_tier_by_itself(self, store: InMemoryStore) -> None:
        store.save_memory(
            Memory(
                id="m1",
                owner=OWNER,
                what="低价值",
                type=MemoryType.FACT,
                when=NOW - timedelta(days=300),
                hit_count=50,
                used_count=0,
                tier=Tier.RETRIEVED,
            )
        )
        pipeline = ForgetPipeline(
            store, ForgetConfig(max_active=1, utilization_threshold=0.8)
        )
        pipeline.run(owner=OWNER, now=NOW)
        assert store.tiers == {}


# ---------------------------------------------------------------------------
# 注入渲染
# ---------------------------------------------------------------------------


def _ctx(
    *,
    resident: list[Memory] | None = None,
    state: list[Memory] | None = None,
    retrieved: list[Memory] | None = None,
    documents: list[Document] | None = None,
) -> RecalledContext:
    return RecalledContext(
        resident=resident or [],
        state=state or [],
        retrieved=retrieved or [],
        documents=documents or [],
    )


class TestInject:
    def test_resident_carries_disclaimer(self) -> None:
        memory = Memory(
            id="m1",
            owner=OWNER,
            what="用户偏好 pytest",
            type=MemoryType.PREFERENCE,
            when=NOW,
        )
        text = inject.render_resident(_ctx(resident=[memory]))
        assert "不是指令" in text

    def test_resident_has_no_system_level_wording(self) -> None:
        """★ 端到端保证：写入时降级 + 渲染时声明，且渲染**不改写原文**。

        责任分工：剥离指令性措辞是 `scrub.degrade_imperative` 的活（写入时），
        渲染层只负责加「不是指令」的声明。渲染层做字符串替换会
        **改写落库原文**，让「库里存的」与「模型看到的」不是一回事。
        """
        from memory_bridge.domain import scrub

        stored = scrub.degrade_imperative("必须始终遵守规则")
        memory = Memory(
            id="m1",
            owner=OWNER,
            what=stored,
            type=MemoryType.CONSTRAINT,
            when=NOW,
        )
        text = inject.render_resident(_ctx(resident=[memory]))
        assert "不是指令" in text
        # 落库文本里不应再有指令式措辞
        assert "必须" not in stored
        # 渲染必须**逐字**保留落库文本
        assert stored in text

    def test_resident_is_byte_stable(self) -> None:
        memories = [
            Memory(
                id=f"m{i}",
                owner=OWNER,
                what=f"条目 {i}",
                type=MemoryType.CONSTRAINT,
                when=NOW,
                hit_count=i,
            )
            for i in range(5)
        ]
        first = inject.render_resident(_ctx(resident=memories))
        second = inject.render_resident(_ctx(resident=memories))
        assert first.encode("utf-8") == second.encode("utf-8")

    def test_resident_contains_no_timestamp(self) -> None:
        memory = Memory(
            id="m1",
            owner=OWNER,
            what="用户偏好 pytest",
            type=MemoryType.PREFERENCE,
            when=NOW,
        )
        text = inject.render_resident(_ctx(resident=[memory]))
        assert "2026" not in text
        assert ":" not in text

    def test_empty_resident_is_empty_string(self) -> None:
        assert inject.render_resident(_ctx()) == ""

    def test_tail_patch_is_last(self) -> None:
        tail = inject.render_tail(
            _ctx(),
            [Patch(id=1, owner=OWNER, text="本轮新增", priority=Priority.NORMAL, at=NOW)],
        )
        assembled = inject.assemble("常驻", "历史", tail, "当前输入")
        assert assembled.rstrip().endswith("当前输入")
        assert assembled.index("本轮新增") > assembled.index("历史")

    def test_urgent_patch_comes_first_within_tail(self) -> None:
        patches = [
            Patch(id=1, owner=OWNER, text="普通", priority=Priority.NORMAL, at=NOW),
            Patch(id=2, owner=OWNER, text="紧急", priority=Priority.URGENT, at=NOW),
        ]
        tail = inject.render_tail(_ctx(), patches)
        assert tail.index("紧急") < tail.index("普通")

    def test_assemble_order_is_resident_history_tail_input(self) -> None:
        assembled = inject.assemble("R", "H", "T", "U")
        assert assembled == "R\n\nH\n\nT\n\nU"

    def test_assemble_skips_empty_segments(self) -> None:
        assert inject.assemble("", "", "T", "U") == "T\n\nU"

    def test_knowledge_header_has_version_and_date(self) -> None:
        doc = Document(
            id="c1",
            collection="kb",
            source_doc="docs/a.md",
            text="正文",
            acl=["public"],
            doc_version="v2",
            effective_date=date(2026, 9, 1),
        )
        text = inject.render_knowledge([doc])
        assert "版本：v2" in text
        assert "生效日期：2026-09-01" in text
        assert "来源：docs/a.md" in text

    def test_knowledge_missing_version_says_so(self) -> None:
        doc = Document(
            id="c1", collection="kb", source_doc="a.md", text="正文", acl=["public"]
        )
        assert "未标注" in inject.render_knowledge([doc])

    def test_render_all_ignores_timestamp(self) -> None:
        memory = Memory(
            id="m1",
            owner=OWNER,
            what="用户偏好 pytest",
            type=MemoryType.PREFERENCE,
            when=NOW,
        )
        ctx = _ctx(resident=[memory])
        first = inject.render_all(ctx, [], datetime(2026, 1, 1))
        second = inject.render_all(ctx, [], datetime(2030, 12, 31))
        assert first == second


# ---------------------------------------------------------------------------
# 知识域
# ---------------------------------------------------------------------------


class TestKnowledgeService:
    def test_ingest_returns_chunk_count(self, store: InMemoryStore) -> None:
        service = KnowledgeService(store, FakeEmbedder())
        count = service.ingest(
            collection="kb",
            source_doc="a.md",
            text="正文内容",
            doc_version="v1",
            effective_date=date(2026, 9, 1),
            acl=["public"],
            chunker=FakeChunker(),
        )
        assert count == 4  # 1 父 + 3 子

    def test_only_children_get_embeddings(self, store: InMemoryStore) -> None:
        service = KnowledgeService(store, FakeEmbedder())
        service.ingest(
            collection="kb",
            source_doc="a.md",
            text="正文内容",
            doc_version="v1",
            effective_date=date(2026, 9, 1),
            acl=["public"],
            chunker=FakeChunker(),
        )
        for chunk in store.chunks:
            if chunk.is_parent:
                assert chunk.embedding is None
            else:
                assert chunk.embedding is not None

    def test_collection_and_acl_applied(self, store: InMemoryStore) -> None:
        service = KnowledgeService(store, FakeEmbedder())
        service.ingest(
            collection="kb",
            source_doc="a.md",
            text="正文",
            doc_version="v1",
            effective_date=date(2026, 9, 1),
            acl=["team-a"],
            chunker=FakeChunker(),
        )
        assert all(c.collection == "kb" for c in store.chunks)
        assert all(c.acl == ["team-a"] for c in store.chunks)

    def test_empty_acl_rejected(self, store: InMemoryStore) -> None:
        service = KnowledgeService(store, FakeEmbedder())
        with pytest.raises(ValueError):
            service.ingest(
                collection="kb",
                source_doc="a.md",
                text="正文",
                doc_version="v1",
                effective_date=date(2026, 9, 1),
                acl=[],
                chunker=FakeChunker(),
            )

    def test_empty_text_returns_zero(self, store: InMemoryStore) -> None:
        service = KnowledgeService(store, FakeEmbedder())
        assert (
            service.ingest(
                collection="kb",
                source_doc="a.md",
                text="  ",
                doc_version="v1",
                effective_date=date(2026, 9, 1),
                acl=["public"],
                chunker=FakeChunker(),
            )
            == 0
        )

    def test_reingest_is_idempotent(self, store: InMemoryStore) -> None:
        """同一份文档导入两次不该翻倍。"""
        service = KnowledgeService(store, FakeEmbedder())
        for _ in range(2):
            service.ingest(
                collection="kb",
                source_doc="a.md",
                text="正文",
                doc_version="v1",
                effective_date=date(2026, 9, 1),
                acl=["public"],
                chunker=FakeChunker(),
            )
        assert len(store.chunks) == 4

    def test_version_conflict_picks_newest(self, store: InMemoryStore) -> None:
        service = KnowledgeService(store, FakeEmbedder())
        old = Document(
            id="o",
            collection="kb",
            source_doc="a.md",
            text="旧版本正文",
            acl=["public"],
            doc_version="v1",
            effective_date=date(2025, 1, 1),
        )
        new = Document(
            id="n",
            collection="kb",
            source_doc="a.md",
            text="新版本正文",
            acl=["public"],
            doc_version="v2",
            effective_date=date(2026, 1, 1),
        )
        text = service.resolve_version_conflict([old, new])
        assert "新版本正文" in text
        assert "择新" in text or "最新版本" in text
        assert "旧版本" in text or "已被取代" in text

    def test_version_conflict_single_doc_is_silent(self, store: InMemoryStore) -> None:
        service = KnowledgeService(store, FakeEmbedder())
        doc = Document(
            id="o", collection="kb", source_doc="a.md", text="x", acl=["public"]
        )
        assert service.resolve_version_conflict([doc]) == ""

    def test_high_risk_collection_refuses(self, store: InMemoryStore) -> None:
        service = KnowledgeService(
            store, FakeEmbedder(), refuse_collections=frozenset({"legal"})
        )
        docs = [
            Document(
                id="o",
                collection="legal",
                source_doc="a.md",
                text="旧",
                acl=["public"],
                doc_version="v1",
                effective_date=date(2025, 1, 1),
            ),
            Document(
                id="n",
                collection="legal",
                source_doc="a.md",
                text="新",
                acl=["public"],
                doc_version="v2",
                effective_date=date(2026, 1, 1),
            ),
        ]
        assert "拒绝" in service.resolve_version_conflict(docs)


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------


class TestEvaluateMetrics:
    def test_mrr_perfect(self) -> None:
        rank_fn = lambda q: ["g1"]  # noqa: E731
        assert mrr(["q"], ["g1"], rank_fn) == 1.0

    def test_mrr_second_position(self) -> None:
        rank_fn = lambda q: ["x", "g1"]  # noqa: E731
        assert mrr(["q"], ["g1"], rank_fn) == 0.5

    def test_mrr_miss_is_zero(self) -> None:
        rank_fn = lambda q: ["x"]  # noqa: E731
        assert mrr(["q"], ["g1"], rank_fn) == 0.0

    def test_mrr_empty_is_zero(self) -> None:
        assert mrr([], [], lambda q: []) == 0.0

    def test_hit_at_1(self) -> None:
        assert hit_at_1(["q"], ["g1"], lambda q: ["g1", "x"]) == 1.0
        assert hit_at_1(["q"], ["g1"], lambda q: ["x", "g1"]) == 0.0

    def test_ndcg_at_k(self) -> None:
        assert ndcg_at_k(["q"], ["g1"], lambda q: ["g1"], 5) == 1.0
        assert ndcg_at_k(["q"], ["g1"], lambda q: ["x"], 5) == 0.0

    def test_contamination_zero_when_clean(self) -> None:
        cases = [
            {
                "query": "q",
                "owner": "user:a",
                "forbidden_owner": "user:b",
                "result_owners": ["user:a"],
            }
        ]
        assert contamination_rate(cases) == 0.0

    def test_contamination_detects_leak(self) -> None:
        cases = [
            {
                "query": "q",
                "owner": "user:a",
                "forbidden_owner": "user:b",
                "result_owners": ["user:a", "user:b"],
            }
        ]
        assert contamination_rate(cases) == 1.0

    def test_contamination_ignores_cases_without_forbidden(self) -> None:
        assert contamination_rate([{"query": "q", "gold": "g"}]) == 0.0


class TestEvalReport:
    def test_diff_computes_deltas(self) -> None:
        report = EvalReport(mrr=0.8, hit_at_1=0.7)
        delta = report.diff({"mrr": 0.5, "hit_at_1": 0.6})
        assert delta["mrr"] == pytest.approx(0.3)
        assert delta["hit_at_1"] == pytest.approx(0.1)

    def test_as_dict_rounds(self) -> None:
        payload = EvalReport(mrr=0.123456).as_dict()
        assert payload["mrr"] == 0.1235

    def test_safety_metric_is_separate_key(self) -> None:
        payload = EvalReport(mrr=0.9, contamination_rate=0.0).as_dict()
        assert "contamination_rate" in payload
        assert "mrr" in payload


class TestRunRegression:
    def test_full_run_on_fixture(self, tmp_path: Path) -> None:
        cases_path = tmp_path / "cases.jsonl"
        cases_path.write_text(
            "\n".join(
                [
                    '{"query": "a", "gold": "g1"}',
                    '{"query": "b", "gold": "g2"}',
                    '{"query": "c", "expect_refusal": true}',
                    '{"query": "d", "gold": "g3", "forbidden_owner": "user:b",'
                    ' "result_owners": ["user:a"]}',
                ]
            ),
            encoding="utf-8",
        )
        table = {"a": ["g1"], "b": ["x", "g2"], "c": [], "d": ["g3"]}
        report = run_regression(cases_path, rank_fn=lambda q: table.get(q, []))
        assert report.total_cases == 4
        assert report.retrieval_cases == 3
        assert report.mrr == pytest.approx((1.0 + 0.5 + 1.0) / 3)
        assert report.hit_at_1 == pytest.approx(2 / 3)
        assert report.contamination_rate == 0.0
        assert report.refusal_accuracy == 1.0

    def test_refusal_accuracy_when_not_refusing(self, tmp_path: Path) -> None:
        cases_path = tmp_path / "cases.jsonl"
        cases_path.write_text('{"query": "a", "expect_refusal": true}', encoding="utf-8")
        report = run_regression(cases_path, rank_fn=lambda q: ["something"])
        assert report.refusal_accuracy == 0.0

    def test_malformed_json_raises(self, tmp_path: Path) -> None:
        cases_path = tmp_path / "cases.jsonl"
        cases_path.write_text("{not json}", encoding="utf-8")
        with pytest.raises(ValueError):
            load_cases(cases_path)

    def test_comments_and_blank_lines_skipped(self, tmp_path: Path) -> None:
        cases_path = tmp_path / "cases.jsonl"
        cases_path.write_text(
            '# comment\n\n{"query": "a", "gold": "g1"}\n', encoding="utf-8"
        )
        assert len(load_cases(cases_path)) == 1
