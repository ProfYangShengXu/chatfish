"""``impl/`` 的测试。

分两类，**刻意分开**：

1. **无外部依赖**：RRF 融合、BM25 封装、reranker 的装配校验 —— 这些不需要数据库也不需要模型。
2. **真连 PostgreSQL**：向量适配、``is_parent`` 列、外键 ON DELETE、幂等导入。
   ★ 这一类**必须真跑** —— 「mock 全绿、真库全炸」正是这类项目最容易踩的坑
     （``%s`` vs ``:name``、``vector`` 参数适配、缺列、FK 无级联，在 mock 里一个都看不见）。
   通过环境变量 ``MEMORY_BRIDGE_TEST_DSN`` 启用；没设就 skip，而不是假装通过。
"""

from __future__ import annotations

import os
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from memory_bridge.domain.models import (
    Document,
    Memory,
    MemoryType,
    Patch,
    Priority,
    SourceKind,
    Tier,
)
from memory_bridge.impl.hybrid import Bm25Index, HybridRetriever, rrf_fuse, tokenize
from memory_bridge.impl.pg_store import (
    SUPERSEDE_SIMILARITY,
    PgMemoryStore,
    _parse_vector,
    _split_sql,
    _v,
    libpq_to_sqlalchemy_url,
)

NOW = datetime(2026, 9, 21, 12, 0, 0)
DSN = os.environ.get("MEMORY_BRIDGE_TEST_DSN", "")

requires_db = pytest.mark.skipif(
    not DSN, reason="需要 MEMORY_BRIDGE_TEST_DSN（真连 PostgreSQL 才有意义）"
)


# ---------------------------------------------------------------------------
# 纯函数
# ---------------------------------------------------------------------------


class TestRrfFuse:
    def test_single_list_preserves_order(self) -> None:
        assert rrf_fuse([["a", "b", "c"]]) == ["a", "b", "c"]

    def test_document_in_both_lists_wins(self) -> None:
        fused = rrf_fuse([["a", "b"], ["b", "a"]])
        assert set(fused) == {"a", "b"}

    def test_agreement_beats_single_first_place(self) -> None:
        # x 在两路都排第 2；y 只在一路排第 1
        fused = rrf_fuse([["y", "x"], ["z", "x"]])
        assert fused[0] == "x"

    def test_empty_input(self) -> None:
        assert rrf_fuse([]) == []

    def test_rank_is_zero_based(self) -> None:
        """★ 钉住 0-based：换成 1-based 会让小 k 时的名次差别不可忽略。"""
        assert rrf_fuse([["a"]], k=60) == ["a"]
        # 两路排名相同 → 分数翻倍，但顺序不变
        assert rrf_fuse([["a", "b"], ["a", "b"]]) == ["a", "b"]

    def test_k_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            rrf_fuse([["a"]], k=0)

    def test_ties_broken_deterministically(self) -> None:
        first = rrf_fuse([["b", "a"], ["a", "b"]])
        second = rrf_fuse([["a", "b"], ["b", "a"]])
        assert first == second


class TestTokenize:
    def test_chinese_is_segmented(self) -> None:
        tokens = tokenize("用户不喜欢 ORM")
        assert "ORM" in tokens
        assert "不" not in tokens  # 停用词被过滤

    def test_punctuation_dropped(self) -> None:
        assert all(token.isalnum() for token in tokenize("你好，世界！"))

    def test_empty(self) -> None:
        assert tokenize("") == []


class TestBm25Index:
    def test_rebuild_and_search(self) -> None:
        index = Bm25Index()
        memories = [
            Memory(
                id="m1",
                owner="user:a",
                what="用户不喜欢 ORM 喜欢手写 SQL",
                type=MemoryType.PREFERENCE,
                when=NOW,
            ),
            Memory(
                id="m2",
                owner="user:a",
                what="部署机器在 WSL 里",
                type=MemoryType.FACT,
                when=NOW,
            ),
        ]
        index.rebuild(memories)
        hits = index.search("ORM SQL", 5)
        assert hits
        assert hits[0][0] == "m1"

    def test_owner_filter_prevents_cross_owner_leak(self) -> None:
        """★ BM25 的 ``search`` 没有 owner 参数 —— 必须在**排名之前**划子集。"""
        index = Bm25Index()
        index.rebuild(
            [
                Memory(
                    id="mine",
                    owner="user:a",
                    what="我喜欢 ORM",
                    type=MemoryType.PREFERENCE,
                    when=NOW,
                ),
                Memory(
                    id="theirs",
                    owner="user:b",
                    what="我喜欢 ORM",
                    type=MemoryType.PREFERENCE,
                    when=NOW,
                ),
            ]
        )
        hits = index.search("ORM", 5, owner="user:a")
        assert [h[0] for h in hits] == ["mine"]

    def test_allowed_ids_filter(self) -> None:
        index = Bm25Index()
        index.rebuild(
            [
                Memory(
                    id="m1",
                    owner="user:a",
                    what="我喜欢 ORM",
                    type=MemoryType.PREFERENCE,
                    when=NOW,
                ),
                Memory(
                    id="m2",
                    owner="user:a",
                    what="我喜欢 ORM 也喜欢 SQL",
                    type=MemoryType.PREFERENCE,
                    when=NOW,
                ),
            ]
        )
        hits = index.search("ORM", 5, allowed_ids={"m2"})
        assert [h[0] for h in hits] == ["m2"]

    def test_empty_index(self) -> None:
        index = Bm25Index()
        index.rebuild([])
        assert index.search("anything", 5) == []

    def test_search_before_rebuild(self) -> None:
        assert Bm25Index().search("x", 5) == []

    def test_zero_and_negative_k(self) -> None:
        index = Bm25Index()
        index.rebuild(
            [
                Memory(
                    id="m1",
                    owner="user:a",
                    what="内容",
                    type=MemoryType.FACT,
                    when=NOW,
                )
            ]
        )
        assert index.search("内容", 0) == []


# ---------------------------------------------------------------------------
# 向量 / DSN 适配（纯函数，不需要连库）
# ---------------------------------------------------------------------------


class TestVectorAdaptation:
    def test_v_formats_as_pgvector_literal(self) -> None:
        assert _v([1.0, 2.5]) == "[1,2.5]"

    def test_v_roundtrip(self) -> None:
        vector = [0.1, -0.25, 3.0]
        assert _parse_vector(_v(vector)) == pytest.approx(vector)

    def test_parse_vector_handles_string(self) -> None:
        assert _parse_vector("[1,2,3]") == [1.0, 2.0, 3.0]

    def test_parse_vector_handles_none(self) -> None:
        assert _parse_vector(None) is None

    def test_parse_vector_rejects_garbage(self) -> None:
        with pytest.raises(TypeError):
            _parse_vector(42)

    def test_compact_format_is_short(self) -> None:
        """512 维不该被格式化成 10000 字符。"""
        assert len(_v([0.123456789] * 512)) < 512 * 12


class TestDsnConversion:
    def test_libpq_to_sqlalchemy_url(self) -> None:
        url = libpq_to_sqlalchemy_url(
            "host=127.0.0.1 port=5432 dbname=memory_bridge "
            "user=memory_bridge password=secret"
        )
        assert url.startswith("postgresql+psycopg://")
        assert "memory_bridge:secret@" in url
        assert url.endswith("/memory_bridge")

    def test_special_characters_are_quoted(self) -> None:
        url = libpq_to_sqlalchemy_url(
            "host=127.0.0.1 dbname=db user=u password=p@ss/w%rd"
        )
        assert "p%40ss%2Fw%25rd" in url

    def test_invalid_dsn_raises(self) -> None:
        with pytest.raises(ValueError):
            libpq_to_sqlalchemy_url("this is not a conninfo")

    def test_empty_dsn_rejected_by_store(self) -> None:
        with pytest.raises(ValueError):
            PgMemoryStore(dsn="")

    def test_wrong_dim_rejected(self) -> None:
        with pytest.raises(ValueError):
            PgMemoryStore(dsn="host=127.0.0.1 dbname=x", embedding_dim=768)


class TestSplitSql:
    def test_splits_on_semicolons(self) -> None:
        assert _split_sql("SELECT 1; SELECT 2;") == ["SELECT 1", "SELECT 2"]

    def test_ignores_semicolons_in_comments(self) -> None:
        statements = _split_sql("-- a; b\nSELECT 1;")
        assert statements == ["SELECT 1"]


# ---------------------------------------------------------------------------
# 真连库
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def store() -> PgMemoryStore:
    if not DSN:
        pytest.skip("需要 MEMORY_BRIDGE_TEST_DSN")
    pg = PgMemoryStore(dsn=DSN, embedding_dim=512)
    schema = Path(__file__).resolve().parent.parent / "sql" / "001_schema.sql"
    pg.apply_schema(schema)
    yield pg
    pg.dispose()


@pytest.fixture
def owner() -> str:
    return f"test:{uuid.uuid4().hex[:8]}"


def _memory(
    *,
    owner: str,
    what: str,
    embedding: list[float] | None = None,
    mtype: MemoryType = MemoryType.FACT,
    scope: str = "global",
    when: datetime = NOW,
) -> Memory:
    return Memory(
        id=f"m_{uuid.uuid4().hex[:12]}",
        owner=owner,
        what=what,
        type=mtype,
        when=when,
        scope=scope,
        confidence=0.6,
        importance=0.5,
        embedding=embedding,
    )


def _unit_vector(dim: int, index: int) -> list[float]:
    vector = [0.0] * dim
    vector[index % dim] = 1.0
    return vector


@requires_db
class TestPgStoreRoundtrip:
    def test_schema_is_idempotent(self, store: PgMemoryStore) -> None:
        """DDL 必须能重复执行（验收时会反复跑）。"""
        schema = Path(__file__).resolve().parent.parent / "sql" / "001_schema.sql"
        store.apply_schema(schema)
        store.apply_schema(schema)
        assert store.ping()

    def test_save_and_read_back(self, store: PgMemoryStore, owner: str) -> None:
        memory = _memory(owner=owner, what="用户不喜欢 ORM")
        store.save_memory(memory)
        rows = store.active_memories(owner)
        assert len(rows) == 1
        assert rows[0].what == "用户不喜欢 ORM"
        assert rows[0].type is MemoryType.FACT
        assert rows[0].source_kind is SourceKind.USER_INPUT
        assert rows[0].tier is Tier.RETRIEVED
        assert rows[0].hit_count == 0
        assert rows[0].used_count == 0

    def test_datetime_is_naive_after_roundtrip(
        self, store: PgMemoryStore, owner: str
    ) -> None:
        """★ 读回来必须是 naive —— 否则 domain 层的减法会抛 TypeError。"""
        store.save_memory(_memory(owner=owner, what="时间戳测试"))
        row = store.active_memories(owner)[0]
        assert row.when.tzinfo is None

    def test_aware_datetime_is_normalized_on_write(
        self, store: PgMemoryStore, owner: str
    ) -> None:
        from datetime import timezone

        aware = datetime(2026, 9, 21, 12, 0, tzinfo=timezone.utc)
        store.save_memory(_memory(owner=owner, what="时区测试", when=aware))
        row = store.active_memories(owner)[0]
        assert row.when.tzinfo is None
        assert row.when.hour == 12

    def test_quotes_in_text_survive(self, store: PgMemoryStore, owner: str) -> None:
        payload = "用户说：'; DROP TABLE memories; --"
        store.save_memory(_memory(owner=owner, what=payload))
        assert store.active_memories(owner)[0].what == payload
        assert store.count_memories(owner)["total"] >= 1

    def test_unicode_and_emoji_survive(self, store: PgMemoryStore, owner: str) -> None:
        payload = "用户喜欢 🎯 精确匹配 · 中文 · Ω"
        store.save_memory(_memory(owner=owner, what=payload))
        assert store.active_memories(owner)[0].what == payload


@requires_db
class TestPgStoreSupersede:
    def test_similar_embedding_supersedes_atomically(
        self, store: PgMemoryStore, owner: str
    ) -> None:
        """★ 同一事务里：插入新行 + 旧行 superseded_by 指向新行。"""
        vector = _unit_vector(512, 0)
        old = _memory(owner=owner, what="用户喜欢用 pytest", embedding=vector)
        store.save_memory(old)

        new = _memory(
            owner=owner,
            what="用户喜欢用 pytest 做测试",
            embedding=[*vector[:511], 0.001],
        )
        store.save_memory(new)

        old_row = store.get_memory(old.id)
        new_row = store.get_memory(new.id)
        assert old_row is not None and new_row is not None
        assert old_row.superseded_by == new.id
        assert new_row.superseded_by is None
        # 旧条目不再出现在 active 集里
        assert [m.id for m in store.active_memories(owner)] == [new.id]

    def test_dissimilar_embedding_coexists(
        self, store: PgMemoryStore, owner: str
    ) -> None:
        first = _memory(owner=owner, what="喜欢 Rust", embedding=_unit_vector(512, 0))
        store.save_memory(first)
        second = _memory(owner=owner, what="部署在 WSL", embedding=_unit_vector(512, 300))
        store.save_memory(second)
        assert len(store.active_memories(owner)) == 2

    def test_different_owner_never_superseded(
        self, store: PgMemoryStore, owner: str
    ) -> None:
        vector = _unit_vector(512, 5)
        store.save_memory(_memory(owner=owner, what="我喜欢 ORM", embedding=vector))
        other = f"{owner}-other"
        store.save_memory(_memory(owner=other, what="我喜欢 ORM", embedding=vector))
        assert len(store.active_memories(owner)) == 1
        assert len(store.active_memories(other)) == 1

    def test_supersede_threshold_is_high(self) -> None:
        assert SUPERSEDE_SIMILARITY >= 0.8


@requires_db
class TestPgStoreVectorSearch:
    def test_vector_search_returns_nearest(
        self, store: PgMemoryStore, owner: str
    ) -> None:
        """★ 这一步验证 ``vector`` 参数适配 —— 传 list[float] 会报算子不存在。"""
        near = _unit_vector(512, 0)
        far = _unit_vector(512, 256)
        store.save_memory(_memory(owner=owner, what="最近的", embedding=near))
        store.save_memory(_memory(owner=owner, what="最远的", embedding=far))

        hits = store.vector_search(owner, near, 5)
        assert hits
        assert hits[0][1] == pytest.approx(0.0, abs=1e-6)

    def test_vector_search_isolated_by_owner(
        self, store: PgMemoryStore, owner: str
    ) -> None:
        vector = _unit_vector(512, 10)
        store.save_memory(_memory(owner=owner, what="我的", embedding=vector))
        store.save_memory(
            _memory(owner=f"{owner}-x", what="别人的", embedding=vector)
        )
        hits = store.vector_search(owner, vector, 10)
        assert len(hits) == 1

    def test_superseded_excluded_from_vector_search(
        self, store: PgMemoryStore, owner: str
    ) -> None:
        vector = _unit_vector(512, 20)
        old = _memory(owner=owner, what="旧事实", embedding=vector)
        store.save_memory(old)
        store.save_memory(
            _memory(owner=owner, what="新事实", embedding=[*vector[:511], 0.001])
        )
        hits = store.vector_search(owner, vector, 10)
        assert old.id not in [memory_id for memory_id, _ in hits]

    def test_empty_k(self, store: PgMemoryStore, owner: str) -> None:
        assert store.vector_search(owner, _unit_vector(512, 0), 0) == []


@requires_db
class TestPgStoreTouchAndTier:
    def test_touch_increments_hit_count(self, store: PgMemoryStore, owner: str) -> None:
        memory = _memory(owner=owner, what="被召回")
        store.save_memory(memory)
        store.touch([memory.id], NOW)
        store.touch([memory.id], NOW)
        row = store.active_memories(owner)[0]
        assert row.hit_count == 2
        assert row.last_access is not None

    def test_update_used_count_is_separate_signal(
        self, store: PgMemoryStore, owner: str
    ) -> None:
        """★ hit_count 与 used_count 必须是两个独立计数器。"""
        memory = _memory(owner=owner, what="双计数器")
        store.save_memory(memory)
        store.touch([memory.id], NOW)
        store.update_used_count([memory.id], NOW)
        store.update_used_count([memory.id], NOW)
        row = store.active_memories(owner)[0]
        assert row.hit_count == 1
        assert row.used_count == 2

    def test_update_tier(self, store: PgMemoryStore, owner: str) -> None:
        memory = _memory(owner=owner, what="升档目标")
        store.save_memory(memory)
        store.update_tier(memory.id, "resident")
        assert store.active_memories(owner)[0].tier is Tier.RESIDENT

    def test_update_tier_rejects_garbage(
        self, store: PgMemoryStore, owner: str
    ) -> None:
        memory = _memory(owner=owner, what="x")
        store.save_memory(memory)
        with pytest.raises(ValueError):
            store.update_tier(memory.id, "not-a-tier")

    def test_touch_empty_list_is_noop(self, store: PgMemoryStore) -> None:
        store.touch([], NOW)


@requires_db
class TestPgStoreArchiveAndPurge:
    def test_archive_hides_from_active_but_keeps_row(
        self, store: PgMemoryStore, owner: str
    ) -> None:
        memory = _memory(owner=owner, what="待归档")
        store.save_memory(memory)
        assert store.archive_memories([memory.id]) == 1
        assert store.active_memories(owner) == []
        assert store.archived_count(owner) == 1
        assert store.get_memory(memory.id) is not None  # 行还在

    def test_archive_is_idempotent(self, store: PgMemoryStore, owner: str) -> None:
        memory = _memory(owner=owner, what="归档两次")
        store.save_memory(memory)
        assert store.archive_memories([memory.id]) == 1
        assert store.archive_memories([memory.id]) == 0

    def test_purge_deletes_rows(self, store: PgMemoryStore, owner: str) -> None:
        memory = _memory(owner=owner, what="待删除")
        store.save_memory(memory)
        assert store.purge_memories([memory.id]) == 1
        assert store.get_memory(memory.id) is None

    def test_purge_new_row_of_supersede_pair_does_not_violate_fk(
        self, store: PgMemoryStore, owner: str
    ) -> None:
        """★ 旧行的 ``superseded_by`` 指着新行；删新行**不能**抛 ForeignKeyViolation。

        架构文档那份 DDL 是裸 ``REFERENCES``（无 ON DELETE），
        照抄它的话这条会直接炸 —— 而「删除权」是验收项 C 的一条。
        """
        vector = _unit_vector(512, 40)
        old = _memory(owner=owner, what="旧", embedding=vector)
        store.save_memory(old)
        new = _memory(owner=owner, what="新", embedding=[*vector[:511], 0.001])
        store.save_memory(new)

        assert store.get_memory(old.id) is not None
        assert store.get_memory(old.id).superseded_by == new.id  # type: ignore[union-attr]

        store.remove_memory(new.id)  # 必须不抛
        refreshed = store.get_memory(old.id)
        assert refreshed is not None
        assert refreshed.superseded_by is None  # ON DELETE SET NULL

    def test_remove_missing_memory_is_noop(
        self, store: PgMemoryStore, owner: str
    ) -> None:
        store.remove_memory("m_does_not_exist")


@requires_db
class TestPgStoreChunks:
    def _docs(self, source: str) -> list[Document]:
        parent = Document(
            id=f"c_{uuid.uuid4().hex[:10]}",
            collection="kb",
            source_doc=source,
            text="父块正文",
            acl=["public"],
            doc_version="v1",
            effective_date=date(2026, 9, 1),
            chunk_index=0,
            parent_id=None,
            is_parent=True,
            embedding_model="test",
        )
        child = Document(
            id=f"c_{uuid.uuid4().hex[:10]}",
            collection="kb",
            source_doc=source,
            text="子块正文",
            acl=["public"],
            doc_version="v1",
            effective_date=date(2026, 9, 1),
            chunk_index=1,
            parent_id=parent.id,
            is_parent=False,
            embedding_model="test",
            embedding=_unit_vector(512, 3),
        )
        return [parent, child]

    def test_save_and_count(self, store: PgMemoryStore) -> None:
        source = f"test-{uuid.uuid4().hex[:8]}.md"
        store.save_chunks(self._docs(source))
        stats = store.count_chunks(source)
        assert stats["total"] == 2
        assert stats["parents"] == 1
        assert stats["children"] == 1

    def test_reingest_is_idempotent(self, store: PgMemoryStore) -> None:
        """★ 同一文档导入两次不该翻倍（架构文档那份 DDL 会翻倍）。"""
        source = f"test-{uuid.uuid4().hex[:8]}.md"
        first = self._docs(source)
        store.save_chunks(first)
        before = store.count_chunks(source)["total"]

        second = self._docs(source)
        store.save_chunks(second)
        after = store.count_chunks(source)["total"]
        assert before == after == 2

    def test_is_parent_column_exists(self, store: PgMemoryStore) -> None:
        """★ 架构文档那份 DDL 缺 ``is_parent``，照抄会让首次 chunk_search 报错。"""
        source = f"test-{uuid.uuid4().hex[:8]}.md"
        store.save_chunks(self._docs(source))
        hits = store.chunk_search(
            collection="kb", acl=["public"], emb=_unit_vector(512, 3), k=10
        )
        assert hits  # 能查通就说明列存在

    def test_chunk_search_excludes_parents(self, store: PgMemoryStore) -> None:
        source = f"test-{uuid.uuid4().hex[:8]}.md"
        docs = self._docs(source)
        store.save_chunks(docs)
        hits = store.chunk_search(
            collection="kb", acl=["public"], emb=_unit_vector(512, 3), k=10
        )
        parent_ids = {d.id for d in docs if d.is_parent}
        assert not parent_ids & {chunk_id for chunk_id, _ in hits}

    def test_acl_filter_blocks_other_teams(self, store: PgMemoryStore) -> None:
        source = f"test-{uuid.uuid4().hex[:8]}.md"
        docs = self._docs(source)
        for doc in docs:
            doc.acl = ["team-a"]
        store.save_chunks(docs)
        assert (
            store.chunk_search(
                collection="kb", acl=["team-b"], emb=_unit_vector(512, 3), k=10
            )
            == []
        )

    def test_fetch_parents_deduplicates(self, store: PgMemoryStore) -> None:
        """★ 多个子块同属一个父块 → 返回列表里只出现一次，且**保序**。"""
        source = f"test-{uuid.uuid4().hex[:8]}.md"
        docs = self._docs(source)
        parent = docs[0]
        extra = Document(
            id=f"c_{uuid.uuid4().hex[:10]}",
            collection="kb",
            source_doc=source,
            text="第二个子块",
            acl=["public"],
            doc_version="v1",
            effective_date=date(2026, 9, 1),
            chunk_index=2,
            parent_id=parent.id,
            is_parent=False,
            embedding_model="test",
            embedding=_unit_vector(512, 4),
        )
        docs.append(extra)
        store.save_chunks(docs)

        child_ids = [docs[1].id, extra.id]
        parents = store.fetch_parents(child_ids)
        assert len(parents) == 1
        assert parents[0].id == parent.id

    def test_fetch_parents_preserves_order(self, store: PgMemoryStore) -> None:
        source = f"test-{uuid.uuid4().hex[:8]}.md"
        docs = self._docs(source)
        store.save_chunks(docs)
        only = store.fetch_parents([docs[1].id])
        assert [d.id for d in only] == [docs[0].id]

    def test_fetch_parents_empty(self, store: PgMemoryStore) -> None:
        assert store.fetch_parents([]) == []

    def test_effective_date_roundtrip(self, store: PgMemoryStore) -> None:
        source = f"test-{uuid.uuid4().hex[:8]}.md"
        store.save_chunks(self._docs(source))
        parents = store.list_chunks(source, parents_only=True)
        assert parents[0].effective_date == date(2026, 9, 1)


@requires_db
class TestPgStorePatches:
    def test_append_and_read_patches(self, store: PgMemoryStore, owner: str) -> None:
        patch = Patch(
            id=0, owner=owner, text="本轮新增", priority=Priority.URGENT, at=NOW
        )
        store.append_patch(patch)
        assert patch.id > 0  # 自增主键回填
        pending = store.pending_patches(owner)
        assert len(pending) == 1
        assert pending[0].text == "本轮新增"
        assert pending[0].priority is Priority.URGENT

    def test_clear_patches(self, store: PgMemoryStore, owner: str) -> None:
        store.append_patch(
            Patch(id=0, owner=owner, text="x", priority=Priority.NORMAL, at=NOW)
        )
        store.clear_patches(owner)
        assert store.pending_patches(owner) == []

    def test_patches_isolated_by_owner(self, store: PgMemoryStore, owner: str) -> None:
        store.append_patch(
            Patch(id=0, owner=owner, text="mine", priority=Priority.NORMAL, at=NOW)
        )
        assert store.pending_patches(f"{owner}-other") == []


@requires_db
class TestPgStoreCounts:
    def test_count_memories(self, store: PgMemoryStore, owner: str) -> None:
        for index in range(3):
            store.save_memory(_memory(owner=owner, what=f"计数 {index}"))
        stats = store.count_memories(owner)
        assert stats["total"] == 3
        assert stats["active"] == 3


@requires_db
class TestHybridRetrieverAgainstDb:
    def test_pool_is_respected(self, store: PgMemoryStore, owner: str) -> None:
        """★ 传了 pool 就只从里面挑 —— 否则「预算退回检索档」会静默失效。"""
        from memory_bridge.domain.models import HybridConfig
        from tests.test_service import FakeEmbedder

        # ★ 必须带 embedding 落库 —— 否则向量路返回空，
        #   这条用例就退化成「只要 BM25 命中就算过」，测不出 pool 的作用。
        base = FakeEmbedder(512).encode(["ORM"])[0]
        # ⚠️ 两条向量必须**足够不像**，否则写入时的 supersede 判定会把第一条
        #    标成已被取代（同 owner + 同 type + 同 scope + 相似度 >= 阈值）。
        variant = [*base[:511], 1.0]
        first = _memory(owner=owner, what="用户在写 ORM 相关的代码", embedding=base)
        second = _memory(
            owner=owner, what="用户也在写 ORM 相关的测试", embedding=variant
        )
        store.save_memory(first)
        store.save_memory(second)
        assert len(store.active_memories(owner)) == 2

        retriever = HybridRetriever(
            store, FakeEmbedder(512), Bm25Index(), HybridConfig()
        )
        only_first = [m for m in store.active_memories(owner) if m.id == first.id]
        hits = retriever.retrieve("ORM", owner, 10, pool=only_first)
        assert [m.id for m in hits] == [first.id]

        # 不传 pool 时两条都能被召回 —— 证明上面的差异来自 pool，而不是数据。
        # 用 base 当 query 向量（两条都带 embedding，至少第一条必然进向量路）
        hits_all = retriever.retrieve("ORM", owner, 10)
        assert hits_all

    def test_empty_pool_returns_empty(self, store: PgMemoryStore, owner: str) -> None:
        from memory_bridge.domain.models import HybridConfig
        from tests.test_service import FakeEmbedder

        retriever = HybridRetriever(
            store, FakeEmbedder(512), Bm25Index(), HybridConfig()
        )
        assert retriever.retrieve("ORM", owner, 10, pool=[]) == []

    def test_no_pool_falls_back_to_store(
        self, store: PgMemoryStore, owner: str
    ) -> None:
        from memory_bridge.domain.models import HybridConfig
        from tests.test_service import FakeEmbedder

        store.save_memory(_memory(owner=owner, what="用户喜欢手写 SQL 查询"))
        retriever = HybridRetriever(
            store, FakeEmbedder(512), Bm25Index(), HybridConfig()
        )
        assert retriever.retrieve("SQL", owner, 10)


@requires_db
class TestForgetOrderWithRealCounters:
    def test_never_hit_memory_survives_forget(
        self, store: PgMemoryStore, owner: str
    ) -> None:
        """★ 「used_count 恒为 0」不该导致 forget 杀掉**从未被召回**的条目。

        这条是真实风险的回归测试：如果 is_low_value 只看 used_count==0，
        那么每次容量治理都会优先杀掉最常被召回的记忆。
        """
        from memory_bridge.domain.models import ForgetConfig
        from memory_bridge.service.forget import ForgetPipeline

        fresh = _memory(owner=owner, what="刚写还没被召回")
        store.save_memory(fresh)
        pipeline = ForgetPipeline(
            store, ForgetConfig(max_active=1, utilization_threshold=0.5)
        )
        report = pipeline.run(owner=owner, now=NOW + timedelta(days=1))
        assert fresh.id not in report.archived
