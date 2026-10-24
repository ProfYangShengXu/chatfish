"""L2 实现：``MemoryStore`` 的唯一实现（PostgreSQL + pgvector）。

不做什么：
  ❌ 不在这个文件里写业务判断
  ❌ 不做 BM25（那在 ``hybrid.py``）
  ❌ 不缓存（缓存归上层）

──────────── 几个必须解释的实现决定 ────────────

1. **向量参数适配**：规格里的 ``ORDER BY embedding <=> %s`` 直接传 ``list[float]``
   在 psycopg 里会被适配成 ``float[]``，PG 报「operator does not exist: vector <=> double precision[]」。
   所以这里把向量文本化成 ``'[1,2,3]'`` 再显式 ``::vector`` 转型。
   读回来同理：``vector`` 以字符串形式返回，必须手工 parse。

2. **supersede 原子性**：``save_memory(m)`` 内部就完成了
   「插入新行 + 把旧行的 superseded_by 指向新行」，并且**在同一个事务里**。
   调用方**不需要**再单独调 ``supersede()`` —— 那个方法保留是为了满足端口签名
   （并且是幂等的），但主路径不依赖它，这样中途崩溃不会留下「新旧并存」。
   旧行的选择：同 owner、同 type、同 scope、``superseded_by IS NULL``、
   与 ``m.embedding`` 余弦相似度 >= ``SUPERSEDE_SIMILARITY`` 的最近一条。

3. **pgvector 0.6.0 的过滤召回塌陷**：官方 README 明确「近似索引的过滤发生在索引扫描之后」，
   默认 ``hnsw.ef_search = 40``。所以：
     - 每次 ANN 查询前 ``SET LOCAL hnsw.ef_search = ...``（本地事务级，不污染其它连接）
     - SQL 层用「内层放大 k、外层再 LIMIT k」的两段式，把被过滤掉的名额补回来
   0.6.0 没有 ``hnsw.iterative_scan``（那是 0.8.0+），所以只能这么做。

4. **时间**：列是 ``TIMESTAMP WITHOUT TIME ZONE``，落库前一律 ``to_naive_utc()``；
   读回来是 naive，按 UTC 解释。见 ``domain/clock.py``。
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any, Sequence

from sqlalchemy import (
    ARRAY,
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    Integer,
    MetaData,
    Table,
    Text,
    create_engine,
    delete,
    func,
    insert,
    select,
    text,
    update,
)
from sqlalchemy.engine import Engine, RowMapping

from ..domain.clock import to_naive_utc
from ..domain.models import (
    Document,
    Memory,
    MemoryType,
    Patch,
    Priority,
    SourceKind,
    Tier,
)
from ..ports import MemoryStore

# 余弦相似度阈值：只有「说的基本是同一件事」才自动覆盖旧条目。
# 太低会误杀（把「喜欢 Python」覆盖掉「喜欢 Postgres」），太高会漏（同一事实换种说法就并存）。
SUPERSEDE_SIMILARITY = 0.90

# filtered ANN 的 ef_search。默认 40 在过滤命中率 10% 时平均只回 4 行；
# 512 维、万级行数下这个值的额外延迟可以忽略。
ANN_EF_SEARCH = 200

_metadata = MetaData()

# ⚠️ 这里的列类型必须与 sql/001_schema.sql 一致 ——
#    它们不生成 DDL（DDL 只在那一个文件里），只用于让 SQLAlchemy 正确适配参数：
#    Text 换成 DateTime/ARRAY 之后，datetime 与 list 才会被绑定成正确的 PG 类型。
memories = Table(
    "memories",
    _metadata,
    Column("id", Text, primary_key=True),
    Column("owner", Text, nullable=False),
    Column("what", Text, nullable=False),
    Column("type", Text, nullable=False),
    Column("scope", Text, nullable=False),
    Column("source", Text),
    Column("source_kind", Text, nullable=False),
    Column("confidence", Float, nullable=False),
    Column("importance", Float, nullable=False),
    Column("when", DateTime, nullable=False),
    Column("expires_at", DateTime),
    Column("hit_count", Integer, nullable=False),
    Column("used_count", Integer, nullable=False),
    Column("tier", Text, nullable=False),
    Column("last_access", DateTime),
    Column("superseded_by", Text),
    Column("archived_at", DateTime),
)

chunks = Table(
    "chunks",
    _metadata,
    Column("id", Text, primary_key=True),
    Column("collection", Text, nullable=False),
    Column("source_doc", Text, nullable=False),
    Column("doc_version", Text),
    Column("effective_date", Date),
    Column("section", Text),
    Column("chunk_index", Integer, nullable=False),
    Column("text", Text, nullable=False),
    Column("acl", ARRAY(Text), nullable=False),
    Column("parent_id", Text),
    Column("is_parent", Boolean, nullable=False),
    Column("embedding_model", Text, nullable=False),
)

patches = Table(
    "patches",
    _metadata,
    Column("id", Integer, primary_key=True),
    Column("owner", Text, nullable=False),
    Column("text", Text, nullable=False),
    Column("priority", Text, nullable=False),
    Column("at", DateTime, nullable=False),
)


def _v(value: Sequence[float]) -> str:
    """``list[float]`` → pgvector 字面量。"""
    return "[" + ",".join(f"{x:.7g}" for x in value) + "]"


def _parse_vector(raw: object) -> list[float] | None:
    """pgvector 读回来是字符串（``'[0.1,0.2]'``）或已是 list。"""
    if raw is None:
        return None
    if isinstance(raw, list):
        return [float(x) for x in raw]
    if isinstance(raw, str):
        return [float(x) for x in raw.strip().strip("[]").split(",") if x]
    raise TypeError(f"无法解析 embedding：{type(raw)}")


def _dt(value: datetime | None) -> datetime | None:
    return None if value is None else to_naive_utc(value)


class PgMemoryStore(MemoryStore):
    """``MemoryStore`` 的唯一实现。

    ⚠️ 连接池：SQLAlchemy 默认 ``pool_size=5, max_overflow=10``（=15 个连接），
       而 FastAPI 的同步端点跑在 anyio 线程池（默认 40 线程）上。
       所以这里显式把池放大到 ``pool_size=10, max_overflow=20``（=30），
       并设 ``pool_timeout=5``：**宁可快速失败，也不要让 30 秒的连接等待
       把 rerank 的延迟预算吃光**。已被列入汇报的待确认项。
    """

    def __init__(
        self,
        dsn: str,
        embedding_dim: int = 512,
        *,
        pool_size: int = 10,
        max_overflow: int = 20,
        pool_timeout: float = 5.0,
    ) -> None:
        if not dsn:
            raise ValueError("PgMemoryStore: dsn 为空")
        if embedding_dim != 512:
            raise ValueError(
                f"PgMemoryStore: 本实现用 vector(512)，收到 embedding_dim={embedding_dim}"
            )
        self._dsn = dsn
        self._embedding_dim = embedding_dim
        self._engine: Engine = create_engine(
            libpq_to_sqlalchemy_url(dsn),
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_timeout=pool_timeout,
            pool_pre_ping=True,
            pool_recycle=1800,
            future=True,
        )

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    @property
    def engine(self) -> Engine:
        return self._engine

    def dispose(self) -> None:
        self._engine.dispose()

    def apply_schema(self, schema_path: Path) -> None:
        """执行 ``sql/001_schema.sql``（唯一一份 DDL）。幂等。"""
        ddl = schema_path.read_text(encoding="utf-8")
        statements = _split_sql(ddl)
        with self._engine.begin() as conn:
            for statement in statements:
                conn.execute(text(statement))

    def ping(self) -> bool:
        try:
            with self._engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        except Exception:  # noqa: BLE001 - 探活不该把异常抛给调用方
            return False
        return True

    def extend_embedding_dim(self, dim: int) -> None:  # pragma: no cover - 迁移钩子
        raise NotImplementedError("维度变更需要迁移，本实现固定 512")

    # ------------------------------------------------------------------
    # 记忆
    # ------------------------------------------------------------------
    def save_memory(self, m: Memory) -> None:
        """落库一条记忆。

        ★ 原子性在这里：插入新行 + 更新旧行 ``superseded_by`` 同事务完成。
          旧行靠**向量近邻**在 SQL 里选出来（不依赖调用方先做文本比对），
          这样即使调用方漏判，也不会留下两条矛盾的 active 记忆。
        """
        now = to_naive_utc(m.when)
        with self._engine.begin() as conn:
            conn.execute(
                insert(memories).values(
                    id=m.id,
                    owner=m.owner,
                    what=m.what,
                    type=m.type.value,
                    scope=m.scope,
                    source=m.source,
                    source_kind=m.source_kind.value,
                    confidence=float(m.confidence),
                    importance=float(m.importance),
                    when=now,
                    expires_at=_dt(m.expires_at),
                    hit_count=m.hit_count,
                    used_count=m.used_count,
                    tier=m.tier.value,
                    last_access=_dt(m.last_access),
                    superseded_by=None,
                    archived_at=None,
                )
            )

            # ⚠️ embedding 必须**单独 UPDATE** 写入。
            #    `insert()` 的 values 里放不了 `vector` 字面量（SQLAlchemy 不知道怎么
            #    绑定它），只能插入后再 CAST 赋值。
            #    漏掉这一步的后果是：**写入不报错、行也在，但 embedding 永远是 NULL** ——
            #    于是 supersede 判定找不到旧条目、vector_search 永远返回空，
            #    而所有不连库的单测全绿（实测踩过这个坑）。
            if m.embedding is not None:
                conn.execute(
                    text(
                        "UPDATE memories SET embedding = CAST(:vec AS vector) "
                        "WHERE id = :id"
                    ),
                    {"vec": _v(m.embedding), "id": m.id},
                )
                old_id = self._find_supersede_target(conn, m)
                if old_id is not None:
                    self._supersede_within(conn, old_id, m.id)

    def _find_supersede_target(self, conn: Any, m: Memory) -> str | None:
        """在 SQL 里找「应该被这条新记忆取代」的旧条目。

        条件：同 owner + 同 type + 同 scope + 仍 active + 余弦相似度 >= 阈值。
        取最相似的一条（LIMIT 1）。
        """
        assert m.embedding is not None
        vector = _v(m.embedding)
        rows = conn.execute(
            text(
                """
                SELECT id, 1 - (embedding <=> CAST(:vec AS vector)) AS sim
                  FROM memories
                 WHERE owner = :owner
                   AND type = :type
                   AND scope = :scope
                   AND superseded_by IS NULL
                   AND archived_at IS NULL
                   AND id <> :new_id
                   AND embedding IS NOT NULL
                 ORDER BY embedding <=> CAST(:vec AS vector)
                 LIMIT 1
                """
            ),
            {
                "vec": vector,
                "owner": m.owner,
                "type": m.type.value,
                "scope": m.scope,
                "new_id": m.id,
            },
        ).first()
        if rows is None:
            return None
        similarity = float(rows[1])
        if similarity < SUPERSEDE_SIMILARITY:
            return None
        return str(rows[0])

    def _supersede_within(self, conn: Any, old_id: str, new_id: str) -> None:
        conn.execute(
            update(memories)
            .where(memories.c.id == old_id)
            .values(superseded_by=new_id)
        )

    def supersede(self, old_id: str, new_id: str) -> None:
        """独立的 supersede（幂等）。

        ⚠️ 主路径**不调用它** —— ``save_memory`` 已经同事务完成。
           保留它是为了满足端口签名，也让「显式纠正历史」有入口。
        """
        with self._engine.begin() as conn:
            self._supersede_within(conn, old_id, new_id)

    def active_memories(self, owner: str) -> list[Memory]:
        """``WHERE owner = %s AND superseded_by IS NULL``（走部分索引）。

        ⚠️ **带 ORDER BY**：不排序的话 Python 侧 ``sorted`` 是稳定排序，
           相同键的条目会退化成 DB 返回序，而 PG 不保证顺序 ——
           于是「常驻档两轮输出逐字节相同」会变成 flaky。
        """
        stmt = (
            select(memories)
            .where(
                memories.c.owner == owner,
                memories.c.superseded_by.is_(None),
                memories.c.archived_at.is_(None),
            )
            .order_by(memories.c.id)
        )
        with self._engine.connect() as conn:
            return [self._row_to_memory(row) for row in conn.execute(stmt).mappings()]

    def touch(self, ids: list[str], now: datetime) -> None:
        if not ids:
            return
        current = to_naive_utc(now)
        with self._engine.begin() as conn:
            conn.execute(
                update(memories)
                .where(memories.c.id.in_(ids))
                .values(
                    hit_count=memories.c.hit_count + 1,
                    last_access=current,
                )
            )

    def update_used_count(self, ids: list[str], now: datetime) -> None:
        """``used_count += 1``（由 ``session.mark_used`` 调用）。"""
        if not ids:
            return
        current = to_naive_utc(now)
        with self._engine.begin() as conn:
            conn.execute(
                update(memories)
                .where(memories.c.id.in_(ids))
                .values(
                    used_count=memories.c.used_count + 1,
                    last_access=current,
                )
            )

    def update_tier(self, memory_id: str, tier: str) -> None:
        if tier not in ("resident", "state", "retrieved"):
            raise ValueError(f"update_tier: 非法档位 {tier!r}")
        with self._engine.begin() as conn:
            conn.execute(
                update(memories)
                .where(memories.c.id == memory_id)
                .values(tier=tier)
            )

    def archive_memories(self, memory_ids: list[str]) -> int:
        """归档而非删除：写 ``archived_at``，行还在（可回溯 + 合规）。"""
        if not memory_ids:
            return 0
        with self._engine.begin() as conn:
            result = conn.execute(
                update(memories)
                .where(
                    memories.c.id.in_(memory_ids),
                    memories.c.archived_at.is_(None),
                )
                .values(archived_at=func.now())
            )
            return int(result.rowcount or 0)

    def archived_count(self, owner: str) -> int:
        stmt = select(func.count()).select_from(memories).where(
            memories.c.owner == owner, memories.c.archived_at.isnot(None)
        )
        with self._engine.connect() as conn:
            return int(conn.execute(stmt).scalar_one())

    def remove_memory(self, memory_id: str) -> None:
        """用户行使删除权 —— 真删除。

        ⚠️ ``superseded_by`` 的外键是 ``ON DELETE SET NULL``，
           所以删掉「被别人指着的行」不会抛 ForeignKeyViolation。
        """
        with self._engine.begin() as conn:
            conn.execute(delete(memories).where(memories.c.id == memory_id))

    def purge_memories(self, memory_ids: list[str]) -> int:
        if not memory_ids:
            return 0
        with self._engine.begin() as conn:
            result = conn.execute(delete(memories).where(memories.c.id.in_(memory_ids)))
            return int(result.rowcount or 0)

    def count_memories(self, owner: str | None = None) -> dict[str, int]:
        """计数（验收 C 要贴行数）。"""
        base = select(func.count()).select_from(memories)
        if owner is not None:
            base = base.where(memories.c.owner == owner)
        with self._engine.connect() as conn:
            total = int(conn.execute(base).scalar_one())
            active_stmt = base.where(
                memories.c.superseded_by.is_(None), memories.c.archived_at.is_(None)
            )
            active = int(conn.execute(active_stmt).scalar_one())
        return {"total": total, "active": active}

    def get_memory(self, memory_id: str) -> Memory | None:
        stmt = select(memories).where(memories.c.id == memory_id)
        with self._engine.connect() as conn:
            row = conn.execute(stmt).mappings().first()
        return None if row is None else self._row_to_memory(row)

    # ------------------------------------------------------------------
    # 记忆检索（向量分支）
    # ------------------------------------------------------------------
    def vector_search(
        self,
        owner: str,
        emb: list[float],
        k: int,
        *,
        allowed_ids: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        """``ORDER BY embedding <=> %s LIMIT k``

        ``WHERE owner = %s AND superseded_by IS NULL`` ← 过滤写在 SQL 里。

        ⚠️ pgvector 0.6.0 上这个过滤**物理上发生在索引扫描之后**，所以
           用「内层放大 K 倍、外层再 LIMIT k」的两段式补召回。

        ``allowed_ids`` 是调用方已经算好的候选池（可见性 + 分档 + 预算退回）。
        给了它就把它作为**额外的 WHERE 条件**（而不是查完再在 Python 里过滤）——
        否则内层 ``LIMIT inner_k`` 可能被池外的行占满，池内的行一条都回不来。
        """
        if k <= 0:
            return []
        if allowed_ids is not None and not allowed_ids:
            return []
        vector = _v(emb)
        inner = max(k * 4, k + 20)
        params: dict[str, Any] = {
            "vec": vector,
            "owner": owner,
            "inner_k": inner,
            "k": k,
        }
        id_filter = ""
        if allowed_ids is not None:
            id_filter = " AND id = ANY(:allowed)"
            params["allowed"] = sorted(allowed_ids)

        with self._engine.begin() as conn:
            conn.execute(text(f"SET LOCAL hnsw.ef_search = {ANN_EF_SEARCH}"))
            rows = conn.execute(
                text(
                    f"""
                    SELECT id, score FROM (
                        SELECT id,
                               embedding <=> CAST(:vec AS vector) AS score
                          FROM memories
                         WHERE owner = :owner
                           AND superseded_by IS NULL
                           AND archived_at IS NULL
                           AND embedding IS NOT NULL
                           {id_filter}
                         ORDER BY embedding <=> CAST(:vec AS vector)
                         LIMIT :inner_k
                    ) AS ann
                    ORDER BY score
                    LIMIT :k
                    """
                ),
                params,
            ).all()
        return [(str(r[0]), float(r[1])) for r in rows]

    # ------------------------------------------------------------------
    # 知识检索（★ 父子两步）
    # ------------------------------------------------------------------
    def chunk_search(
        self,
        *,
        collection: str,
        acl: list[str],
        emb: list[float],
        k: int,
    ) -> list[tuple[str, float]]:
        """只查子块，权限过滤写在 SQL 里。

        ``WHERE collection = %s AND acl && %s AND is_parent = false``

        ★ 权限过滤（``acl &&``）在 ANN 之前 —— 见架构文档第九节。
          ⚠️ 同 ``vector_search``：0.6.0 上这是「后过滤」，用放大 k 的两段式缓解。
        """
        if k <= 0 or not acl:
            return []
        vector = _v(emb)
        inner = max(k * 4, k + 20)
        with self._engine.begin() as conn:
            conn.execute(text(f"SET LOCAL hnsw.ef_search = {ANN_EF_SEARCH}"))
            rows = conn.execute(
                text(
                    """
                    SELECT id, score FROM (
                        SELECT id,
                               embedding <=> CAST(:vec AS vector) AS score
                          FROM chunks
                         WHERE collection = :collection
                           AND acl && CAST(:acl AS text[])
                           AND is_parent = false
                           AND embedding IS NOT NULL
                         ORDER BY embedding <=> CAST(:vec AS vector)
                         LIMIT :inner_k
                    ) AS ann
                    ORDER BY score
                    LIMIT :k
                    """
                ),
                {
                    "vec": vector,
                    "collection": collection,
                    "acl": acl,
                    "inner_k": inner,
                    "k": k,
                },
            ).all()
        return [(str(r[0]), float(r[1])) for r in rows]

    def fetch_parents(self, child_ids: list[str]) -> list[Document]:
        """一次查回落点，并**按父块去重**。

        ``SELECT * FROM chunks WHERE id IN (SELECT parent_id FROM chunks WHERE id = ANY(%s))``

        ★ 多个子块可能同属一个父块 —— 返回列表里只出现一次，
          否则一个父块会占掉多个 topk 位置。
        ★ 返回顺序按「父块被子块命中的最早名次」排列：child_ids 是**有序**的，
          去重必须保序，否则「按父块去重」会把排序信息一起丢掉。
        """
        if not child_ids:
            return []
        with self._engine.connect() as conn:
            rows = conn.execute(
                select(chunks.c.id, chunks.c.parent_id).where(
                    chunks.c.id.in_(child_ids)
                )
            ).all()

        parent_of = {str(r[0]): (None if r[1] is None else str(r[1])) for r in rows}

        ordered_parents: list[str] = []
        seen: set[str] = set()
        for child_id in child_ids:
            parent_id = parent_of.get(child_id)
            if parent_id is None or parent_id in seen:
                continue
            seen.add(parent_id)
            ordered_parents.append(parent_id)
        if not ordered_parents:
            return []

        with self._engine.connect() as conn:
            parent_rows = {
                str(r["id"]): r
                for r in conn.execute(
                    select(chunks).where(chunks.c.id.in_(ordered_parents))
                ).mappings()
            }

        documents: list[Document] = []
        for parent_id in ordered_parents:
            row = parent_rows.get(parent_id)
            if row is not None:
                documents.append(self._row_to_document(row))
        return documents

    def save_chunks(self, docs: list[Document]) -> None:
        """落库 chunk。

        ★ 幂等：先按 ``source_doc`` 删掉该文档的既有行，再整体插入。
          为什么不用唯一约束：重切块会改变 chunk 数量与 id，唯一约束拦不住余下的旧行。
          同事务完成，中途失败不会留下半份文档。
        ★ 父块先插、子块后插（``parent_id`` 有外键）。
        """
        if not docs:
            return
        source_docs = {d.source_doc for d in docs}
        parents = [d for d in docs if d.is_parent]
        children = [d for d in docs if not d.is_parent]

        with self._engine.begin() as conn:
            conn.execute(delete(chunks).where(chunks.c.source_doc.in_(source_docs)))
            for doc in [*parents, *children]:
                conn.execute(
                    insert(chunks).values(
                        id=doc.id,
                        collection=doc.collection,
                        source_doc=doc.source_doc,
                        doc_version=doc.doc_version,
                        effective_date=doc.effective_date,
                        section=doc.section,
                        chunk_index=doc.chunk_index,
                        text=doc.text,
                        acl=list(doc.acl),
                        parent_id=doc.parent_id,
                        is_parent=doc.is_parent,
                        embedding_model=doc.embedding_model,
                    )
                )
                if doc.embedding is not None:
                    conn.execute(
                        text(
                            "UPDATE chunks SET embedding = CAST(:vec AS vector) "
                            "WHERE id = :id"
                        ),
                        {"vec": _v(doc.embedding), "id": doc.id},
                    )

    def count_chunks(self, source_doc: str | None = None) -> dict[str, int]:
        with self._engine.connect() as conn:
            total_stmt = select(func.count()).select_from(chunks)
            child_stmt = select(func.count()).select_from(chunks).where(
                chunks.c.is_parent.is_(False)
            )
            if source_doc is not None:
                total_stmt = total_stmt.where(chunks.c.source_doc == source_doc)
                child_stmt = child_stmt.where(chunks.c.source_doc == source_doc)
            total = int(conn.execute(total_stmt).scalar_one())
            child = int(conn.execute(child_stmt).scalar_one())
        return {"total": total, "children": child, "parents": total - child}

    def list_chunks(self, source_doc: str, *, parents_only: bool = False) -> list[Document]:
        stmt = select(chunks).where(chunks.c.source_doc == source_doc)
        if parents_only:
            stmt = stmt.where(chunks.c.is_parent.is_(True))
        stmt = stmt.order_by(chunks.c.chunk_index)
        with self._engine.connect() as conn:
            return [
                self._row_to_document(r) for r in conn.execute(stmt).mappings()
            ]

    # ------------------------------------------------------------------
    # 补丁
    # ------------------------------------------------------------------
    def append_patch(self, p: Patch) -> None:
        with self._engine.begin() as conn:
            row = conn.execute(
                insert(patches)
                .values(
                    owner=p.owner,
                    text=p.text,
                    priority=p.priority.value,
                    at=to_naive_utc(p.at),
                )
                .returning(patches.c.id)
            ).first()
        if row is not None:
            p.id = int(row[0])

    def pending_patches(self, owner: str) -> list[Patch]:
        stmt = (
            select(patches)
            .where(patches.c.owner == owner)
            .order_by(patches.c.id)
        )
        with self._engine.connect() as conn:
            return [
                Patch(
                    id=int(r["id"]),
                    owner=str(r["owner"]),
                    text=str(r["text"]),
                    priority=Priority(str(r["priority"])),
                    at=r["at"],
                )
                for r in conn.execute(stmt).mappings()
            ]

    def clear_patches(self, owner: str) -> None:
        with self._engine.begin() as conn:
            conn.execute(delete(patches).where(patches.c.owner == owner))

    def count_patches(self, owner: str | None = None) -> int:
        stmt = select(func.count()).select_from(patches)
        if owner is not None:
            stmt = stmt.where(patches.c.owner == owner)
        with self._engine.connect() as conn:
            return int(conn.execute(stmt).scalar_one())

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    @staticmethod
    def _row_to_memory(row: RowMapping) -> Memory:
        return Memory(
            id=str(row["id"]),
            owner=str(row["owner"]),
            what=str(row["what"]),
            type=MemoryType(str(row["type"])),
            when=row["when"],
            scope=str(row["scope"]),
            source=None if row["source"] is None else str(row["source"]),
            source_kind=SourceKind(str(row["source_kind"])),
            confidence=float(row["confidence"]),
            importance=float(row["importance"]),
            expires_at=row["expires_at"],
            hit_count=int(row["hit_count"]),
            used_count=int(row["used_count"]),
            tier=Tier(str(row["tier"])),
            last_access=row["last_access"],
            superseded_by=(
                None if row["superseded_by"] is None else str(row["superseded_by"])
            ),
            embedding=None,  # 全量拉取时不带向量（1 万条 ≈ 数百 MB）
        )

    @staticmethod
    def _row_to_document(row: RowMapping) -> Document:
        effective = row["effective_date"]
        if isinstance(effective, str):
            effective = date.fromisoformat(effective)
        acl = row["acl"]
        return Document(
            id=str(row["id"]),
            collection=str(row["collection"]),
            source_doc=str(row["source_doc"]),
            text=str(row["text"]),
            acl=list(acl) if acl is not None else [],
            doc_version=None if row["doc_version"] is None else str(row["doc_version"]),
            effective_date=effective,
            section=None if row["section"] is None else str(row["section"]),
            chunk_index=int(row["chunk_index"]),
            parent_id=None if row["parent_id"] is None else str(row["parent_id"]),
            is_parent=bool(row["is_parent"]),
            embedding_model=str(row["embedding_model"]),
            embedding=None,
        )


def _split_sql(ddl: str) -> list[str]:
    """把 DDL 文件切成可逐条执行的语句。

    ⚠️ 不是「按分号 split」那么简单 —— 文件里有 ``--`` 行注释，
    注释里可能出现分号与引号。这里逐行剥离注释后再按分号切。
    """
    lines: list[str] = []
    for raw in ddl.splitlines():
        line = raw.split("--", 1)[0]
        if line.strip():
            lines.append(line)
    body = "\n".join(lines)
    return [stmt.strip() for stmt in body.split(";") if stmt.strip()]


def libpq_to_sqlalchemy_url(dsn: str) -> str:
    """把 libpq 形式的 DSN 转成 SQLAlchemy URL。

    规格给的 DSN 是 ``host=... port=... dbname=... user=... password=...``
    （libpq key=value 形式），而 ``create_engine`` 只认 URL 或
    ``postgresql+psycopg://`` 连接串。所以必须显式转换。

    ⚠️ 先交给 ``psycopg.conninfo`` 解析一遍：它才是 libpq 语法的权威，
       能正确处理引号、反斜杠转义、``service=``、以及缺省值。
       手工 split 空格在密码含特殊字符时会出错。
    """
    from urllib.parse import quote

    from psycopg.conninfo import conninfo_to_dict

    try:
        params = conninfo_to_dict(dsn)
    except Exception as exc:  # noqa: BLE001 - 把 libpq 的报错原样带上
        raise ValueError(f"DSN 不是合法的 libpq 连接串：{exc}") from exc

    # conninfo_to_dict 的值类型是 str | int | None，统一转成 str 再用
    def field(name: str, default: str = "") -> str:
        value = params.get(name)
        return default if value is None else str(value)

    credentials = quote(field("user"), safe="")
    password = field("password")
    if password:
        credentials = f"{credentials}:{quote(password, safe='')}"
    host = field("host", "127.0.0.1")
    port = field("port", "5432")
    dbname = field("dbname") or field("database")
    return f"postgresql+psycopg://{credentials}@{host}:{port}/{dbname}"
