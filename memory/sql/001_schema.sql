-- Memory Bridge · 唯一一份 DDL
--
-- ★ 规格第〇节：Python 代码里**不重复写建表语句**，只认这个文件。
-- ★ 本文件是幂等的：可以重复执行（IF NOT EXISTS / DO 块）。
--
-- 关键字段说明（遗漏会导致自适应机制失效）：
--
--   hit_count   被【召回】的次数        → 往下赶：召回多但没用上 → 淘汰
--   used_count  被召回后【确实用上】的次数 → 往上抬：召回且用上了   → 升常驻
--
--   ★ 这是两个不同的信号，分别驱动「淘汰」和「升档」：
--       召回 10 次用上 8 次 → 升常驻（policy.should_promote 看 used_count）
--       召回 10 次用上 0 次 → 淘汰优先杀（scoring.is_low_value 看两者之比）
--     只记 hit_count 会让这两条规则打架。
--
--   tier  resident | state | retrieved —— 档位是【自适应】的，
--         初始值由 policy.assign_tier 在写入时给一次，之后靠升降级修正。

CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------------------
-- memories：记忆域
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS memories (
    id              TEXT PRIMARY KEY,
    owner           TEXT        NOT NULL,
    what            TEXT        NOT NULL,
    type            TEXT        NOT NULL,               -- preference|fact|constraint|project_state
    scope           TEXT        NOT NULL DEFAULT 'global',
    source          TEXT,
    source_kind     TEXT        NOT NULL DEFAULT 'user',  -- user|tool|external
    confidence      REAL        NOT NULL DEFAULT 0.5,
    importance      REAL        NOT NULL DEFAULT 0.5,
    "when"          TIMESTAMP   NOT NULL,               -- 写入时刻（naive UTC，见 domain/clock.py）
    expires_at      TIMESTAMP,
    hit_count       INT         NOT NULL DEFAULT 0,     -- 被召回次数
    used_count      INT         NOT NULL DEFAULT 0,     -- 召回后确实被用上的次数
    tier            TEXT        NOT NULL DEFAULT 'retrieved',  -- resident|state|retrieved
    last_access     TIMESTAMP,
    superseded_by   TEXT,                               -- 指向取代它的新条目
    archived_at     TIMESTAMP,
    created_at      TIMESTAMP   NOT NULL DEFAULT (now() AT TIME ZONE 'utc'),
    -- ★ 记忆的向量。
    --   ⚠️ 规格第〇节的 DDL 要求清单里**没有列这一列**，但
    --      impl/pg_store.py 的 supersede 判定（同 type/scope 下按余弦相似度找旧条目）
    --      与 vector_search（混合检索的向量路）都依赖它。
    --      少了它：写入能成功，但**每次检索都报 column "embedding" does not exist** ——
    --      而 mock 掉 store 的单测全绿。
    embedding       vector(512),

    -- ★ 外键加 ON DELETE SET NULL：
    --   架构文档的 DDL 是裸 REFERENCES，删除「被引用的新条目」时
    --   会抛 ForeignKeyViolation —— 而「删除权」是验收项 C 的一条。
    CONSTRAINT memories_superseded_by_fkey
        FOREIGN KEY (superseded_by) REFERENCES memories (id) ON DELETE SET NULL,

    CONSTRAINT memories_tier_check
        CHECK (tier IN ('resident', 'state', 'retrieved')),
    CONSTRAINT memories_source_kind_check
        CHECK (source_kind IN ('user', 'tool', 'external')),
    CONSTRAINT memories_confidence_check
        CHECK (confidence >= 0 AND confidence <= 1),
    CONSTRAINT memories_importance_check
        CHECK (importance >= 0 AND importance <= 1)
);

-- ---------------------------------------------------------------------------
-- 幂等迁移：老库补列
-- 只有 001_schema.sql 一份 DDL，没有 up/down 体系；所以「加列」这类变更
-- 用 IF NOT EXISTS 的形式写在同一个文件里，重复执行是安全的。
-- ---------------------------------------------------------------------------
ALTER TABLE memories ADD COLUMN IF NOT EXISTS embedding vector(512);
ALTER TABLE memories ADD COLUMN IF NOT EXISTS archived_at TIMESTAMP;
ALTER TABLE memories ADD COLUMN IF NOT EXISTS used_count INT NOT NULL DEFAULT 0;
ALTER TABLE memories ADD COLUMN IF NOT EXISTS tier TEXT NOT NULL DEFAULT 'retrieved';
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS is_parent BOOLEAN NOT NULL DEFAULT false;

-- ★ 部分索引：它是「隔离先于 ANN」的性能支撑。
--   active_memories() 与 vector_search() 都带
--   `WHERE owner = %s AND superseded_by IS NULL`，正好走这个索引。
CREATE INDEX IF NOT EXISTS memories_owner_active_idx
    ON memories (owner) WHERE superseded_by IS NULL;

-- 归档扫描与淘汰排序用
CREATE INDEX IF NOT EXISTS memories_owner_archived_idx
    ON memories (owner) WHERE archived_at IS NULL;

-- 记忆域的向量索引。
-- ⚠️ 与 chunks 不同，这里**故意用 HNSW + 低优先级**：记忆表通常只有几百到几万行，
--    而且检索总是先按 owner 划子集（多 owner 时分区的选择性很高）。
--    收益是 `vector_search` 不必退化成顺序扫描；代价是 filtered ANN 的召回塌陷，
--    已经在 impl/pg_store.py 用 `SET LOCAL hnsw.ef_search` + 放大 k 缓解。
CREATE INDEX IF NOT EXISTS memories_embedding_hnsw_idx
    ON memories USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);


-- ★ 防「同一事实并发写入两条 active」：同一 owner 下不允许两条内容完全相同的
--   active 记忆。没有这条约束时，server 与 cron 同时写同一事实会留下两条 active 行
--   （规格第九节的「精神分裂」）。
--   注意：这里只挡**完全相同**的文本；语义重复由写入管道的 should_remember 挡。
CREATE UNIQUE INDEX IF NOT EXISTS memories_owner_what_active_uniq
    ON memories (owner, what) WHERE superseded_by IS NULL AND archived_at IS NULL;

-- ---------------------------------------------------------------------------
-- chunks：知识域（父子两级）
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS chunks (
    id              TEXT PRIMARY KEY,
    collection      TEXT        NOT NULL,
    source_doc      TEXT        NOT NULL,
    doc_version     TEXT,
    effective_date  DATE,
    section         TEXT,
    chunk_index     INT         NOT NULL DEFAULT 0,
    text            TEXT        NOT NULL,
    acl             TEXT[]      NOT NULL DEFAULT '{}',
    -- ★ 只有【子块】填 parent_id；父块为 NULL
    parent_id       TEXT,
    -- ★ 区分父子（检索只查子块 —— WHERE is_parent = false）
    --   ⚠️ 架构文档那份 DDL 里没有这一列，照抄会在首次 chunk_search 时报
    --      column "is_parent" does not exist。
    is_parent       BOOLEAN     NOT NULL DEFAULT false,
    embedding_model TEXT        NOT NULL DEFAULT '',
    embedding       vector(512),                    -- 只有子块有值
    created_at      TIMESTAMP   NOT NULL DEFAULT (now() AT TIME ZONE 'utc'),

    CONSTRAINT chunks_parent_fkey
        FOREIGN KEY (parent_id) REFERENCES chunks (id) ON DELETE CASCADE
);

-- ★ 让 kb_ingest 幂等：同一份文档重复导入不会静默翻倍。
--   实现方式是「按 source_doc 先删后插」（见 impl/pg_store.py 的 save_chunks），
--   **不是**唯一约束 —— 因为重切块会改变 chunk 数量，唯一约束拦不住余下的旧行。
--   （架构文档那份 DDL 两者都没有，连跑两次 kb-ingest 会让行数翻倍。）

-- ★ 权限过滤（acl &&）与 collection 过滤要在 ANN 之前尽量缩小候选集。
--   ⚠️ pgvector 0.6.0 的 ANN 是「先扫索引、后过滤」，做不到真正的「过滤先于 ANN」；
--      这里建 B-tree 让 planner 在命中率低时改走精确扫描，是 0.6.0 上能做的手段之一。
CREATE INDEX IF NOT EXISTS chunks_collection_idx ON chunks (collection);
CREATE INDEX IF NOT EXISTS chunks_acl_idx ON chunks USING GIN (acl);
CREATE INDEX IF NOT EXISTS chunks_parent_idx ON chunks (parent_id);

-- ★ HNSW（pgvector >= 0.5.0 支持，本机 0.6.0 可用）。
--   代价：filtered ANN 时召回会塌陷（默认 hnsw.ef_search=40，10% 命中率下平均只回 4 行）。
--   缓解：查询侧显式 SET LOCAL hnsw.ef_search，并在应用层放大 k（见 impl/pg_store.py）。
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- ---------------------------------------------------------------------------
-- patches：尾部补丁缓冲
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS patches (
    id          BIGSERIAL PRIMARY KEY,
    owner       TEXT      NOT NULL,
    text        TEXT      NOT NULL,
    priority    TEXT      NOT NULL DEFAULT 'normal',   -- normal|urgent
    at          TIMESTAMP NOT NULL,
    created_at  TIMESTAMP NOT NULL DEFAULT (now() AT TIME ZONE 'utc'),

    CONSTRAINT patches_priority_check CHECK (priority IN ('normal', 'urgent'))
);

CREATE INDEX IF NOT EXISTS patches_owner_idx ON patches (owner, id);
