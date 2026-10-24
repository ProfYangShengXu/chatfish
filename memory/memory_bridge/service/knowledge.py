"""L2 服务：知识域（导入 / 版本 / 冲突）。

设计文档第九节。

不做什么：
  ❌ 不做知识图谱（明确不做）
  ❌ 不跟记忆混排（分别限流）
"""

from __future__ import annotations

from datetime import date

from ..domain.models import Document
from ..ports import Chunker, Embedder, MemoryStore


class KnowledgeService:
    def __init__(
        self,
        store: MemoryStore,
        embedder: Embedder,
        *,
        refuse_collections: frozenset[str] = frozenset(),
        embedding_model: str = "bge-small-zh-v1.5",
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._refuse_collections = refuse_collections
        self._embedding_model = embedding_model

    def ingest(
        self,
        *,
        collection: str,
        source_doc: str,
        text: str,
        doc_version: str,
        effective_date: date,
        acl: list[str],
        chunker: Chunker,
    ) -> int:
        """导入一份文档：

        ① ``chunker`` 切块（两级切：按结构 → 超长二次切）
        ② ``embedder.encode``
        ③ ``store.save_chunks``（带 ``embedding_model`` 标记，防混用）
        返回 chunk 数（父块 + 子块）。

        ⚠️ 只有**子块**带 embedding（检索只用子块）；父块也要入库（回填时取）。
        """
        if not text.strip():
            return 0
        if not acl:
            raise ValueError("ingest: acl 不能为空 —— 空 acl 的 chunk 检索永远命中不了")

        chunks = chunker.split(text, source_doc=source_doc)
        if not chunks:
            return 0

        for chunk in chunks:
            chunk.collection = collection
            chunk.doc_version = doc_version
            chunk.effective_date = effective_date
            chunk.acl = list(acl)
            chunk.embedding_model = self._embedding_model

        children = [c for c in chunks if not c.is_parent]
        if children:
            vectors = self._embedder.encode([c.text for c in children])
            for child, vector in zip(children, vectors, strict=True):
                child.embedding = vector

        self._store.save_chunks(chunks)
        return len(chunks)

    def resolve_version_conflict(self, docs: list[Document]) -> str:
        """同一问题召回到多个版本时，按策略处理：

        - 默认「择新 + 说明」（把旧版本作为注释带上）
        - 高风险 collection 配置为「拒答」

        返回可直接注入上下文的文本（不是只给个标记 —— 规格没规定返回什么，
        这里选「返回渲染好的说明」，让调用方零加工）。
        """
        if not docs:
            return ""
        if len(docs) == 1:
            return ""

        collections = {d.collection for d in docs}
        if collections & self._refuse_collections:
            names = "、".join(sorted(collections & self._refuse_collections))
            return (
                f"[版本冲突] 集合 {names} 属于高风险集合，"
                "召回到了多个版本且无法确定适用版本，**拒绝据此作答**，请人工确认。"
            )

        ordered = sorted(docs, key=_version_sort_key, reverse=True)
        newest, *older = ordered

        header = (
            f"[版本冲突] 同一问题召回到 {len(docs)} 个版本，"
            f"以下以最新版本为准（版本 {newest.doc_version or '未标注'}"
            f"，生效日期 {newest.effective_date.isoformat() if newest.effective_date else '未标注'}）。"
        )
        lines = [header, "", newest.text.strip()]
        for doc in older:
            version = doc.doc_version or "未标注"
            effective = (
                doc.effective_date.isoformat() if doc.effective_date else "未标注"
            )
            lines.append("")
            lines.append(
                f"（已被取代的旧版本 —— 版本 {version} | 生效日期 {effective} | "
                f"来源 {doc.source_doc}）"
            )
        return "\n".join(lines)


def _version_sort_key(doc: Document) -> tuple[date, str, int]:
    """版本新旧的排序键：先比生效日期，再比版本号字符串，最后比 chunk 序号。

    ⚠️ 版本号是自由文本（``"v2"`` / ``"2026-09"`` / ``"草案"``），没有统一语义，
       所以只能做「日期优先、文本兜底」的排序。已列入汇报的待确认项。
    """
    return (
        doc.effective_date or date.min,
        doc.doc_version or "",
        doc.chunk_index,
    )
