#!/usr/bin/env python3
"""知识库导入：解析文档（md / txt）→ 两级切块 → 建 embedding → 入库。

★ 两级切的产物天然就是**父子结构** —— 不要切成一组平铺的块。

| 只有平铺块                                   | 有父子两级                            |
|----------------------------------------------|---------------------------------------|
| 超长节二次切出来的小块，检索准但喂给模型的也是它 | **小块检索、父块喂模型** —— 两头都占 |

参数（规格第六节）：
  父块 1000-2000 字 / 子块 300-500 字 / 比例 3-5 个子块每父块
"""

from __future__ import annotations

import argparse
import re
import sys
import uuid
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from memory_bridge.domain.models import Document  # noqa: E402
from memory_bridge.ports import Chunker  # noqa: E402

if TYPE_CHECKING:  # pragma: no cover
    from memory_bridge.cli import Pipeline

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_FENCE = re.compile(r"^\s*```")


def build_parent_child(
    text: str,
    *,
    source_doc: str = "",
    parent_max: int = 2000,
    child_max: int = 400,
    overlap: int = 60,
) -> list[Document]:
    """两级切块。

    第一级：按结构切（标题 / 段落 / 列表）→ 【父块】喂给模型用
             ★ 短节【不合并】，该多短就多短
    第二级：父块内部按长度切 → 【子块】建索引用，每个记 ``parent_id``

    返回：父块（``is_parent=True``, ``parent_id=None``）+ 子块（``parent_id=父块 id``）

    ⚠️ 只有子块带 embedding（检索只用子块）；父块也要入库（回填时取）。
    """
    if overlap >= child_max:
        raise ValueError("overlap 必须小于 child_max，否则切块会不前进")

    sections = _split_sections(text)
    documents: list[Document] = []
    parent_index = 0

    for section_title, body in sections:
        # 第一级：按结构切完之后，超长节才二次切
        for parent_text in _hard_split(body, parent_max):
            if not parent_text.strip():
                continue
            parent_id = f"c_{uuid.uuid4().hex[:10]}"
            documents.append(
                Document(
                    id=parent_id,
                    collection="",  # 由 KnowledgeService 填
                    source_doc=source_doc,
                    text=parent_text,
                    acl=[],
                    section=section_title or None,
                    chunk_index=parent_index,
                    parent_id=None,
                    is_parent=True,
                )
            )
            parent_index += 1

            # 第二级：父块内部按长度切 → 子块
            for child_text in _sliding_split(parent_text, child_max, overlap):
                documents.append(
                    Document(
                        id=f"c_{uuid.uuid4().hex[:10]}",
                        collection="",
                        source_doc=source_doc,
                        text=child_text,
                        acl=[],
                        section=section_title or None,
                        chunk_index=parent_index,
                        parent_id=parent_id,
                        is_parent=False,
                    )
                )
                parent_index += 1

    return documents


def _split_sections(text: str) -> list[tuple[str, str]]:
    """按 markdown 标题切节；没有标题就整篇当一节。"""
    lines = text.splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_title = ""
    current_body: list[str] = []
    in_fence = False

    for line in lines:
        if _FENCE.match(line):
            in_fence = not in_fence
        match = None if in_fence else _HEADING.match(line)
        if match:
            if current_body:
                sections.append((current_title, current_body))
            current_title = match.group(2).strip()
            current_body = []
            continue
        current_body.append(line)

    if current_body:
        sections.append((current_title, current_body))

    out: list[tuple[str, str]] = []
    for title, body in sections:
        joined = "\n".join(body).strip()
        if joined:
            out.append((title, joined))
    return out or [("", text.strip())]


def _hard_split(text: str, limit: int) -> list[str]:
    """超长节按段落边界二次切，尽量不切在段落中间。"""
    if len(text) <= limit:
        return [text]

    parts: list[str] = []
    buffer = ""
    for paragraph in text.split("\n\n"):
        candidate = f"{buffer}\n\n{paragraph}" if buffer else paragraph
        if len(candidate) <= limit:
            buffer = candidate
            continue
        if buffer:
            parts.append(buffer)
        if len(paragraph) <= limit:
            buffer = paragraph
        else:
            # 单段就超长 → 按字符硬切
            for start in range(0, len(paragraph), limit):
                parts.append(paragraph[start : start + limit])
            buffer = ""
    if buffer:
        parts.append(buffer)
    return parts


def _sliding_split(text: str, size: int, overlap: int) -> list[str]:
    """父块内部按长度滑窗切子块。"""
    if len(text) <= size:
        return [text]
    chunks: list[str] = []
    step = size - overlap
    for start in range(0, len(text), step):
        chunk = text[start : start + size]
        if chunk.strip():
            chunks.append(chunk)
        if start + size >= len(text):
            break
    return chunks


class MarkdownChunker(Chunker):
    """``Chunker`` 端口的实现（薄封装，参数走规格给的默认值）。"""

    def __init__(
        self, *, parent_max: int = 2000, child_max: int = 400, overlap: int = 60
    ) -> None:
        self._parent_max = parent_max
        self._child_max = child_max
        self._overlap = overlap

    def split(self, text: str, *, source_doc: str) -> list[Document]:
        return build_parent_child(
            text,
            source_doc=source_doc,
            parent_max=self._parent_max,
            child_max=self._child_max,
            overlap=self._overlap,
        )


def read_document(path: Path) -> str:
    """读 md / txt（pdf 走 pypdf）。

    ⚠️ 一律显式 ``utf-8``：Windows 侧默认 locale 是 cp936。
    """
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="replace")


def ingest_file(
    pipeline: Pipeline,
    *,
    collection: str,
    path: Path,
    doc_version: str,
    effective_date: date,
    acl: list[str],
) -> int:
    """导入一个文件，返回 chunk 数。"""
    text = read_document(path)
    return pipeline.knowledge.ingest(
        collection=collection,
        source_doc=str(path),
        text=text,
        doc_version=doc_version,
        effective_date=effective_date,
        acl=acl,
        chunker=MarkdownChunker(),
    )


def main(argv: list[str] | None = None) -> int:
    from memory_bridge.cli import build_pipeline
    from memory_bridge.config import load_config, load_env_file

    parser = argparse.ArgumentParser(description="知识库导入")
    parser.add_argument("collection")
    parser.add_argument("files", nargs="+")
    parser.add_argument("--version", default="v1")
    parser.add_argument("--effective-date", default=None)
    parser.add_argument("--acl", default="public")
    args = parser.parse_args(argv)

    load_env_file(REPO_ROOT / ".env")
    cfg = load_config()
    pipeline = build_pipeline(cfg)

    effective = (
        date.fromisoformat(args.effective_date) if args.effective_date else date.today()
    )
    acl = [item.strip() for item in args.acl.split(",") if item.strip()]

    total = 0
    for raw in args.files:
        path = Path(raw)
        if not path.exists():
            print(f"[skip] 文件不存在：{path}", file=sys.stderr)
            return 1
        count = ingest_file(
            pipeline,
            collection=args.collection,
            path=path,
            doc_version=args.version,
            effective_date=effective,
            acl=acl,
        )
        total += count
        print(f"[ok] {path} → {count} chunks (version={args.version})")

    counters = getattr(pipeline.store, "count_chunks", None)
    if callable(counters):
        print(f"chunks 表：{counters()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
