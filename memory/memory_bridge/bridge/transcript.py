"""L3 桥：transcript 解析。

把各家 agent 的会话记录解析成纯文本。**这一层是「零配合触发」的基础。**

不做什么：
  ❌ 不改写原文（清洗是 ``domain/scrub`` 的事）
  ❌ 不做语义分析
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from enum import Enum
from pathlib import Path

from ..ports import TranscriptReader


class TranscriptFormat(str, Enum):
    CLAUDE_JSONL = "claude"  # ~/.claude/projects/**/*.jsonl
    DSH_SESSION = "dsh"  # ~/.dsh/sessions/**
    CODEX_JSONL = "codex"  # ~/.codex/sessions/**
    PLAIN_TEXT = "plain"


_PATH_HINTS: tuple[tuple[str, TranscriptFormat], ...] = (
    (".claude", TranscriptFormat.CLAUDE_JSONL),
    (".dsh", TranscriptFormat.DSH_SESSION),
    (".codex", TranscriptFormat.CODEX_JSONL),
)

_TRUNCATION_MARK = "\n\n...（中间部分因长度上限被省略）...\n\n"


def detect_format(path: Path) -> TranscriptFormat:
    """按路径特征 + 首行结构判格式。"""
    normalized = str(path).replace("\\", "/").lower()
    for hint, fmt in _PATH_HINTS:
        if hint in normalized:
            return fmt

    if path.suffix.lower() == ".jsonl":
        first_line = _first_nonempty_line(path)
        if first_line is not None:
            try:
                payload = json.loads(first_line)
            except json.JSONDecodeError:
                return TranscriptFormat.PLAIN_TEXT
            if isinstance(payload, dict):
                keys = set(payload)
                if "sessionId" in keys or "parentUuid" in keys:
                    return TranscriptFormat.CLAUDE_JSONL
                if "payload" in keys or "session_id" in keys:
                    return TranscriptFormat.DSH_SESSION
                if "role" in keys or "type" in keys:
                    return TranscriptFormat.CODEX_JSONL
        return TranscriptFormat.CLAUDE_JSONL

    return TranscriptFormat.PLAIN_TEXT


def read_transcript(path: Path, *, max_chars: int = 200_000) -> str:
    """读成纯文本（只保留 ``role + content``，丢掉工具噪声）。

    ⚠️ 截断要【保留首尾】—— 关键决定往往在开头（设定）和结尾（结论）。

    ⚠️ 一律 ``encoding="utf-8", errors="replace"``：
       transcript 是 UTF-8，而 Windows 侧默认 locale 是 cp936，
       不带 encoding 会让任何 emoji / 生僻字触发 UnicodeDecodeError ——
       在 cron 场景下这会变成**静默失败**（记忆永远不写入）。
    """
    if not path.exists():
        raise FileNotFoundError(f"transcript 不存在：{path}")

    fmt = detect_format(path)
    if fmt is TranscriptFormat.PLAIN_TEXT:
        text = path.read_text(encoding="utf-8", errors="replace")
    else:
        text = _read_jsonl_as_text(path)

    text = text.strip()
    if len(text) <= max_chars:
        return text

    head = max_chars * 2 // 3
    tail = max_chars - head
    return text[:head] + _TRUNCATION_MARK + text[-tail:]


def find_latest_session(agent: str, cwd: Path) -> Path | None:
    """定位最近一次会话的文件（会话结束钩子用它）。

    ``agent`` 取 ``claude`` / ``dsh`` / ``codex`` / ``plain``；
    找不到该 agent 的目录就返回 ``None``（**不抛异常** —— 钩子不该因为没样本就崩）。
    """
    roots = _candidate_roots(agent)
    best: tuple[int, float, str] | None = None
    best_path: Path | None = None

    for root in roots:
        if not root.exists():
            continue
        for candidate in _iter_transcript_files(root):
            try:
                mtime = candidate.stat().st_mtime
            except OSError:
                continue
            # 排序键：(是否属于当前项目, 修改时间, 路径) —— 路径兜底保证确定性
            key = (
                1 if _is_under_project(candidate, cwd) else 0,
                mtime,
                str(candidate),
            )
            if best is None or key > best:
                best = key
                best_path = candidate
    return best_path


def _candidate_roots(agent: str) -> list[Path]:
    home = Path.home()
    mapping = {
        "claude": [home / ".claude" / "projects"],
        "dsh": [Path(os.environ.get("DSH_HOME", home / ".dsh")) / "sessions"],
        "codex": [home / ".codex" / "sessions"],
    }
    if agent in mapping:
        return mapping[agent]
    return [path for paths in mapping.values() for path in paths]


def _iter_transcript_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    out: list[Path] = []
    # ★ DSH 会话是 session.v3.jsonl.zstd（zstd 压缩的 JSONL）。原 glob 匹不到 .zstd，
    #   于是 --scan --agent dsh 永远返回空（静默失败）。
    for pattern in ("*.jsonl.zstd", "*.jsonl", "*.json", "*.txt", "*.md"):
        out.extend(root.rglob(pattern))
    return out


def _is_under_project(candidate: Path, cwd: Path) -> bool:
    """claude 的 transcript 目录名是 cwd 的路径编码（``-`` 代替 ``/``）。"""
    try:
        cwd_slug = str(cwd).replace("\\", "-").replace("/", "-").lstrip("-")
    except Exception:  # noqa: BLE001
        return False
    return cwd_slug.lower() in str(candidate).replace("\\", "/").lower()


def _first_nonempty_line(path: Path) -> str | None:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.strip():
                    return line
    except OSError:
        return None
    return None


def _read_jsonl_as_text(path: Path) -> str:
    """逐行解析 JSONL，只保留 ``role`` + 文本内容。

    ⚠️ 坏行（半行 JSON、正在追加中被读到的行）**跳过而不是抛异常** ——
       transcript 可能在会话仍在写入时被扫到。
    """
    lines: list[str] = []

    # ★ DSH 存的是 session.v3.jsonl.zstd。先解压再按行解析；
    #   zstandard 不是硬依赖 —— 没装就返回空串而不报错，
    #   避免 cron 场景下整批静默失败。
    if path.suffix.lower() == ".zstd":
        try:
            import zstandard  # type: ignore
        except ImportError:
            return ""
        try:
            with path.open("rb") as fh:
                text = zstandard.ZstdDecompressor().stream_reader(fh).read().decode("utf-8", "replace")
        except Exception:  # noqa: BLE001 — 压缩文件可能正在写
            return ""
        raw_lines = text.splitlines()
    else:
        raw_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    for raw in raw_lines:
        stripped = raw.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        rendered = _render_record(payload)
        if rendered:
            lines.append(rendered)
    return "\n".join(lines)


def _render_record(payload: object) -> str:
    if isinstance(payload, str):
        return payload
    if not isinstance(payload, dict):
        return ""

    # 常见形状： {"role": ..., "content": ...} / {"type": ..., "message": {...}}
    role = payload.get("role") or payload.get("type") or ""
    content = payload.get("content")
    if content is None:
        message = payload.get("message")
        if isinstance(message, dict):
            role = message.get("role") or role
            content = message.get("content")
        else:
            content = message if isinstance(message, str) else None
    if content is None:
        # ★ DSH session.v3 的真实形状：
        #   {"type":"assistant/message","data":{"message":{"role":"assistant","content":[{"type":"text","text":"..."}]}}}
        #   原来只找 payload/payload，DSH 的整行渲染成空 → 提取不到任何内容。
        data = payload.get("data")
        if isinstance(data, dict):
            msg = data.get("message")
            if isinstance(msg, dict):
                role = msg.get("role") or role
                content = msg.get("content")
            if content is None:
                content = data.get("text")
    if content is None:
        # dsh / 其它： payload 里挂文本
        inner = payload.get("payload")
        if isinstance(inner, dict):
            content = inner.get("content") or inner.get("text")
    if content is None:
        content = payload.get("text")

    text = _flatten_content(content)
    if not text:
        return ""
    speaker = str(role) if role else "unknown"
    return f"{speaker}: {text}"


def _flatten_content(content: object) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                # 丢掉工具调用噪声，只留文本
                item_type = str(item.get("type", ""))
                if item_type in ("tool_use", "tool_result", "thinking"):
                    continue
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    if isinstance(content, dict):
        text = content.get("text") or content.get("content")
        return text if isinstance(text, str) else ""
    return ""


class FileTranscriptReader(TranscriptReader):
    """``TranscriptReader`` 端口的默认实现（就是本模块的函数）。"""

    def read(self, path: Path, *, max_chars: int) -> str:
        return read_transcript(path, max_chars=max_chars)

    def latest(self, agent: str, cwd: Path) -> Path | None:
        return find_latest_session(agent, cwd)


def parse_timestamp(value: object) -> datetime | None:
    """容错的时间戳解析（transcript 里的时间格式各家常不一样）。"""
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value))
    if not isinstance(value, str):
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


__all__ = [
    "TranscriptFormat",
    "detect_format",
    "read_transcript",
    "find_latest_session",
    "FileTranscriptReader",
    "parse_timestamp",
]
