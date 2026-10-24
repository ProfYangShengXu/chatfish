"""``bridge/`` 的测试。

重点是 D 层的三条硬约束：

1. ``ContextFileWriter`` **不碰标记之外的任何字节**
2. 常驻档在同一会话内**逐字节稳定**
3. 尾部补丁出现在文本**最后**
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from memory_bridge.bridge.context_file import (
    BEGIN_MARK,
    END_MARK,
    ContextFileWriter,
    extract_unmanaged,
    find_marker_lines,
)
from memory_bridge.bridge.server import (
    LOOPBACK_HOSTS,
    MemoryBridgeService,
    create_app,
    is_loopback,
)
from memory_bridge.bridge.session import SessionLifecycle, apply_tier_evolution
from memory_bridge.bridge.transcript import (
    TranscriptFormat,
    detect_format,
    find_latest_session,
    parse_timestamp,
    read_transcript,
)
from memory_bridge.config import Config
from memory_bridge.domain.models import (
    ForgetConfig,
    Memory,
    MemoryType,
    RecallConfig,
    SourceKind,
    Tier,
    TierConfig,
)
from memory_bridge.impl.reranker import LexicalAttributor, NoopAttributor
from memory_bridge.ports import NoopReranker
from memory_bridge.service.forget import ForgetPipeline
from memory_bridge.service.recall import RecallPipeline
from memory_bridge.service.write import WritePipeline

from .test_service import FakeEmbedder, FakeRetriever, InMemoryStore

NOW = datetime(2026, 9, 21, 12, 0, 0)
OWNER = "user:local"


def make_lifecycle(
    *,
    store: InMemoryStore,
    targets: list[Path],
    attributor: object | None = None,
) -> SessionLifecycle:
    write = WritePipeline(store, FakeEmbedder(), _policy_cfg(), TierConfig())
    recall = RecallPipeline(
        store,
        FakeRetriever(store),
        NoopReranker(),
        RecallConfig(recall_k=10, recall_final_k=3),
        TierConfig(),
    )
    forget = ForgetPipeline(store, ForgetConfig(), tier_cfg=TierConfig())
    return SessionLifecycle(
        write=write,
        recall=recall,
        store=store,
        context_file=ContextFileWriter(targets),
        forget=forget,
        tier_cfg=TierConfig(),
        forget_cfg=ForgetConfig(),
        attributor=attributor,  # type: ignore[arg-type]
    )


def _policy_cfg():  # type: ignore[no-untyped-def]
    from memory_bridge.domain.models import PolicyConfig

    return PolicyConfig()


# ---------------------------------------------------------------------------
# context_file —— D.1 的落点
# ---------------------------------------------------------------------------


class TestContextFileWriter:
    def test_creates_file_when_missing(self, tmp_path: Path) -> None:
        target = tmp_path / "CLAUDE.md"
        writer = ContextFileWriter([target])
        written = writer.write(owner=OWNER, resident="记忆内容", state="")
        assert written == [target]
        text = target.read_text(encoding="utf-8")
        assert BEGIN_MARK in text
        assert END_MARK in text
        assert "记忆内容" in text

    def test_appends_when_markers_absent(self, tmp_path: Path) -> None:
        """★ 标记不存在时必须**追加**，不能覆盖已有内容。"""
        target = tmp_path / "CLAUDE.md"
        original = "# 我的项目说明\n\n手写的、很重要的内容。\n"
        target.write_text(original, encoding="utf-8")

        ContextFileWriter([target]).write(owner=OWNER, resident="记忆", state="")
        text = target.read_text(encoding="utf-8")
        assert text.startswith(original)
        assert "记忆" in text

    def test_replaces_only_between_markers(self, tmp_path: Path) -> None:
        """★ 铁律：标记之外的字节一个都不能动。"""
        target = tmp_path / "CLAUDE.md"
        before = "# 项目说明\n\n用户手写的前言。\n\n"
        after = "\n\n## 用户手写的结尾\n\n这段也必须原样保留。\n"
        target.write_text(
            f"{before}{BEGIN_MARK}\n旧记忆\n{END_MARK}{after}", encoding="utf-8"
        )

        unmanaged_before = extract_unmanaged(target)
        ContextFileWriter([target]).write(owner=OWNER, resident="新记忆", state="")

        text = target.read_text(encoding="utf-8")
        assert "旧记忆" not in text
        assert "新记忆" in text
        assert extract_unmanaged(target) == unmanaged_before
        assert text.startswith(before)
        assert text.endswith(after)

    def test_identical_content_skips_write(self, tmp_path: Path) -> None:
        """内容没变就不写 —— 避免 mtime 变化引发无意义的重载。"""
        target = tmp_path / "CLAUDE.md"
        writer = ContextFileWriter([target])
        writer.write(owner=OWNER, resident="稳定内容", state="")
        mtime_before = target.stat().st_mtime_ns

        written = writer.write(owner=OWNER, resident="稳定内容", state="")
        assert written == []
        assert target.stat().st_mtime_ns == mtime_before

    def test_changed_content_does_write(self, tmp_path: Path) -> None:
        target = tmp_path / "CLAUDE.md"
        writer = ContextFileWriter([target])
        writer.write(owner=OWNER, resident="内容 A", state="")
        assert writer.write(owner=OWNER, resident="内容 B", state="") == [target]

    def test_read_current_returns_block(self, tmp_path: Path) -> None:
        target = tmp_path / "CLAUDE.md"
        writer = ContextFileWriter([target])
        writer.write(owner=OWNER, resident="读回来", state="")
        assert "读回来" in writer.read_current(owner=OWNER)

    def test_read_current_on_missing_file(self, tmp_path: Path) -> None:
        assert ContextFileWriter([tmp_path / "nope.md"]).read_current(owner=OWNER) == ""

    def test_read_block_without_markers(self, tmp_path: Path) -> None:
        target = tmp_path / "CLAUDE.md"
        target.write_text("没有标记\n", encoding="utf-8")
        assert ContextFileWriter([target]).read_current(owner=OWNER) == ""

    def test_multiple_targets(self, tmp_path: Path) -> None:
        first = tmp_path / "CLAUDE.md"
        second = tmp_path / "AGENTS.md"
        written = ContextFileWriter([first, second]).write(
            owner=OWNER, resident="都写", state=""
        )
        assert written == [first, second]
        assert "都写" in first.read_text(encoding="utf-8")
        assert "都写" in second.read_text(encoding="utf-8")

    def test_encoding_is_utf8_not_locale(self, tmp_path: Path) -> None:
        """★ 写成 GBK 的话「逐字节相同」验收照样通过，但 agent 读到的是乱码。"""
        target = tmp_path / "CLAUDE.md"
        ContextFileWriter([target]).write(
            owner=OWNER, resident="中文 · emoji 🎯 · Ω", state=""
        )
        raw = target.read_bytes()
        assert "中文".encode() in raw
        assert "🎯".encode() in raw
        assert raw.decode("utf-8")  # 必须能按 UTF-8 解码

    def test_state_none_renders_placeholder(self, tmp_path: Path) -> None:
        target = tmp_path / "CLAUDE.md"
        assert ContextFileWriter([target]).write(
            owner=OWNER, resident="", state=""
        ) == [target]
        assert "暂无" in target.read_text(encoding="utf-8")

    def test_marker_mentioned_inline_is_not_a_delimiter(self, tmp_path: Path) -> None:
        """★ 回归测试：标记被『提及』（写在正文/反引号里）时**绝不能**当分隔符。

        实测踩过的坑：用户在 CLAUDE.md 里写了一句
        「记忆桥只能改 `<!-- memory-bridge:begin -->` 与 … 之间的东西」，
        按子串查找的实现会把整段记忆区**插进那句话中间**，
        直接毁掉用户的文件 —— 而「标记之外一字未动」的检查是看不出来的
        （因为标记本身就是被改写的那一行）。
        """
        target = tmp_path / "CLAUDE.md"
        original = (
            "# 项目说明\n\n"
            f"记忆桥只能改 `{BEGIN_MARK}` 与 `{END_MARK}` 之间的东西。\n\n"
            "## 我自己的约定\n\n- 不要在周一发版\n"
        )
        target.write_text(original, encoding="utf-8")

        writer = ContextFileWriter([target])
        writer.write(owner=OWNER, resident="记忆内容", state="")

        text = target.read_text(encoding="utf-8")
        # 原来那句话必须**逐字**还在
        assert f"只能改 `{BEGIN_MARK}` 与 `{END_MARK}` 之间的东西" in text
        # 记忆区被追加到末尾，而不是插进句子里
        assert text.startswith(original)
        assert text.count(BEGIN_MARK) == 2  # 原句里的那次提及 + 真正的分隔符
        assert find_marker_lines(text) != (-1, -1)
        # 第二次写：只改真正的分隔符区，原句不受影响
        writer.write(owner=OWNER, resident="更新后的记忆", state="")
        after = target.read_text(encoding="utf-8")
        assert f"只能改 `{BEGIN_MARK}` 与 `{END_MARK}` 之间的东西" in after
        assert "更新后的记忆" in after
        assert "记忆内容" not in after

    def test_marker_recovery_from_broken_file(self, tmp_path: Path) -> None:
        """只有 BEGIN 没有 END（用户误删）→ 当标记不存在处理，追加。"""
        target = tmp_path / "CLAUDE.md"
        target.write_text(f"用户内容\n{BEGIN_MARK}\n坏了\n", encoding="utf-8")
        ContextFileWriter([target]).write(owner=OWNER, resident="新", state="")
        text = target.read_text(encoding="utf-8")
        assert "用户内容" in text
        assert text.count(BEGIN_MARK) == 1


# ---------------------------------------------------------------------------
# session —— D.2 / D.3
# ---------------------------------------------------------------------------


class TestSessionStart:
    def test_resident_is_byte_stable_across_calls(self, tmp_path: Path) -> None:
        """★ D.3：同一会话跑两轮，常驻档输出逐字节相同。"""
        store = InMemoryStore()
        lifecycle = make_lifecycle(store=store, targets=[tmp_path / "CLAUDE.md"])
        lifecycle._write.remember_explicit(owner=OWNER, text="不要用 ORM", now=NOW)

        first = lifecycle.on_session_start(owner=OWNER, cwd=tmp_path)
        second = lifecycle.on_session_start(owner=OWNER, cwd=tmp_path)
        assert first.encode("utf-8") == second.encode("utf-8")

    def test_resident_written_to_context_file(self, tmp_path: Path) -> None:
        target = tmp_path / "CLAUDE.md"
        store = InMemoryStore()
        lifecycle = make_lifecycle(store=store, targets=[target])
        lifecycle._write.remember_explicit(owner=OWNER, text="不要用 ORM", now=NOW)

        text = lifecycle.on_session_start(owner=OWNER, cwd=tmp_path)
        assert text
        assert "ORM" in target.read_text(encoding="utf-8")

    def test_no_targets_does_not_crash(self) -> None:
        store = InMemoryStore()
        lifecycle = make_lifecycle(store=store, targets=[])
        assert isinstance(lifecycle.on_session_start(owner=OWNER), str)

    def test_empty_store_returns_empty(self, tmp_path: Path) -> None:
        store = InMemoryStore()
        lifecycle = make_lifecycle(store=store, targets=[tmp_path / "CLAUDE.md"])
        assert lifecycle.on_session_start(owner=OWNER) == ""


class TestSessionEnd:
    def test_end_to_end_from_text(self, tmp_path: Path) -> None:
        store = InMemoryStore()
        lifecycle = make_lifecycle(store=store, targets=[tmp_path / "CLAUDE.md"])
        report = lifecycle.on_session_end(
            owner=OWNER,
            transcript_text="用户：先看看 A\n用户：最终决定采用方案 B\n",
            now=NOW,
        )
        assert report.written
        assert store.memories

    def test_end_from_file(self, tmp_path: Path) -> None:
        transcript = tmp_path / "s.jsonl"
        transcript.write_text(
            json.dumps({"role": "user", "content": "决定采用 Rust"}, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
        store = InMemoryStore()
        lifecycle = make_lifecycle(store=store, targets=[tmp_path / "CLAUDE.md"])
        report = lifecycle.on_session_end(
            owner=OWNER, transcript_path=transcript, now=NOW
        )
        assert report.written

    def test_requires_some_input(self, tmp_path: Path) -> None:
        store = InMemoryStore()
        lifecycle = make_lifecycle(store=store, targets=[tmp_path / "CLAUDE.md"])
        with pytest.raises(ValueError):
            lifecycle.on_session_end(owner=OWNER, now=NOW)

    def test_empty_transcript_does_not_crash(self, tmp_path: Path) -> None:
        store = InMemoryStore()
        lifecycle = make_lifecycle(store=store, targets=[tmp_path / "CLAUDE.md"])
        report = lifecycle.on_session_end(owner=OWNER, transcript_text="   ", now=NOW)
        assert report.written == []

    def test_end_runs_forget(self, tmp_path: Path) -> None:
        store = InMemoryStore()
        lifecycle = make_lifecycle(store=store, targets=[tmp_path / "CLAUDE.md"])
        lifecycle.on_session_end(
            owner=OWNER, transcript_text="用户：决定采用 B\n", now=NOW
        )
        assert store.memories  # 治理跑过且没把新写的删掉


class TestMarkUsed:
    def test_mark_used_increments_used_count(self, tmp_path: Path) -> None:
        """★ 这是 used_count 的**唯一写入入口** —— 它必须真的可用。"""
        store = InMemoryStore()
        lifecycle = make_lifecycle(store=store, targets=[tmp_path / "CLAUDE.md"])
        memory = lifecycle._write.remember_explicit(
            owner=OWNER, text="用户偏好 pytest", now=NOW
        )
        assert store.memories[memory.id].used_count == 0
        count = lifecycle.mark_used(owner=OWNER, memory_ids=[memory.id], now=NOW)
        assert count == 1
        assert store.memories[memory.id].used_count == 1

    def test_mark_used_empty_is_noop(self, tmp_path: Path) -> None:
        store = InMemoryStore()
        lifecycle = make_lifecycle(store=store, targets=[tmp_path / "CLAUDE.md"])
        assert lifecycle.mark_used(owner=OWNER, memory_ids=[]) == 0

    def test_noop_attributor_does_not_inflate_used_count(
        self, tmp_path: Path
    ) -> None:
        """默认归因器必须**诚实**：什么都不标，而不是用启发式灌水。"""
        store = InMemoryStore()
        lifecycle = make_lifecycle(
            store=store, targets=[tmp_path / "CLAUDE.md"], attributor=NoopAttributor()
        )
        lifecycle._write.remember_explicit(owner=OWNER, text="不要用 ORM", now=NOW)
        lifecycle.recall_tail(owner=OWNER, query="ORM")
        lifecycle.on_session_end(
            owner=OWNER, transcript_text="用户：不要用 ORM\n", now=NOW
        )
        assert all(m.used_count == 0 for m in store.memories.values())

    def test_lexical_attributor_marks_exact_reuse(self, tmp_path: Path) -> None:
        store = InMemoryStore()
        lifecycle = make_lifecycle(
            store=store,
            targets=[tmp_path / "CLAUDE.md"],
            attributor=LexicalAttributor(threshold=0.5),
        )
        memory = lifecycle._write.remember_explicit(
            owner=OWNER, text="用户偏好 pytest", now=NOW
        )
        lifecycle.recall_tail(owner=OWNER, query="pytest")
        lifecycle.on_session_end(
            owner=OWNER, transcript_text="用户偏好 pytest", now=NOW
        )
        assert store.memories[memory.id].used_count >= 1

    def test_promotion_after_mark_used(self, tmp_path: Path) -> None:
        """used_count 攒够 → 升常驻档。"""
        store = InMemoryStore()
        lifecycle = make_lifecycle(store=store, targets=[tmp_path / "CLAUDE.md"])
        memory = lifecycle._write.remember_explicit(
            owner=OWNER, text="用户偏好 pytest", now=NOW
        )
        store.memories[memory.id].tier = Tier.RETRIEVED
        for _ in range(TierConfig().promote_threshold):
            lifecycle.mark_used(owner=OWNER, memory_ids=[memory.id], now=NOW)

        changes = apply_tier_evolution(
            [store.memories[memory.id]], TierConfig(promote_threshold=3)
        )
        assert changes == [(memory.id, "resident")]


class TestFlushPatches:
    def test_flush_merges_and_clears(self, tmp_path: Path) -> None:
        """★ 补丁必须有消费者 —— 否则尾部补丁无限增长。"""
        target = tmp_path / "CLAUDE.md"
        store = InMemoryStore()
        lifecycle = make_lifecycle(store=store, targets=[target])

        lifecycle._write.remember_explicit(owner=OWNER, text="不要用 ORM", now=NOW)
        assert len(store.patches) == 1

        flushed = lifecycle.flush_patches(owner=OWNER)
        assert flushed == 1
        assert store.patches == []
        assert "ORM" in target.read_text(encoding="utf-8")

    def test_patch_buffer_is_bounded(self, tmp_path: Path) -> None:
        """★ 补丁缓冲必须有上限 —— 否则每次 recall 都把全部历史补丁重放一遍。"""
        store = InMemoryStore()
        lifecycle = make_lifecycle(store=store, targets=[tmp_path / "CLAUDE.md"])
        for index in range(30):
            lifecycle._write.remember_explicit(
                owner=OWNER, text=f"不要用第 {index} 种写法", now=NOW
            )
        assert len(store.patches) > lifecycle._patch_flush_threshold
        lifecycle.recall_tail(owner=OWNER, query="写法")
        assert len(store.patches) <= lifecycle._patch_flush_threshold

    def test_flush_without_patches(self, tmp_path: Path) -> None:
        store = InMemoryStore()
        lifecycle = make_lifecycle(store=store, targets=[tmp_path / "CLAUDE.md"])
        assert lifecycle.flush_patches(owner=OWNER) == 0


class TestRenderTail:
    def test_recall_tail_contains_patch(self, tmp_path: Path) -> None:
        store = InMemoryStore()
        lifecycle = make_lifecycle(store=store, targets=[tmp_path / "CLAUDE.md"])
        lifecycle._write.remember_explicit(owner=OWNER, text="不要用 ORM", now=NOW)
        tail = lifecycle.recall_tail(owner=OWNER, query="ORM")
        assert "ORM" in tail

    def test_render_tail_alias(self, tmp_path: Path) -> None:
        store = InMemoryStore()
        lifecycle = make_lifecycle(store=store, targets=[tmp_path / "CLAUDE.md"])
        assert lifecycle.render_tail(owner=OWNER, query="x") == lifecycle.recall_tail(
            owner=OWNER, query="x"
        )


# ---------------------------------------------------------------------------
# transcript
# ---------------------------------------------------------------------------


class TestTranscript:
    def test_detect_claude(self, tmp_path: Path) -> None:
        path = tmp_path / ".claude" / "projects" / "x" / "a.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps({"sessionId": "s", "role": "user", "content": "hi"}) + "\n",
            encoding="utf-8",
        )
        assert detect_format(path) is TranscriptFormat.CLAUDE_JSONL

    def test_detect_dsh_by_path(self, tmp_path: Path) -> None:
        path = tmp_path / ".dsh" / "sessions" / "a.json"
        path.parent.mkdir(parents=True)
        path.write_text("{}\n", encoding="utf-8")
        assert detect_format(path) is TranscriptFormat.DSH_SESSION

    def test_detect_plain_text(self, tmp_path: Path) -> None:
        path = tmp_path / "notes.txt"
        path.write_text("hello\n", encoding="utf-8")
        assert detect_format(path) is TranscriptFormat.PLAIN_TEXT

    def test_read_plain_keeps_content(self, tmp_path: Path) -> None:
        path = tmp_path / "notes.txt"
        path.write_text("用户：决定采用 B\n", encoding="utf-8")
        assert "决定采用 B" in read_transcript(path, max_chars=1000)

    def test_read_jsonl_keeps_role_and_content(self, tmp_path: Path) -> None:
        path = tmp_path / "s.jsonl"
        path.write_text(
            "\n".join(
                [
                    json.dumps({"role": "user", "content": "你好"}, ensure_ascii=False),
                    json.dumps(
                        {"role": "assistant", "content": "好的"}, ensure_ascii=False
                    ),
                ]
            ),
            encoding="utf-8",
        )
        text = read_transcript(path, max_chars=1000)
        assert "user: 你好" in text
        assert "assistant: 好的" in text

    def test_tool_noise_is_dropped(self, tmp_path: Path) -> None:
        path = tmp_path / "s.jsonl"
        path.write_text(
            json.dumps(
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "name": "bash", "input": {"cmd": "ls"}},
                        {"type": "text", "text": "真正的回答"},
                    ],
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        text = read_transcript(path, max_chars=1000)
        assert "真正的回答" in text
        assert "tool_use" not in text

    def test_broken_json_line_is_skipped(self, tmp_path: Path) -> None:
        """★ 会话仍在写入时可能读到半行 —— 必须跳过而不是抛异常。"""
        path = tmp_path / "s.jsonl"
        path.write_text(
            '{"role": "user", "content": "ok"}\n{"role": "user", "cont',
            encoding="utf-8",
        )
        text = read_transcript(path, max_chars=1000)
        assert "ok" in text

    def test_truncation_keeps_head_and_tail(self, tmp_path: Path) -> None:
        """★ 关键决定往往在开头（设定）和结尾（结论）。"""
        path = tmp_path / "notes.txt"
        path.write_text("头部关键设定" + "填" * 5000 + "尾部最终结论", encoding="utf-8")
        text = read_transcript(path, max_chars=400)
        assert len(text) <= 500
        assert text.startswith("头部关键设定")
        assert text.endswith("尾部最终结论")
        assert "省略" in text

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            read_transcript(tmp_path / "nope.jsonl", max_chars=100)

    def test_emoji_does_not_crash(self, tmp_path: Path) -> None:
        """★ 不带 encoding 读到非 UTF-8 字节会抛 UnicodeDecodeError。"""
        path = tmp_path / "s.jsonl"
        path.write_bytes(b'{"role":"user","content":"\xff\xfe bad bytes"}\n')
        assert isinstance(read_transcript(path, max_chars=1000), str)

    def test_find_latest_session_missing_root(self, tmp_path: Path) -> None:
        assert find_latest_session("definitely-not-an-agent", tmp_path) is None

    def test_parse_timestamp_iso(self) -> None:
        assert parse_timestamp("2026-09-21T12:00:00Z") is not None

    def test_parse_timestamp_epoch(self) -> None:
        assert parse_timestamp(1_700_000_000) is not None

    def test_parse_timestamp_garbage(self) -> None:
        assert parse_timestamp("not a date") is None


# ---------------------------------------------------------------------------
# server
# ---------------------------------------------------------------------------


class TestServerSafety:
    def test_only_loopback_allowed(self) -> None:
        assert is_loopback("127.0.0.1")
        assert is_loopback("::1")
        assert not is_loopback("0.0.0.0")
        assert not is_loopback("192.168.1.5")

    def test_default_host_is_loopback(self) -> None:
        assert Config().host in LOOPBACK_HOSTS

    def test_create_app_rejects_non_loopback(self, tmp_path: Path) -> None:
        """★ 非回环必须在**启动时**就拒绝，而不是监听之后。"""
        cfg = Config(host="0.0.0.0", port=9999, context_files=[])
        service = MemoryBridgeService(pipeline=_FakePipeline(), cfg=cfg)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="回环"):
            create_app(service)

    def test_health_reports_db_failure_honestly(self) -> None:
        """★ /health 不能恒真 —— PG 挂了必须能看出来。"""

        class BrokenStore:
            def ping(self) -> bool:
                raise RuntimeError("connection refused")

        cfg = Config(host="127.0.0.1")
        pipeline = _FakePipeline()
        pipeline.store = BrokenStore()  # type: ignore[assignment]
        service = MemoryBridgeService(pipeline=pipeline, cfg=cfg)  # type: ignore[arg-type]
        payload = service.health()
        assert payload["ok"] is False
        assert payload["db"] is False
        assert payload["db_error"] == "RuntimeError"

    def test_health_true_when_all_good(self) -> None:
        cfg = Config(host="127.0.0.1")
        service = MemoryBridgeService(pipeline=_FakePipeline(), cfg=cfg)  # type: ignore[arg-type]
        payload = service.health()
        assert payload["ok"] is True
        assert payload["db"] is True


class _FakePipeline:
    """只满足 ``MemoryBridgeService`` 用到的那几个属性。"""

    class _Store:
        def ping(self) -> bool:
            return True

    def __init__(self) -> None:
        self.store = self._Store()
        self.loaded_models = ["bge-small-zh-v1.5"]
        self.missing_models: list[str] = []


# ---------------------------------------------------------------------------
# 端到端：真模型 + 真库 + 真文件（只在显式开启时跑）
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestInjectionEndToEnd:
    def test_marker_region_untouched_and_stable(self, tmp_path: Path) -> None:
        target = tmp_path / "CLAUDE.md"
        user_content = "# 用户手写标题\n\n这段内容必须一字不动。\n"
        target.write_text(user_content, encoding="utf-8")

        store = InMemoryStore()
        lifecycle = make_lifecycle(store=store, targets=[target])
        lifecycle._write.remember_explicit(owner=OWNER, text="不要用 ORM", now=NOW)

        lifecycle.on_session_start(owner=OWNER, cwd=tmp_path)
        first = target.read_text(encoding="utf-8")
        lifecycle.on_session_start(owner=OWNER, cwd=tmp_path)
        second = target.read_text(encoding="utf-8")

        assert first == second
        assert first.startswith(user_content)
        assert "不要用 ORM" in first
