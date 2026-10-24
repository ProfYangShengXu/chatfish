"""L3 桥：会话生命周期。

定义「会话开始/结束」这两个时刻要做什么。**这是决策 2（确定性触发）的落点。**

两条触发路径（都不依赖模型自觉）：

* 路径 A：有 hook 的 agent → 退出时调 ``scripts/session_end.py``
* 路径 B：无 hook → 由 cron / 后台进程定期扫描 transcript 目录
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ..domain import policy
from ..domain.clock import to_naive_utc
from ..domain.models import ForgetConfig, Memory, Patch, TierConfig, utc_now
from ..ports import (
    MemoryStore,
    Reranker,
    Retriever,
    TranscriptReader,
    UsageAttributor,
)
from ..service import inject
from ..service.forget import ForgetPipeline
from ..service.recall import RecallPipeline, _count_tokens
from ..service.write import WritePipeline, WriteReport
from .context_file import ContextFileWriter


class NeedManualIntegration(RuntimeError):
    """容量治理要求「当场整合或删除」时抛出（吸收自 Hermes）。

    ⚠️ 这是**领域失败**，但规格明说「由上层【报错】要求模型当场整合或删除」，
       所以这里用异常 —— 它需要打断调用方，而不是让它继续。
    """


@dataclass(slots=True)
class SessionEndReport:
    written: list[Memory] = field(default_factory=list)
    archived: list[str] = field(default_factory=list)
    promoted: list[str] = field(default_factory=list)
    demoted: list[str] = field(default_factory=list)
    attributed: list[str] = field(default_factory=list)
    need_manual: bool = False
    note: str = ""


class SessionLifecycle:
    def __init__(
        self,
        write: WritePipeline,
        recall: RecallPipeline,
        store: MemoryStore,
        context_file: ContextFileWriter,
        forget: ForgetPipeline,
        *,
        tier_cfg: TierConfig | None = None,
        forget_cfg: ForgetConfig | None = None,
        reranker: Reranker | None = None,
        retriever: Retriever | None = None,
        attributor: UsageAttributor | None = None,
        transcript_reader: TranscriptReader | None = None,
        max_transcript_chars: int = 200_000,
        recall_cfg_patch_threshold: int = 20,
    ) -> None:
        self._write = write
        self._recall = recall
        self._store = store
        self._context_file = context_file
        self._forget = forget
        self._tier_cfg = tier_cfg or TierConfig()
        self._forget_cfg = forget_cfg or ForgetConfig()
        self._reranker = reranker
        self._retriever = retriever
        self._attributor = attributor
        self._reader = transcript_reader
        self._max_transcript_chars = max_transcript_chars
        self._patch_flush_threshold = recall_cfg_patch_threshold
        self._last_recall: dict[str, list[Memory]] = {}

    # ------------------------------------------------------------------
    # 会话开始
    # ------------------------------------------------------------------
    def on_session_start(self, *, owner: str, cwd: Path | None = None) -> str:
        """返回要注入的常驻文本。

        ⚠️ 这个值在【整个会话内必须稳定】—— 不要带上当前时间或轮次计数。
           所以这里不调检索（``query`` 会变），只取常驻档 + 状态档。

        ★ 这也是 ``ContextFileWriter.write`` 的调用点 ——
          「写进 CLAUDE.md」是手段，借 agent 原生的上下文加载机制。
        """
        ctx = self._recall.recall(
            owner=owner,
            query="",  # ★ 空 query → 不产生检索档，也不 touch
            cwd=str(cwd) if cwd is not None else None,
            now=self._stable_now(),
        )
        resident = inject.render_resident(ctx)
        state = inject.render_state(ctx)
        if self._context_file.targets:
            self._context_file.write(owner=owner, resident=resident, state=state)
        return resident

    # ------------------------------------------------------------------
    # 会话结束
    # ------------------------------------------------------------------
    def on_session_end(
        self,
        *,
        owner: str,
        transcript_path: Path | None = None,
        transcript_text: str | None = None,
        now: datetime | None = None,
    ) -> SessionEndReport:
        """会话结束的完整流程：

        ① ``transcript.read_transcript``
        ② ``write.ingest_session_end``
        ③ ``forget.run``（顺带做一次容量治理）+ 档位自适应
        ④ 若 ``need_manual=True`` → 报告里标记，由上层报错
        """
        current = to_naive_utc(now) if now is not None else utc_now()

        if transcript_text is None:
            if transcript_path is None:
                raise ValueError("on_session_end: 必须给 transcript_path 或 transcript_text")
            if self._reader is None:
                from .transcript import read_transcript

                text = read_transcript(
                    transcript_path, max_chars=self._max_transcript_chars
                )
            else:
                text = self._reader.read(
                    transcript_path, max_chars=self._max_transcript_chars
                )
        else:
            text = transcript_text

        report = SessionEndReport()

        # ② 批量提取
        if text.strip():
            write_report = self._ingest_text(owner=owner, text=text, now=current)
            report.written = list(write_report.written)

        # ③ 容量治理（离线跑，不在读取路径上）
        forget_report = self._forget.run(owner=owner, now=current)
        report.archived = list(forget_report.archived)
        report.need_manual = forget_report.need_manual
        report.note = forget_report.reason

        # ③b 档位自适应（与淘汰分开，可单独调用）
        evolution = self._forget.apply_tier_evolution(owner=owner, now=current)
        report.promoted = list(evolution.promoted)
        report.demoted = list(evolution.demoted)

        # ③c 归因「哪些记忆真的被用上了」→ used_count
        recalled = self._last_recall.get(owner) or []
        if recalled:
            report.attributed = self._attribute(owner=owner, memories=recalled, answer=text)

        return report

    def _ingest_text(self, *, owner: str, text: str, now: datetime) -> WriteReport:
        """让 ``WritePipeline.ingest_session_end`` 的返回既可读又不丢信息。

        规格里 ``ingest_session_end`` 返回 ``list[Memory]``；这里需要报告，
        所以优先调用详细版本（存在就用），否则包一层。
        """
        detailed = getattr(self._write, "ingest_session_end_detailed", None)
        if callable(detailed):
            result: WriteReport = detailed(owner=owner, transcript=text, now=now)
            return result
        memories = self._write.ingest_session_end(owner=owner, transcript=text, now=now)
        return WriteReport(written=list(memories))

    # ------------------------------------------------------------------
    # 反馈闭环
    # ------------------------------------------------------------------
    def mark_used(self, *, owner: str, memory_ids: list[str], now: datetime | None = None) -> int:
        """★ 标记「这些记忆在本轮被实际用上了」（``used_count += 1``）。

        **为什么不能跟 ``hit_count`` 一起在召回时加**：

          召回 = 「检索器认为它相关」      → ``hit_count``
          用上 = 「模型确实拿它做了判断」  → ``used_count``

        两者混同的话，「升档」和「淘汰」两条规则都会失效。

        ⚠️ 规格要求这条路径必须真的被接上（否则 ``used_count`` 恒为 0，
           ``is_low_value`` 会退化成「优先淘汰最常被召回的记忆」）。
           入口有两个：``session.end`` 的归因、以及 ``memory mark-used`` 命令。
        """
        if not memory_ids:
            return 0
        current = to_naive_utc(now) if now is not None else utc_now()
        updater = getattr(self._store, "update_used_count", None)
        if not callable(updater):
            return 0
        updater(memory_ids, current)
        return len(memory_ids)

    def _attribute(self, *, owner: str, memories: list[Memory], answer: str) -> list[str]:
        if self._attributor is None:
            return []
        used = self._attributor.attribute(memories=memories, answer=answer)
        if used:
            self.mark_used(owner=owner, memory_ids=used)
        return used

    # ------------------------------------------------------------------
    # 每轮轻量调用
    # ------------------------------------------------------------------
    def recall_tail(self, *, owner: str, query: str, cwd: Path | None = None) -> str:
        """每轮调用的轻量方法（走常驻进程，不重新加载模型）。"""
        ctx = self._recall.recall(
            owner=owner,
            query=query,
            cwd=str(cwd) if cwd is not None else None,
            now=utc_now(),
        )
        # ★ 记录「本轮模型实际看到了哪些记忆」= 常驻档 + 状态档 + 检索档。
        #   只记检索档是错的：常驻档**每轮都在上下文里**，
        #   它才是最可能被「用上」的那一批 —— 漏掉它会让 used_count
        #   对常驻条目永远是 0，于是 should_promote 永不触发、
        #   而 is_low_value 的 used==0 条件对它们恒真。
        self._last_recall[owner] = [*ctx.resident, *ctx.state, *ctx.retrieved]

        # ★ 补丁上限：超过阈值就先合并进常驻区再清空缓冲。
        #   「合并」是确定性的（在读取路径上按条数触发），不赌模型自觉。
        patches = self._store.pending_patches(owner)
        if len(patches) > self._patch_flush_threshold:
            self.flush_patches(owner=owner)
            patches = self._store.pending_patches(owner)
        return inject.render_tail(ctx, patches)

    def render_tail(self, *, owner: str, query: str) -> str:
        """``recall_tail`` 的别名 —— 规格里这个方法叫 ``render_tail``。"""
        return self.recall_tail(owner=owner, query=query)

    def flush_patches(self, *, owner: str) -> int:
        """把尾部补丁合并进常驻区后清空缓冲。

        ★ 这是「压缩时合并」的落点：规格第九节的铁律是
          「一切可变内容只允许出现在 prompt 的最后」，所以补丁不能一直堆着 ——
          它必须在某个确定性时刻被吸收进常驻档（即 ``CLAUDE.md`` 记忆区），
          然后 ``clear_patches``。没有这个入口，补丁会无限增长。
        """
        patches = self._store.pending_patches(owner)
        if not patches:
            return 0
        ctx = self._recall.recall(owner=owner, query="", now=self._stable_now())
        self._context_file.write(
            owner=owner,
            resident=inject.render_resident(ctx),
            state=inject.render_state(ctx),
        )
        self._store.clear_patches(owner)
        return len(patches)

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    @staticmethod
    def _stable_now() -> datetime:
        """常驻档专用时间源。

        ⚠️ 常驻渲染不允许出现时间戳，所以这个 ``now`` 只用于
           「可见性过滤 / 过期判断 / 预算排序」这些**不进文本**的计算。
           返回当前时刻即可 —— 但**不能**把它交给任何渲染函数。
        """
        return utc_now()

    @staticmethod
    def count_tokens(text: str) -> int:
        return _count_tokens(text)


def apply_tier_evolution(memories: list[Memory], cfg: TierConfig) -> list[tuple[str, str]]:
    """纯函数版升降级判定（不改库），返回 ``[(memory_id, new_tier), ...]``。

    放在这里是为了让「档位决策」能被单测穷举，不需要数据库。
    """
    changes: list[tuple[str, str]] = []
    for memory in memories:
        if policy.should_promote(memory, cfg):
            changes.append((memory.id, "resident"))
    return changes


__all__ = [
    "SessionLifecycle",
    "SessionEndReport",
    "NeedManualIntegration",
    "apply_tier_evolution",
    "Patch",
]
