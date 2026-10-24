"""L2 服务：写入管道。

职责：把「一个事件」变成「若干条落库的记忆」（设计文档第六节）。

不做什么：
  ❌ 不直接连数据库（走 ``store``）
  ❌ 不做清洗规则本身（调 ``domain/scrub``）
"""

from __future__ import annotations

import re
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime

from ..domain import policy, scrub
from ..impl import llm_extractor
from ..domain.models import (
    Memory,
    MemoryType,
    Patch,
    PolicyConfig,
    Priority,
    SourceKind,
    Tier,
    TierConfig,
    utc_now,
)
from ..ports import Embedder, MemoryStore

# 意图 → 基础重要性
_IMPORTANCE_BY_INTENT: dict[scrub.Intent, float] = {
    scrub.Intent.STATEMENT: 0.5,
    scrub.Intent.REPORTED: 0.35,
    scrub.Intent.IMPERATIVE: 0.4,  # 降级后重要性一并下调
    scrub.Intent.HYPOTHETIC: 0.0,
    scrub.Intent.RHETORICAL: 0.0,
}

# 关键词 → 记忆类型
# ⚠️ CONSTRAINT 只认**否定式 / 限定式**（禁止/不准/不得/只能…）。
#    把「必须 / 一定要」也算进来会让口头偏好（"你必须每次都输出 JSON"）
#    拿到常驻档权限 —— 那是把一句抱怨升级成系统规则。
_CONSTRAINT_RE = re.compile(
    r"(禁止|严禁|不准|不许|不得|不能|不可以|不要用|别用|只能|只准|仅能|仅可|硬约束)"
)
_PROJECT_MARKERS = ("当前", "现在", "正在用", "这个项目", "本仓库", "进行到", "下一步")
_PREFERENCE_MARKERS = ("我喜欢", "我不喜欢", "我偏好", "我习惯", "总是", "从不", "偏好")


def make_memory_id(now: datetime) -> str:
    """``m_<YYYYMMDD>_<HHMM>_<6位随机>``（架构文档第九节的形式）。"""
    stamp = now.strftime("%Y%m%d_%H%M")
    return f"m_{stamp}_{uuid.uuid4().hex[:6]}"


def classify_type(text: str) -> MemoryType:
    """按关键词判记忆类型（规则，不调模型）。

    ★ 也可以直接传 ``scrub.Intent``：``Intent.IMPERATIVE`` 的判定更宽
      （含「必须」），但**类型**必须更窄 —— 见 ``_CONSTRAINT_RE`` 的注释。
    """
    if _CONSTRAINT_RE.search(text):
        return MemoryType.CONSTRAINT
    if any(k in text for k in _PROJECT_MARKERS):
        return MemoryType.PROJECT
    if any(k in text for k in _PREFERENCE_MARKERS):
        return MemoryType.PREFERENCE
    return MemoryType.FACT


@dataclass(slots=True)
class WriteReport:
    """写入结果。领域失败用字段表达，**不靠抛异常**。"""

    written: list[Memory] = field(default_factory=list)
    superseded: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    degraded: list[str] = field(default_factory=list)
    need_clarification: list[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.written)


class WritePipeline:
    def __init__(
        self,
        store: MemoryStore,
        embedder: Embedder,
        policy_cfg: PolicyConfig,
        tier_cfg: TierConfig | None = None,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._policy_cfg = policy_cfg
        self._tier_cfg = tier_cfg or TierConfig()

    # ------------------------------------------------------------------
    # 事件 1-4：单条事件的实时写入
    # ------------------------------------------------------------------
    def ingest_event(
        self,
        *,
        owner: str,
        text: str,
        source: str,
        source_kind: SourceKind,
        now: datetime,
    ) -> list[Memory]:
        """单条事件的实时写入（对应「五个该写的事件」里的 1-4）。"""
        return self._ingest(
            owner=owner, text=text, source=source, source_kind=source_kind, now=now
        ).written

    def ingest_event_detailed(
        self,
        *,
        owner: str,
        text: str,
        source: str,
        source_kind: SourceKind,
        now: datetime,
    ) -> WriteReport:
        """与 ``ingest_event`` 同流程，但返回完整报告（供 CLI / 服务端展示）。"""
        return self._ingest(
            owner=owner, text=text, source=source, source_kind=source_kind, now=now
        )

    def _ingest(
        self,
        *,
        owner: str,
        text: str,
        source: str,
        source_kind: SourceKind,
        now: datetime,
    ) -> WriteReport:
        report = WriteReport()

        # ⓪ 防注入第一关 —— 必须在最前面：
        #    低信任来源不进记忆域，只进知识库
        if not policy.is_trusted_source(source_kind):
            report.skipped.append("untrusted-source")
            return report

        if not owner.strip():
            report.skipped.append("empty-owner")
            return report

        raw = text.strip()
        if not raw:
            report.skipped.append("empty-input")
            return report

        # ⚠️ 长度闸门必须在**规范化之前**：
        #    normalize() 自己会按 max_text_chars 截断，于是「超长输入」在
        #    should_remember 眼里变成了「一条 4000 字的普通陈述」被放行 ——
        #    落库 4000 字、而 embedding 只编码前 ~512 token，
        #    检索语义与落库内容分叉，之后每轮还会把它注入上下文。
        if len(raw) > self._policy_cfg.max_text_chars:
            report.skipped.append(f"too-long:{len(raw)}")
            return report

        # ① 意图分类 —— 虚拟 / 反问直接丢弃
        intent = scrub.classify_intent(raw)
        if intent in (scrub.Intent.HYPOTHETIC, scrub.Intent.RHETORICAL):
            report.skipped.append(f"intent:{intent.value}")
            return report

        # ② 规范化 + 相对时间绝对化
        normalized = scrub.absolutize_time(scrub.normalize(raw, cfg=self._policy_cfg), now)

        existing = self._store.active_memories(owner)

        # ③ 三问
        verdict = policy.should_remember(
            normalized, existing=existing, cfg=self._policy_cfg
        )
        if verdict is policy.Verdict.SKIP:
            report.skipped.append(normalized)
            return report
        if verdict is policy.Verdict.ASK:
            report.need_clarification.append(normalized)
            return report

        # ★ DEGRADE 必须【真的调用】scrub.degrade_imperative 改写文本，
        #   不能只是拿到 DEGRADE 就放行。
        #   两个「降级」的分工：Verdict 是「要不要降级」，scrub 是「怎么降级」。
        #
        #   ⚠️ **硬约束豁免降级**：`classify_type` 已经认出这是 CONSTRAINT 时，
        #      改写会同时破坏语义（"必须是 Python 3.12" → "用户曾提出要求：是 Python 3。"）
        #      和类型识别（改写后关键词消失，类型退化成 FACT，再也不进常驻档）。
        #      硬约束的正确处置是「进常驻档 + 由 render_resident 做措辞无害化」，
        #      而不是把用户明说的约束降级成一句转述。
        constrained = classify_type(normalized) is MemoryType.CONSTRAINT
        if verdict is policy.Verdict.DEGRADE or intent is scrub.Intent.IMPERATIVE:
            normalized = scrub.degrade_imperative(normalized)
            report.degraded.append(normalized)
            if constrained:
                # 降级后成了「用户约束：不得…」，类型要重判 —— 它在写入时
                # 已经脱离指令式措辞，所以这里应当回到 CONSTRAINT。
                assert classify_type(normalized) is MemoryType.CONSTRAINT, normalized

        # ④ 拆粒度
        fragments = scrub.split_granularity(normalized)
        if not fragments:
            report.skipped.append("empty-after-scrub")
            return report

        importance = _IMPORTANCE_BY_INTENT.get(intent, 0.5)
        # ★★ 只给创建者足重的权重。
        #   QQ 桥接会把每个群友的会话都归档，他们说的话会被提取成记忆。
        #   如果不压权重，别人的偏好/说法会跟主人的一起参与淘汰与排序，
        #   长此以往就被他们带跑。这里把非主人的记忆封顶在 0.3。
        #   注意：importance 只影响淘汰顺序，不影响可见性；
        #   真正的隔离靠 owner 过滤与每个用户独立的 AGENTS.md。
        _primary_owner = os.environ.get("MEMORY_BRIDGE_OWNER", "").strip()
        if _primary_owner and owner != _primary_owner:
            importance = min(importance, 0.3)
        return self._persist(
            owner=owner,
            fragments=fragments,
            source=source,
            source_kind=source_kind,
            now=now,
            importance=importance,
            confidence=0.6 if source_kind is SourceKind.USER_INPUT else 0.45,
            existing=existing,
            report=report,
            force_urgent=False,
        )

    # ------------------------------------------------------------------
    # 事件 5：会话结束的批处理
    # ------------------------------------------------------------------
    def ingest_session_end(
        self,
        *,
        owner: str,
        transcript: str,
        now: datetime,
        source: str = "session-end",
    ) -> list[Memory]:
        """会话结束的批处理（对应事件 5，兜底 + 抓隐性信息）。

        与 ``ingest_event`` 的区别：规格说这里**调模型**做一次性提取
        （「他最后决定用 B」这类结论只有会话结束才成型）。

        ⚠️ 取舍（已列入汇报的待确认项）：规格第〇节的依赖清单里**没有任何 LLM 客户端**
           （openai / anthropic 都不在），所以本实现用**规则抽取**代替模型提取：
          逐行过 ``ingest_event``，再额外抓「决定 / 采用 / 最终 / 结论」这类收尾句。
          接口与调用点保持不动，将来接入 LLM 只需替换这个方法体。
        """
        report = self._ingest_session_end(
            owner=owner, transcript=transcript, now=now, source=source
        )
        return report.written

    def ingest_session_end_detailed(
        self,
        *,
        owner: str,
        transcript: str,
        now: datetime,
        source: str = "session-end",
    ) -> WriteReport:
        """``ingest_session_end`` 的报告版（``bridge/session.py`` 需要 ``need_manual``
        与 ``skipped`` 这些信息来组装 ``SessionEndReport``）。"""
        return self._ingest_session_end(
            owner=owner, transcript=transcript, now=now, source=source
        )

    def _ingest_session_end(
        self, *, owner: str, transcript: str, now: datetime, source: str
    ) -> WriteReport:
        report = WriteReport()
        if not transcript.strip():
            report.skipped.append("empty-transcript")
            return report

        # ★ 优先走 LLM 提炼。
        #   规则抽取（下面那段兜底）会把对话原文切片当事实存 —— 实测抽出来的
        #   长这样："assistant： 喵？ 你搁这学猫叫呢 怎么，有事说事"，
        #   纯粹是机器人自己的台词，召回时只会污染上下文。
        #   LLM 返回 None（没配 key / 调用失败）才回退到规则抽取。
        facts = llm_extractor.extract(transcript)
        if facts is not None:
            if not facts:
                report.skipped.append("no-durable-fact")
                return report
            # ★ 逐条入库，不要 join 成一段。
            #   实测：拼成一段再走 split_granularity，会被粘回一条大记忆
            #   （"…五张手牌 …克制只在速度快的一方生效 …从直觉开始"），
            #   检索时这三件事会互相稀释。原子化才检索得动。
            merged = WriteReport()
            for fact in facts:
                try:
                    sub = self._ingest(
                        owner=owner,
                        text=fact,
                        source=source,
                        source_kind=SourceKind.TOOL_RESULT,
                        now=now,
                    )
                except Exception:
                    merged.skipped.append(f"ingest-failed:{fact[:30]}")
                    continue
                merged.written.extend(sub.written)
                merged.superseded.extend(sub.superseded)
                merged.skipped.extend(sub.skipped)
                merged.degraded.extend(sub.degraded)
                merged.need_clarification.extend(sub.need_clarification)
            return merged

        # ── 以下为无 LLM 时的规则抽取兜底（保持原样）────────────────────
        decision_markers = (
            "决定",
            "采用",
            "最终",
            "结论",
            "确定用",
            "改成",
            "换成",
            "settled on",
            "decided",
        )

        candidates: list[str] = []
        for line in transcript.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if any(k in stripped for k in decision_markers) or len(stripped) <= 120:
                candidates.append(stripped)

        if not candidates:
            candidates = [transcript[:800]]

        # 合并成整段走一次管道，避免逐行重复全量拉取 active_memories
        merged = "\n".join(candidates[:200])
        return self._ingest(
            owner=owner,
            text=merged,
            source=source,
            source_kind=SourceKind.TOOL_RESULT,
            now=now,
        )

    # ------------------------------------------------------------------
    # 显式命令
    # ------------------------------------------------------------------
    def remember_explicit(self, *, owner: str, text: str, now: datetime) -> Memory:
        """用户显式命令（``memory remember "..."``）。

        ``importance`` 拉满、``priority=URGENT``、``source_kind=USER_INPUT``、
        **跳过 ``should_remember``**。

        ⚠️ 已知风险（列入汇报）：这条路径跳过「三问」与意图降级，
           所以 agent 若被网页内容诱导调用 ``memory remember``，
           注入的文本会以高优 + 常驻档落库。SKILL.md 已明写禁止，
           但代码侧目前只用 ``source_kind`` 做信任判断 —— 见汇报的待确认项。
        """
        normalized = scrub.absolutize_time(
            scrub.normalize(text, cfg=self._policy_cfg), now
        )
        if not normalized:
            raise ValueError("remember_explicit: 文本为空")

        fragments = scrub.split_granularity(normalized)
        existing = self._store.active_memories(owner)

        # ★ 幂等：同一 owner 下已有**完全相同**的 active 记忆时直接返回它。
        #   规格说这条路径「跳过 should_remember」，但库上有一条部分唯一索引
        #   （memories_owner_what_active_uniq）挡着完全相同的文本 ——
        #   不先查一下的话，重复 remember 会把 IntegrityError 抛到用户面前
        #   （实测踩过：DuplicateKey 直接冒到 CLI）。
        #   语义上「记住同一件事两次」本来就该是幂等的，返回已有条目即可。
        duplicate = next(
            (
                m
                for m in existing
                if m.what == fragments[0] and len(fragments) == 1
            ),
            None,
        )
        if duplicate is not None:
            return duplicate

        report = WriteReport()
        self._persist(
            owner=owner,
            fragments=fragments,
            source="cli:remember",
            source_kind=SourceKind.USER_INPUT,
            now=now,
            importance=1.0,
            confidence=1.0,
            existing=existing,
            report=report,
            force_urgent=True,
        )
        if not report.written:
            raise RuntimeError("remember_explicit: 未能写入任何记忆")
        return report.written[0]

    # ------------------------------------------------------------------
    # 公共落库路径
    # ------------------------------------------------------------------
    def _persist(
        self,
        *,
        owner: str,
        fragments: list[str],
        source: str,
        source_kind: SourceKind,
        now: datetime,
        importance: float,
        confidence: float,
        existing: list[Memory],
        report: WriteReport,
        force_urgent: bool,
    ) -> WriteReport:
        # ⑤ 先算冲突，再一次性编码（把 embedding 调用压到最少次数）
        plans: list[tuple[str, MemoryType, str | None]] = []
        for fragment in fragments:
            mtype = classify_type(fragment)
            probe = Memory(
                id="",
                owner=owner,
                what=fragment,
                type=mtype,
                when=now,
                confidence=confidence,
            )
            old_id = self._find_conflict(probe, existing)
            plans.append((fragment, mtype, old_id))

        texts = [fragment for fragment, _, _ in plans]
        vectors = self._embedder.encode(texts) if texts else []

        for (fragment, mtype, old_id), vector in zip(plans, vectors, strict=True):
            memory = Memory(
                id=make_memory_id(now),
                owner=owner,
                what=fragment,
                type=mtype,
                when=now,
                scope="global",
                source=source,
                source_kind=source_kind,
                confidence=confidence,
                importance=importance,
                embedding=vector,
            )
            # ⑥ 初始档位（不是留空靠默认值）
            memory.tier = policy.assign_tier(memory, self._tier_cfg)

            # ⑦ 落库：save_memory + supersede 在【同一个事务】里完成
            #    —— 见 impl/pg_store.py 的实现（store.supersede_in 由 save 内部触发）
            self._store.save_memory(memory)
            if old_id is not None:
                self._store.supersede(old_id, memory.id)
                report.superseded.append(old_id)
                # supersede 之后新条目才真正成为「当前事实」，
                # 让后续 fragment 的冲突检测也看见它
                existing = [
                    m for m in existing if m.id != old_id
                ] + [memory]
            else:
                existing = [*existing, memory]

            report.written.append(memory)

            # ★ 进补丁缓冲 → 下一轮 render_tail() 就包含它（当轮生效、零缓存代价）
            self._store.append_patch(
                Patch(
                    id=0,
                    owner=owner,
                    text=memory.what,
                    priority=Priority.URGENT if force_urgent else Priority.NORMAL,
                    at=now,
                )
            )

        return report

    def _find_conflict(self, probe: Memory, existing: list[Memory]) -> str | None:
        """找出被新条目取代的旧条目 id。

        规则（``policy.resolve_conflict``）：
          同 type + 同 scope + 语义相近 → supersede
          不同 scope / 不同 type       → coexist
          新条目置信度明显更低          → reject（即不覆盖，旧条目留着）
        """
        best_id: str | None = None
        best_overlap = 0.0
        for old in existing:
            if old.tier is Tier.RESIDENT and old.type is MemoryType.CONSTRAINT:
                # 硬约束不因为一句语义相近的话就被覆盖
                continue
            decision = policy.resolve_conflict(old, probe, cfg=self._policy_cfg)
            if decision == "supersede":
                overlap = policy.text_overlap(old.what, probe.what)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_id = old.id
        return best_id

    @staticmethod
    def now() -> datetime:
        return utc_now()
