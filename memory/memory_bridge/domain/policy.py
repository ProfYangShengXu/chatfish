"""L1 规则：全项目规则最密集的文件。

三个阶段的「该不该」判断全部收拢在这里，**不散落成 if**：

| 阶段     | 管什么                       | 规则                                             |
|----------|------------------------------|--------------------------------------------------|
| 写入时   | 该不该记、冲突怎么办         | ``should_remember`` / ``resolve_conflict``       |
| 注入时   | 进哪一档、可见吗、超预算怎么办 | ``assign_tier`` / ``visible_to`` / ``rank_for_resident`` |
| 演进时   | 升档还是降档                 | ``should_promote`` / ``should_demote``           |

不做什么：
  ❌ 不做清洗（归 ``scrub.py``）
  ❌ 不做打分（归 ``scoring.py``）
  ❌ **不查数据库**（``existing`` / ``owner`` 由调用方传入）
  ❌ **不在写入时定死档位**（只给初始值，靠升降级修正）
"""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import datetime, timezone
from enum import Enum

from .models import Memory, MemoryType, SourceKind, Tier, TierConfig, PolicyConfig


class Verdict(str, Enum):
    SAVE = "save"  # 该记
    SKIP = "skip"  # 不该记（事件流水 / 过程对话）
    DEGRADE = "degrade"  # 记，但要降级（指令式表述 / 低信任来源）
    ASK = "ask"  # 自相矛盾，需要用户澄清


# --------------------------------------------------------------------------
# 写入阶段
# --------------------------------------------------------------------------

# 事件流水 / 过程对话的信号词 —— 这些「下次新会话不需要」
_EPHEMERAL_MARKERS: tuple[str, ...] = (
    "刚刚运行",
    "这次运行",
    "现在开始",
    "接下来我",
    "然后我",
    "正在",
    "目前已运行",
    "本轮的",
)

# 自相矛盾信号：用户在同一句里既肯定又否定同一件事。
# ⚠️ 只放**真的表示犹豫/反复**的措辞。
#    「一方面 / 另一方面」看着像，其实常用于**罗列两个并列事实**
#    （「一方面要兼容旧格式，另一方面要支持新格式」）——
#    把它当矛盾会让一整类正常陈述落进 ASK 被丢掉。
_CONTRADICTION_MARKERS: tuple[str, ...] = (
    "我改主意",
    "说不准",
    "不确定要不要",
    "到底用哪个",
    "又不想",
    "可是又反悔",
)

# 稳定事实 / 偏好信号 —— 有这些就倾向记
_DURABLE_MARKERS: tuple[str, ...] = (
    "我喜欢",
    "我不喜欢",
    "我偏好",
    "我习惯",
    "我总是",
    "我从不",
    "我是",
    "我用",
    "我的",
    "记住",
    "别忘",
    "以后都",
    "统一用",
    "禁止",
    "必须",
    "不要用",
    "决定",
    "采用",
    "方案",
)


def _token_overlap(a: str, b: str) -> float:
    """字符 bigram 的 Jaccard 相似度 —— 纯函数，不需要 embedding。

    为什么不用 embedding：``should_remember`` 是 domain 层纯函数，
    签名里拿不到向量（规格第九节禁止 domain 依赖任何外部库）。
    代价是「换一种说法表达同一个事实」判不出来，
    这种更强的判重由 ``impl/pg_store.py`` 的向量近邻在写入路径上补。
    """
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    ga = {a[i : i + 2] for i in range(max(len(a) - 1, 1))}
    gb = {b[i : i + 2] for i in range(max(len(b) - 1, 1))}
    if not ga or not gb:
        return 0.0
    inter = len(ga & gb)
    union = len(ga | gb)
    return inter / union if union else 0.0


def text_overlap(a: str, b: str) -> float:
    """公开的文本相近度（供 ``service/write.py`` 做冲突排序用）。

    与 ``_token_overlap`` 是同一个函数 —— 加这个别名是为了不让别的层
    去戳下划线开头的私有名字。
    """
    return _token_overlap(a, b)


def is_duplicate(text: str, existing: list[Memory], *, threshold: float) -> bool:
    """文本层面的重复判定（③ 是否重复）。"""
    return any(_token_overlap(text, m.what) >= threshold for m in existing)


def should_remember(
    text: str,
    *,
    existing: list[Memory],
    cfg: PolicyConfig | None = None,
) -> Verdict:
    """三问（架构文档第六节）：
    ① 下次新会话还需要吗   ② 事实/偏好 vs 事件流水   ③ 是否重复
    """
    conf = cfg or PolicyConfig()
    stripped = text.strip()
    if not stripped:
        return Verdict.SKIP

    # ① + ②：事件流水 / 过程对话 → 不记
    if any(marker in stripped for marker in _EPHEMERAL_MARKERS):
        # 但带「记住 / 别忘 / 以后都」这类明确持久化意图的除外
        if not any(k in stripped for k in ("记住", "别忘", "以后都", "统一用")):
            return Verdict.SKIP

    # 自相矛盾 → 需要用户澄清
    if any(marker in stripped for marker in _CONTRADICTION_MARKERS):
        return Verdict.ASK

    # ③ 重复 → 不记（已有等价条目）
    if is_duplicate(stripped, existing, threshold=conf.duplicate_overlap):
        return Verdict.SKIP

    # 指令式表述 → 记，但要降级（信息保留，权限剥离）
    imperative_markers = ("必须", "一定要", "以后每次", "从现在起", "禁止", "不准")
    if any(k in stripped for k in imperative_markers):
        return Verdict.DEGRADE

    # 有稳定事实 / 偏好信号 → 记
    if any(k in stripped for k in _DURABLE_MARKERS):
        return Verdict.SAVE

    # 默认：短句陈述句视为事实，长句多半是过程描述
    if len(stripped) <= 200:
        return Verdict.SAVE
    return Verdict.SKIP


def resolve_conflict(
    old: Memory,
    new: Memory,
    *,
    cfg: PolicyConfig | None = None,
) -> str:
    """返回 ``"supersede"`` | ``"coexist"`` | ``"reject"``（架构文档第九节）：

    - 同 type + 同 scope + 语义相近 → supersede
    - 不同 scope                    → coexist
    - 新的置信度明显更低            → reject
    """
    conf = cfg or PolicyConfig()

    # 新的置信度明显更低 → 拒绝覆盖（防止弱推断盖掉用户明确说的事实）
    if old.confidence - new.confidence > conf.confidence_reject_margin:
        return "reject"

    # 不同 scope → 各自独立存在
    if old.scope != new.scope:
        return "coexist"

    if old.type != new.type:
        return "coexist"

    # 同 type + 同 scope：语义相近才覆盖
    if _token_overlap(old.what, new.what) >= 0.35:
        return "supersede"
    return "coexist"


def classify_staleness(
    m: Memory,
    now: datetime,
    *,
    cfg: PolicyConfig | None = None,
) -> str:
    """返回 ``"fresh"`` | ``"stale"`` | ``"expired"``（供淘汰与拒答用）。"""
    conf = cfg or PolicyConfig()
    current = _as_utc(now)

    if m.expires_at is not None and _as_utc(m.expires_at) <= current:
        return "expired"

    age_days = (current - _as_utc(m.when)).total_seconds() / 86400.0
    if age_days > conf.stale_after_days:
        return "stale"
    return "fresh"


def is_trusted_source(kind: SourceKind) -> bool:
    """来源信任级（防注入，架构文档第十节）：

    - ``USER_INPUT``  → True
    - ``TOOL_RESULT`` → True，但写入时必须记 ``source``
    - ``EXTERNAL``    → **False** —— 不进记忆域，只进知识库
    """
    return kind is not SourceKind.EXTERNAL


# --------------------------------------------------------------------------
# 注入阶段
# --------------------------------------------------------------------------


def assign_tier(m: Memory, cfg: TierConfig) -> Tier:
    """初始分档（**只在写入时调用一次**）。

    ① ``type == CONSTRAINT``  → RESIDENT（且永不淘汰）
    ② ``type == PROJECT``     → STATE
    ③ ``importance >= 0.9``   → RESIDENT（用户明确说「记住这个」）
    ④ 其余                    → RETRIEVED

    ⚠️ 这只是【初始猜测】—— 写完时你根本不知道哪条会常用。
       真实档位由 ``should_promote`` / ``should_demote`` 自适应修正。
    """
    if m.type is MemoryType.CONSTRAINT:
        return Tier.RESIDENT
    if m.type is MemoryType.PROJECT:
        return Tier.STATE
    if m.importance >= cfg.resident_importance:
        return Tier.RESIDENT
    return Tier.RETRIEVED


def _cwd_matches_project(cwd: str | None, scope: str) -> bool:
    """``scope == "project:<name>"`` 时，cwd 是否属于该项目。

    ★ 规格只说「仅当 cwd 属于该项目」，没给判定规则。这里的取舍是：
      ``project:<name>`` 的 ``<name>`` 与 cwd 的**最后一段目录名**做不区分大小写的相等比较
      （或在 cwd 的任一路径段中出现）。
      这是本实现里最弱的一条约定，已在汇报中列为待确认项。
    """
    if cwd is None:
        return False
    name = scope.split(":", 1)[1].strip().lower()
    if not name:
        return False
    normalized = cwd.replace("\\", "/").rstrip("/").lower()
    segments = [seg for seg in normalized.split("/") if seg]
    return bool(segments) and segments[-1] == name or name in segments


def visible_to(
    m: Memory,
    *,
    owner: str,
    scope: str | None,
    cwd: str | None,
    now: datetime | None = None,
) -> bool:
    """可见性过滤（**注入前必须先过这一关**）。

    - ``m.owner != owner``         → False（★ 安全边界，见下）
    - ``m.scope == "global"``      → True
    - ``m.scope == f"project:{name}"`` → 仅当 cwd 属于该项目
    - 已 superseded / 已过期       → False

    ★ owner 比对是**安全边界**，必须在 SQL 的 ``WHERE`` 里也做一遍
      （见 ``impl/pg_store.py``）；这里再判一次是纵深防御，不是唯一防线。
    """
    if m.owner != owner:
        return False
    if m.superseded_by is not None:
        return False

    current = _as_utc(now) if now is not None else datetime.now(timezone.utc)
    if m.expires_at is not None and _as_utc(m.expires_at) <= current:
        return False

    if m.scope == "global":
        return True

    if m.scope.startswith("project:"):
        return _cwd_matches_project(cwd, m.scope)

    # 其它 scope（如 "session:xxx"）默认只在显式请求同名 scope 时可见
    return scope is not None and scope == m.scope


def rank_for_resident(
    memories: list[Memory],
    cfg: TierConfig,
    count_tokens: Callable[[str], int],
) -> tuple[list[Memory], list[Memory]]:
    """常驻档预算控制（**防止退化成「全部常驻」**）。

    **预算单位是 token，不是条数**（架构文档第七节：≈800 token）——
    因为一条 5 字的约束和一条 80 字的偏好，占的上下文差 16 倍。

    做法：按 ``importance × log1p(1 + hit_count)`` 降序累加，
          累计 token 超过 ``cfg.resident_token_budget`` 即停止；
    返回 (进常驻的, 退回检索档的)。

    ⚠️ 退回是【降级】不是【删除】—— 记忆还在检索档里。
    ⚠️ ``count_tokens`` 由调用方注入（避免 domain 层依赖具体分词器）。
    ⚠️ 规格写的排序键是 ``importance × log1p(hit_count)``；本实现改成
       ``log1p(1 + hit_count)``。原因：``hit_count == 0`` 时 ``log1p(0) == 0``，
       会让**所有新条目的 importance 完全失效**（分数恒为 0，排序退化成 DB 返回序），
       而「新写入 + importance 拉满的显式记忆」正是常驻档最主要的来源。
       取 ``1 + hit_count`` 后：0 次命中 → 系数 log(2)≈0.69，10 次命中 → 系数 log(11)≈2.4，
       单调性不变、importance 重新有效。已列入汇报的待确认项。
    """
    scored = sorted(
        memories,
        key=lambda m: (m.importance * math.log1p(1 + m.hit_count), m.id),
        reverse=True,
    )

    resident: list[Memory] = []
    demoted: list[Memory] = []
    used_tokens = 0
    for m in scored:
        cost = count_tokens(m.what)
        if used_tokens + cost <= cfg.resident_token_budget:
            resident.append(m)
            used_tokens += cost
        else:
            demoted.append(m)
    return resident, demoted


# --------------------------------------------------------------------------
# 演进阶段（自适应升降级）
# --------------------------------------------------------------------------


def should_promote(m: Memory, cfg: TierConfig) -> bool:
    """升档：被召回【且被实际用上】累计 >= ``promote_threshold`` → RESIDENT。"""
    if m.tier is Tier.RESIDENT:
        return False
    if m.type is MemoryType.PROJECT:
        # 项目状态归 STATE 档，不往常驻档升
        return False
    return m.used_count >= cfg.promote_threshold


def should_demote(m: Memory, rounds_unused: int, cfg: TierConfig) -> bool:
    """降档：当前是 RESIDENT，但连续 ``rounds_unused`` 轮没被用到 → RETRIEVED。

    ``rounds_unused`` 由调用方（``service/forget.py``）用
    ``now - last_access``（换算成轮次的近似）算出来传入 —— domain 不碰时间 IO。
    """
    if m.tier is not Tier.RESIDENT:
        return False
    if m.type is MemoryType.CONSTRAINT:
        # 硬约束永不降档
        return False
    return rounds_unused >= cfg.demote_rounds


def _as_utc(value: datetime) -> datetime:
    """naive datetime 一律按 UTC 解释，aware 的统一换到 UTC。

    统一时区是必需的：PG 的 TIMESTAMPTZ 读回来是 aware，
    而单测里很容易造出 naive 的 now，两者相减会抛 TypeError。
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
