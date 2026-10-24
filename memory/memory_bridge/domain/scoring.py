"""L1 打分与淘汰：全部数值计算。

**纯函数，无副作用。**

不做什么：
  ❌ 不做时间 IO（``now`` 由调用方传，保证可测）
  ❌ 不决定删不删（只给分，决策归 ``service/forget.py``）
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from .models import Memory, MemoryType, ScoringConfig, Tier


def decay_factor(
    m: Memory,
    now: datetime,
    half_life_days: float,
    *,
    include_access: bool = True,
) -> float:
    """``0.5 ** (age / half_life)``，age 从 ``max(last_access, when)`` 算起。

    ⚠️ **常驻档不衰减**（``tier == RESIDENT`` → 直接返回 1.0）。
       理由：常驻意味着「每轮都在用」，它没有「久未访问」这个概念；
       用 ``last_access`` 去衰减一条每轮都注入的记忆，是逻辑错误。

       衰减只作用于【检索档】；状态缓存档的衰减由 ``expires_at`` 管。

    ⚠️ ``include_access=False`` 时只看 ``when``（写入时刻）。
       为什么需要这个开关：``store.touch()`` 会同时 bump ``hit_count`` **和**
       ``last_access``，于是「召回越多 → 衰减基准越新 → 分数越高」，
       把「召回 10 次一次没用上」的条目越养越肥。
       ``service/forget.py`` 的淘汰排序用 ``include_access=False`` 来切断这个正反馈。
    """
    if m.tier is Tier.RESIDENT:
        return 1.0
    if half_life_days <= 0 or math.isinf(half_life_days):
        return 1.0

    current = _as_utc(now)
    base = _as_utc(m.when)
    if include_access and m.last_access is not None:
        base = max(base, _as_utc(m.last_access))

    age_days = (current - base).total_seconds() / 86400.0
    if age_days <= 0:
        return 1.0
    return float(0.5 ** (age_days / half_life_days))


def eviction_score(
    m: Memory,
    now: datetime,
    cfg: ScoringConfig,
    *,
    include_access: bool = True,
) -> float:
    """三因子：衰减 × 重要性 × ``log1p(命中频次)``
    —— 命中频次用 log 压，防高频条目永久霸榜。"""
    decay = decay_factor(
        m, now, half_life_for(m.type), include_access=include_access
    )
    return decay * m.importance * math.log1p(1 + m.hit_count)


def is_low_value(
    m: Memory,
    cfg: ScoringConfig,
    *,
    now: datetime | None = None,
) -> bool:
    """「召回 10 次用 0 次」的判定 —— 淘汰先杀这类（占着 topk 没价值）。

    ⚠️ 判据是 ``hit_count`` 与 ``used_count`` 的**比值关系**，不是单看 ``hit_count``：
       ``used_count`` 由 ``bridge/session.py`` 的 ``mark_used()`` 写入。
       如果 ``used_count`` 因为没有任何调用入口而恒为 0，
       这条规则会退化成「**优先淘汰最常被召回的记忆**」—— 那是灾难性的。
       所以这里额外要求「确实被召回过足够多次」（hit >= 阈值）。
    """
    del now  # 判据不依赖当前时间；保留参数是为了调用方签名一致
    return (
        m.hit_count >= cfg.low_value_min_hits
        and m.used_count <= cfg.low_value_max_used
    )


def half_life_for(mtype: MemoryType) -> float:
    """**只在 ``tier == RETRIEVED`` 时使用**（常驻档在 ``decay_factor`` 里已短路）。

    按类型分档：constraint 不衰减 / preference 数月 / project_state 数周。

    ⚠️ 这里的 ``type`` 与 ``tier`` 是两件事：
       - ``tier`` 决定「要不要衰减」（常驻 → 不衰减）
       - ``type`` 决定「衰减多快」（检索档内部再分）
    """
    cfg = ScoringConfig()
    if mtype is MemoryType.CONSTRAINT:
        return cfg.half_life_constraint
    if mtype is MemoryType.PREFERENCE:
        return cfg.half_life_preference
    if mtype is MemoryType.PROJECT:
        return cfg.half_life_project
    return cfg.half_life_fact


def _as_utc(value: datetime) -> datetime:
    """naive datetime 一律按 UTC 解释（与 ``policy._as_utc`` 同一约定）。"""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
