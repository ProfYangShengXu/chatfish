"""L2 服务：淘汰与归档（容量治理，设计文档第十节）。

不做什么：
  ❌ 不自动删（默认只归档）
  ❌ 不在读取路径上触发淘汰（离线跑）
  ❌ **不碰 ``tier``**（档位调整归 ``policy``）——
     但本实现里 ``should_promote`` / ``should_demote`` 需要一个**离线触发点**，
     而规格把「顺带做一次容量治理」放在 ``session.on_session_end``；
     所以档位自适应被显式拆到 ``apply_tier_evolution()``，
     与淘汰逻辑分开、可单独调用，避免「forget 偷偷改档位」。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from ..domain import policy, scoring
from ..domain.clock import to_naive_utc
from ..domain.models import (
    ForgetConfig,
    ScoringConfig,
    TierConfig,
)
from ..ports import MemoryStore


@dataclass(slots=True)
class ForgetReport:
    """一次容量治理的结果。领域失败用字段表达，不抛异常。"""

    scanned: int = 0
    archived: list[str] = field(default_factory=list)
    promoted: list[str] = field(default_factory=list)
    demoted: list[str] = field(default_factory=list)
    remaining: int = 0
    need_manual: bool = False
    reason: str = ""

    @property
    def archived_count(self) -> int:
        return len(self.archived)


class ForgetPipeline:
    def __init__(
        self,
        store: MemoryStore,
        cfg: ForgetConfig,
        scoring_cfg: ScoringConfig | None = None,
        tier_cfg: TierConfig | None = None,
    ) -> None:
        self._store = store
        self._cfg = cfg
        self._scoring_cfg = scoring_cfg or ScoringConfig()
        self._tier_cfg = tier_cfg or TierConfig()

    def run(self, *, owner: str, now: datetime) -> ForgetReport:
        """分层触发：

        ① 若利用率 < 阈值 → 什么都不做
        ② 否则先淘汰 ``is_low_value`` 的（召回 10 次用 0 次的）
        ③ 仍超限 → 返回 ``ForgetReport(need_manual=True)``
                   由上层【报错】要求模型当场整合或删除（吸收自 Hermes）
        """
        current = to_naive_utc(now)
        report = ForgetReport()

        active = self._store.active_memories(owner)
        report.scanned = len(active)
        report.remaining = len(active)

        utilization = len(active) / max(self._cfg.max_active, 1)
        if utilization < self._cfg.utilization_threshold:
            report.reason = (
                f"利用率 {utilization:.2f} < 阈值 {self._cfg.utilization_threshold:.2f}，"
                "什么都不做"
            )
            return report

        # ② 先杀「占着 topk 没价值」的
        low_value = [m for m in active if scoring.is_low_value(m, self._scoring_cfg)]
        # 排序：淘汰分越低越先杀。
        # ⚠️ include_access=False —— touch() 会同时刷新 last_access，
        #    用它当衰减基准会让「召回越多越难淘汰」，正好和这条规则的目标相反。
        low_value.sort(
            key=lambda m: scoring.eviction_score(
                m, current, self._scoring_cfg, include_access=False
            )
        )

        victims = low_value[: self._cfg.batch]
        if victims:
            archived = self._store.archive_memories([m.id for m in victims])
            report.archived = [m.id for m in victims][:archived]

        remaining = len(active) - len(report.archived)
        report.remaining = remaining

        # ③ 仍超限 → 交给上层逼模型整合
        if remaining > self._cfg.max_active:
            report.need_manual = True
            report.reason = (
                f"归档 {len(report.archived)} 条后仍有 {remaining} 条，"
                f"超过上限 {self._cfg.max_active}；需要当场整合或删除"
            )
        else:
            report.reason = f"归档 {len(report.archived)} 条，剩余 {remaining} 条"

        return report

    def apply_tier_evolution(self, *, owner: str, now: datetime) -> ForgetReport:
        """自适应升降级（``policy.should_promote`` / ``should_demote``）。

        ``rounds_unused`` 的算法：规格的 ``should_demote`` 要一个「连续多少轮没被用到」，
        但那个计数**在数据模型里没有字段承载**（规格第〇节 DDL 只列了
        ``used_count`` 与 ``tier``）。这里的取舍是：
          用 ``now - last_access`` 折算成「轮次」的近似（每轮按 1 天算），
          没有 ``last_access`` 的常驻条目从 ``when`` 起算。
        这是本实现里第二弱的约定，已列入汇报的待确认项。
        """
        current = to_naive_utc(now)
        report = ForgetReport()
        active = self._store.active_memories(owner)
        report.scanned = len(active)

        for m in active:
            if policy.should_promote(m, self._tier_cfg):
                self._store.update_tier(m.id, "resident")
                report.promoted.append(m.id)
                continue

            if m.tier.value == "resident":
                anchor = m.last_access or m.when
                rounds_unused = int(
                    max(0.0, (current - to_naive_utc(anchor)).total_seconds() / 86400.0)
                )
                if policy.should_demote(m, rounds_unused, self._tier_cfg):
                    self._store.update_tier(m.id, "retrieved")
                    report.demoted.append(m.id)

        report.remaining = len(active) - len(report.promoted) - len(report.demoted)
        report.reason = (
            f"升档 {len(report.promoted)} 条，降档 {len(report.demoted)} 条"
        )
        return report

    def archive(self, memory_ids: list[str]) -> None:
        """归档而非删除（可回溯 + 合规）。"""
        if memory_ids:
            self._store.archive_memories(memory_ids)

    def purge(self, owner: str, *, older_than: datetime | None = None) -> int:
        """真删除（用户行使删除权时调用）。返回删除条数。

        ``older_than`` 给定时只删 ``when`` 早于它的条目。
        """
        active = self._store.active_memories(owner)
        cutoff = to_naive_utc(older_than) if older_than is not None else None

        targets = [
            m
            for m in active
            if cutoff is None or to_naive_utc(m.when) < cutoff
        ]
        for m in targets:
            self._store.remove_memory(m.id)
        return len(targets)
