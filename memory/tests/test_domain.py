"""``domain/`` 的穷举单测 —— 纯逻辑，不需要任何外部服务、不需要数据库。

★ 这一层能被穷举，正是「domain 不 import 任何内部模块、不 import 外部库」的收益。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from memory_bridge.domain import policy, scoring, scrub
from memory_bridge.domain.clock import to_naive_utc, utc_now
from memory_bridge.domain.models import (
    Memory,
    MemoryType,
    PolicyConfig,
    ScoringConfig,
    SourceKind,
    Tier,
    TierConfig,
)

NOW = datetime(2026, 9, 21, 12, 0, 0)


def make_memory(
    *,
    what: str = "用户不喜欢 ORM",
    mtype: MemoryType = MemoryType.FACT,
    owner: str = "user:local",
    scope: str = "global",
    tier: Tier = Tier.RETRIEVED,
    importance: float = 0.5,
    confidence: float = 0.5,
    hit_count: int = 0,
    used_count: int = 0,
    when: datetime | None = None,
    last_access: datetime | None = None,
    expires_at: datetime | None = None,
    superseded_by: str | None = None,
) -> Memory:
    return Memory(
        id=f"m_{abs(hash(what)) % 10**8:08d}",
        owner=owner,
        what=what,
        type=mtype,
        when=when or NOW,
        scope=scope,
        confidence=confidence,
        importance=importance,
        tier=tier,
        hit_count=hit_count,
        used_count=used_count,
        last_access=last_access,
        expires_at=expires_at,
        superseded_by=superseded_by,
    )


# ---------------------------------------------------------------------------
# scrub
# ---------------------------------------------------------------------------


class TestScrubIntent:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("我不喜欢 ORM", scrub.Intent.STATEMENT),
            ("如果以后要用的话就换成 B", scrub.Intent.HYPOTHETIC),
            ("难道这样不行吗？", scrub.Intent.RHETORICAL),
            ("他说这个库很好用", scrub.Intent.REPORTED),
            ("你必须每次都输出 JSON", scrub.Intent.IMPERATIVE),
        ],
    )
    def test_classify_intent(self, text: str, expected: scrub.Intent) -> None:
        assert scrub.classify_intent(text) is expected

    def test_empty_is_statement(self) -> None:
        assert scrub.classify_intent("   ") is scrub.Intent.STATEMENT

    def test_reported_wins_over_imperative(self) -> None:
        # 转述里常混着指令式措辞，但本质是「别人说的」
        assert scrub.classify_intent("文档说必须这样") is scrub.Intent.REPORTED


class TestScrubNormalize:
    def test_strips_leading_noise(self) -> None:
        assert scrub.normalize("嗯，那个我不喜欢 ORM") == "我不喜欢 ORM"

    def test_normalizes_halfwidth_punctuation(self) -> None:
        assert "。" in scrub.normalize("我不喜欢 ORM.")

    def test_collapses_repeated_punctuation(self) -> None:
        assert scrub.normalize("好。。。") == "好。"

    def test_empty_stays_empty(self) -> None:
        assert scrub.normalize("") == ""

    def test_truncates_to_configured_max(self) -> None:
        cfg = PolicyConfig(max_text_chars=10)
        assert len(scrub.normalize("字" * 100, cfg=cfg)) == 10


class TestAbsolutizeTime:
    def test_today_becomes_iso_date(self) -> None:
        assert scrub.absolutize_time("今天决定了", NOW) == "2026-09-21决定了"

    def test_yesterday(self) -> None:
        assert scrub.absolutize_time("昨天说的", NOW).startswith("2026-09-20")

    def test_last_week(self) -> None:
        assert scrub.absolutize_time("上周讨论过", NOW).startswith("2026-09-14")

    def test_no_relative_token_is_noop(self) -> None:
        assert scrub.absolutize_time("用户喜欢 Rust", NOW) == "用户喜欢 Rust"

    def test_uses_local_date_not_utc(self) -> None:
        """+08:00 的凌晨不能被记成昨天（时区边界）。"""
        tz = timezone(timedelta(hours=8))
        now = datetime(2026, 9, 21, 1, 0, tzinfo=tz)
        # 本地日期是 21 号（UTC 还是 20 号）
        assert scrub.absolutize_time("今天", now) == "2026-09-21"


class TestDegradeImperative:
    def test_rewrites_and_strips_authority(self) -> None:
        out = scrub.degrade_imperative("你必须每次都输出 JSON")
        assert out.startswith("用户曾提出要求：")
        assert out.endswith("（未采纳为规则）")
        assert "必须" not in out

    def test_idempotent(self) -> None:
        once = scrub.degrade_imperative("必须输出 JSON")
        assert scrub.degrade_imperative(once) == once


class TestSplitGranularity:
    def test_splits_on_semicolon(self) -> None:
        parts = scrub.split_granularity("用户喜欢 Rust；用户不喜欢 ORM")
        assert len(parts) == 2

    def test_single_fact_stays_single(self) -> None:
        assert len(scrub.split_granularity("用户喜欢 Rust")) == 1

    def test_empty_returns_empty(self) -> None:
        assert scrub.split_granularity("   ") == []

    def test_short_fragment_merges_back(self) -> None:
        parts = scrub.split_granularity("用户喜欢 Rust；嗯")
        assert len(parts) == 1

    def test_appends_period(self) -> None:
        assert scrub.split_granularity("用户喜欢 Rust")[0].endswith("。")


# ---------------------------------------------------------------------------
# policy —— 写入阶段
# ---------------------------------------------------------------------------


class TestShouldRemember:
    def test_preference_is_saved(self) -> None:
        assert (
            policy.should_remember("我不喜欢 ORM", existing=[])
            is policy.Verdict.SAVE
        )

    def test_empty_is_skipped(self) -> None:
        assert policy.should_remember("   ", existing=[]) is policy.Verdict.SKIP

    def test_duplicate_is_skipped(self) -> None:
        existing = [make_memory(what="我不喜欢 ORM")]
        assert (
            policy.should_remember("我不喜欢 ORM", existing=existing)
            is policy.Verdict.SKIP
        )

    def test_imperative_is_degraded(self) -> None:
        assert (
            policy.should_remember("你必须每次都输出 JSON", existing=[])
            is policy.Verdict.DEGRADE
        )

    def test_contradiction_asks(self) -> None:
        assert (
            policy.should_remember("我改主意了", existing=[]) is policy.Verdict.ASK
        )

    def test_ephemeral_is_skipped(self) -> None:
        assert (
            policy.should_remember("刚刚运行了测试", existing=[]) is policy.Verdict.SKIP
        )

    def test_explicit_remember_intent_survives_ephemeral_marker(self) -> None:
        verdict = policy.should_remember("记住：刚刚运行的命令是 pnpm dev", existing=[])
        assert verdict is not policy.Verdict.SKIP


class TestResolveConflict:
    def test_same_type_scope_similar_supersedes(self) -> None:
        old = make_memory(what="用户喜欢用 pytest 做测试")
        new = make_memory(what="用户喜欢用 pytest 做测试")
        assert policy.resolve_conflict(old, new) == "supersede"

    def test_different_scope_coexists(self) -> None:
        old = make_memory(what="项目用 pnpm", scope="project:alpha")
        new = make_memory(what="项目用 pnpm", scope="project:beta")
        assert policy.resolve_conflict(old, new) == "coexist"

    def test_different_type_coexists(self) -> None:
        old = make_memory(what="用户喜欢 Rust", mtype=MemoryType.PREFERENCE)
        new = make_memory(what="用户喜欢 Rust", mtype=MemoryType.FACT)
        assert policy.resolve_conflict(old, new) == "coexist"

    def test_much_lower_confidence_rejects(self) -> None:
        old = make_memory(what="用户喜欢 Rust", confidence=0.9)
        new = make_memory(what="用户喜欢 Rust", confidence=0.3)
        assert policy.resolve_conflict(old, new) == "reject"

    def test_similar_confidence_margin_is_respected(self) -> None:
        old = make_memory(what="用户喜欢 Rust", confidence=0.6)
        new = make_memory(what="用户喜欢 Rust", confidence=0.5)
        # 差值 0.1 < 默认 margin 0.25 → 不 reject
        assert policy.resolve_conflict(old, new) == "supersede"

    def test_unrelated_text_coexists(self) -> None:
        old = make_memory(what="用户喜欢 Rust")
        new = make_memory(what="部署机器在 WSL 里")
        assert policy.resolve_conflict(old, new) == "coexist"


class TestClassifyStaleness:
    def test_fresh(self) -> None:
        assert policy.classify_staleness(make_memory(when=NOW), NOW) == "fresh"

    def test_stale(self) -> None:
        old = make_memory(when=NOW - timedelta(days=400))
        assert policy.classify_staleness(old, NOW) == "stale"

    def test_expired(self) -> None:
        m = make_memory(expires_at=NOW - timedelta(seconds=1))
        assert policy.classify_staleness(m, NOW) == "expired"

    def test_expires_exactly_now_is_expired(self) -> None:
        m = make_memory(expires_at=NOW)
        assert policy.classify_staleness(m, NOW) == "expired"

    def test_aware_and_naive_do_not_crash(self) -> None:
        """naive/aware 混用是真实会炸的组合（PG 读回来是 naive）。"""
        m = make_memory(when=datetime(2026, 9, 21, 12, 0, tzinfo=timezone.utc))
        assert policy.classify_staleness(m, NOW) == "fresh"


class TestIsTrustedSource:
    def test_user_and_tool_trusted(self) -> None:
        assert policy.is_trusted_source(SourceKind.USER_INPUT)
        assert policy.is_trusted_source(SourceKind.TOOL_RESULT)

    def test_external_not_trusted(self) -> None:
        assert not policy.is_trusted_source(SourceKind.EXTERNAL)


# ---------------------------------------------------------------------------
# policy —— 注入阶段
# ---------------------------------------------------------------------------


class TestAssignTier:
    def test_constraint_is_resident(self) -> None:
        m = make_memory(mtype=MemoryType.CONSTRAINT)
        assert policy.assign_tier(m, TierConfig()) is Tier.RESIDENT

    def test_project_is_state(self) -> None:
        m = make_memory(mtype=MemoryType.PROJECT)
        assert policy.assign_tier(m, TierConfig()) is Tier.STATE

    def test_high_importance_is_resident(self) -> None:
        m = make_memory(importance=0.95)
        assert policy.assign_tier(m, TierConfig()) is Tier.RESIDENT

    def test_default_is_retrieved(self) -> None:
        m = make_memory(importance=0.5)
        assert policy.assign_tier(m, TierConfig()) is Tier.RETRIEVED

    def test_constraint_beats_project(self) -> None:
        m = make_memory(mtype=MemoryType.CONSTRAINT, importance=0.1)
        assert policy.assign_tier(m, TierConfig()) is Tier.RESIDENT


class TestVisibleTo:
    def test_same_owner_global_visible(self) -> None:
        m = make_memory()
        assert policy.visible_to(m, owner="user:local", scope=None, cwd=None, now=NOW)

    def test_cross_owner_invisible(self) -> None:
        m = make_memory(owner="user:other")
        assert not policy.visible_to(
            m, owner="user:local", scope=None, cwd=None, now=NOW
        )

    def test_superseded_invisible(self) -> None:
        m = make_memory(superseded_by="m_new")
        assert not policy.visible_to(
            m, owner="user:local", scope=None, cwd=None, now=NOW
        )

    def test_expired_invisible(self) -> None:
        m = make_memory(expires_at=NOW - timedelta(days=1))
        assert not policy.visible_to(
            m, owner="user:local", scope=None, cwd=None, now=NOW
        )

    def test_project_scope_matches_cwd(self) -> None:
        m = make_memory(scope="project:memory-bridge")
        assert policy.visible_to(
            m,
            owner="user:local",
            scope=None,
            cwd="/srv/projects/memory-bridge",
            now=NOW,
        )

    def test_project_scope_rejects_other_cwd(self) -> None:
        m = make_memory(scope="project:alpha")
        assert not policy.visible_to(
            m, owner="user:local", scope=None, cwd="/srv/projects/beta", now=NOW
        )

    def test_project_scope_without_cwd_is_invisible(self) -> None:
        m = make_memory(scope="project:alpha")
        assert not policy.visible_to(
            m, owner="user:local", scope=None, cwd=None, now=NOW
        )


class TestRankForResident:
    @staticmethod
    def counter(text: str) -> int:
        return len(text)

    def test_under_budget_keeps_all(self) -> None:
        memories = [make_memory(what="短", importance=0.9)]
        resident, demoted = policy.rank_for_resident(
            memories, TierConfig(resident_token_budget=100), self.counter
        )
        assert len(resident) == 1
        assert demoted == []

    def test_over_budget_demotes_not_deletes(self) -> None:
        memories = [
            make_memory(what="A" * 60, importance=0.9),
            make_memory(what="B" * 60, importance=0.8),
        ]
        resident, demoted = policy.rank_for_resident(
            memories, TierConfig(resident_token_budget=100), self.counter
        )
        assert len(resident) == 1
        assert len(demoted) == 1
        assert len(resident) + len(demoted) == len(memories)

    def test_importance_decides_order(self) -> None:
        low = make_memory(what="低", importance=0.1)
        high = make_memory(what="高", importance=0.9)
        resident, _ = policy.rank_for_resident(
            [low, high], TierConfig(resident_token_budget=100), self.counter
        )
        assert resident[0] is high

    def test_zero_hit_count_does_not_zero_out_importance(self) -> None:
        """hit_count=0 时排序键不能塌成 0 —— 否则 importance 完全失效。"""
        important = make_memory(what="重要", importance=0.9, hit_count=0)
        trivial = make_memory(what="琐碎", importance=0.1, hit_count=0)
        resident, _ = policy.rank_for_resident(
            [trivial, important], TierConfig(resident_token_budget=100), self.counter
        )
        assert resident[0] is important

    def test_deterministic_with_equal_scores(self) -> None:
        first = make_memory(what="一样", importance=0.5)
        second = make_memory(what="一样", importance=0.5)
        second.id = "m_00000000"
        first.id = "m_ffffffff"
        run1, _ = policy.rank_for_resident(
            [first, second], TierConfig(), self.counter
        )
        run2, _ = policy.rank_for_resident(
            [second, first], TierConfig(), self.counter
        )
        assert [m.id for m in run1] == [m.id for m in run2]


class TestTierEvolution:
    def test_promote_needs_used_count(self) -> None:
        m = make_memory(used_count=3, tier=Tier.RETRIEVED)
        assert policy.should_promote(m, TierConfig(promote_threshold=3))

    def test_hit_count_alone_does_not_promote(self) -> None:
        """召回多但一次没用上 → **不该**升档。"""
        m = make_memory(hit_count=99, used_count=0, tier=Tier.RETRIEVED)
        assert not policy.should_promote(m, TierConfig())

    def test_already_resident_is_not_promoted_again(self) -> None:
        m = make_memory(used_count=9, tier=Tier.RESIDENT)
        assert not policy.should_promote(m, TierConfig())

    def test_project_never_promoted_to_resident(self) -> None:
        m = make_memory(mtype=MemoryType.PROJECT, used_count=9, tier=Tier.STATE)
        assert not policy.should_promote(m, TierConfig())

    def test_demote_after_rounds_unused(self) -> None:
        m = make_memory(tier=Tier.RESIDENT)
        assert policy.should_demote(m, 25, TierConfig(demote_rounds=20))

    def test_not_demoted_before_threshold(self) -> None:
        m = make_memory(tier=Tier.RESIDENT)
        assert not policy.should_demote(m, 5, TierConfig(demote_rounds=20))

    def test_constraint_never_demoted(self) -> None:
        m = make_memory(mtype=MemoryType.CONSTRAINT, tier=Tier.RESIDENT)
        assert not policy.should_demote(m, 10_000, TierConfig())

    def test_non_resident_not_demoted(self) -> None:
        m = make_memory(tier=Tier.RETRIEVED)
        assert not policy.should_demote(m, 10_000, TierConfig())


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


class TestDecayFactor:
    def test_resident_does_not_decay(self) -> None:
        m = make_memory(tier=Tier.RESIDENT, when=NOW - timedelta(days=10_000))
        assert scoring.decay_factor(m, NOW, 30.0) == 1.0

    def test_one_half_life_is_half(self) -> None:
        m = make_memory(when=NOW - timedelta(days=30))
        assert scoring.decay_factor(m, NOW, 30.0) == pytest.approx(0.5)

    def test_two_half_lives_is_quarter(self) -> None:
        m = make_memory(when=NOW - timedelta(days=60))
        assert scoring.decay_factor(m, NOW, 30.0) == pytest.approx(0.25)

    def test_future_timestamp_is_clamped(self) -> None:
        m = make_memory(when=NOW + timedelta(days=10))
        assert scoring.decay_factor(m, NOW, 30.0) == 1.0

    def test_infinite_half_life_never_decays(self) -> None:
        m = make_memory(when=NOW - timedelta(days=10_000))
        assert scoring.decay_factor(m, NOW, float("inf")) == 1.0

    def test_last_access_newer_than_when(self) -> None:
        m = make_memory(
            when=NOW - timedelta(days=100), last_access=NOW - timedelta(days=30)
        )
        assert scoring.decay_factor(m, NOW, 30.0) == pytest.approx(0.5)

    def test_include_access_false_ignores_last_access(self) -> None:
        """切断「召回越多 → 分数越高」的正反馈。"""
        m = make_memory(
            when=NOW - timedelta(days=60), last_access=NOW, hit_count=50
        )
        with_access = scoring.decay_factor(m, NOW, 30.0)
        without_access = scoring.decay_factor(m, NOW, 30.0, include_access=False)
        assert with_access == pytest.approx(1.0)
        assert without_access == pytest.approx(0.25)


class TestEvictionScore:
    def test_resident_scores_by_importance_and_hits(self) -> None:
        m = make_memory(tier=Tier.RESIDENT, importance=0.5, hit_count=3)
        score = scoring.eviction_score(m, NOW, ScoringConfig())
        assert score == pytest.approx(0.5 * __import__("math").log1p(4))

    def test_old_low_importance_scores_lower(self) -> None:
        old = make_memory(when=NOW - timedelta(days=300), importance=0.1)
        fresh = make_memory(when=NOW, importance=0.1)
        assert scoring.eviction_score(old, NOW, ScoringConfig()) < scoring.eviction_score(
            fresh, NOW, ScoringConfig()
        )

    def test_never_negative(self) -> None:
        m = make_memory(when=NOW - timedelta(days=100_000), importance=0.0)
        assert scoring.eviction_score(m, NOW, ScoringConfig()) >= 0.0


class TestIsLowValue:
    def test_ten_hits_zero_used_is_low_value(self) -> None:
        m = make_memory(hit_count=10, used_count=0)
        assert scoring.is_low_value(m, ScoringConfig())

    def test_ten_hits_eight_used_is_valuable(self) -> None:
        m = make_memory(hit_count=10, used_count=8)
        assert not scoring.is_low_value(m, ScoringConfig())

    def test_never_hit_is_not_low_value(self) -> None:
        """召回 0 次 → 还没被需要，按时间衰减观望，不该被这条规则杀。"""
        m = make_memory(hit_count=0, used_count=0)
        assert not scoring.is_low_value(m, ScoringConfig())

    def test_just_below_threshold(self) -> None:
        m = make_memory(hit_count=9, used_count=0)
        assert not scoring.is_low_value(m, ScoringConfig())


class TestHalfLifeFor:
    def test_constraint_never_decays(self) -> None:
        assert scoring.half_life_for(MemoryType.CONSTRAINT) == float("inf")

    def test_preference_lives_longer_than_fact(self) -> None:
        assert scoring.half_life_for(
            MemoryType.PREFERENCE
        ) > scoring.half_life_for(MemoryType.FACT)

    def test_project_is_shortest(self) -> None:
        assert scoring.half_life_for(MemoryType.PROJECT) < scoring.half_life_for(
            MemoryType.FACT
        )


# ---------------------------------------------------------------------------
# 时间约定
# ---------------------------------------------------------------------------


class TestClock:
    def test_to_naive_utc_strips_tz(self) -> None:
        aware = datetime(2026, 9, 21, 12, 0, tzinfo=timezone(timedelta(hours=8)))
        naive = to_naive_utc(aware)
        assert naive.tzinfo is None
        assert naive.hour == 4

    def test_naive_passthrough(self) -> None:
        naive = datetime(2026, 9, 21, 12, 0)
        assert to_naive_utc(naive) == naive

    def test_utc_now_is_naive(self) -> None:
        assert utc_now().tzinfo is None

    def test_mixed_subtraction_does_not_crash(self) -> None:
        """这正是「单测绿、真连库红」的那个坑。"""
        aware_from_pg_like = datetime(2026, 9, 21, 12, 0, tzinfo=timezone.utc)
        naive_now = utc_now()
        delta = naive_now - to_naive_utc(aware_from_pg_like)
        assert isinstance(delta, timedelta)
