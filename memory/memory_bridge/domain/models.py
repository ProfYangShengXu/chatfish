"""L1 领域数据结构。

职责：定义全部领域对象。**没有方法，只有字段**（行为在 policy / scoring 里）。

不做什么：
  ❌ 不写 ``to_prompt()`` / ``render()`` 之类的方法（渲染归 ``service/inject.py``）
  ❌ 不写数据库映射（归 ``impl/pg_store.py``）
  ❌ 不 import 项目内其它模块，也不 import psycopg / sentence_transformers / fastapi
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
from typing import Optional


class MemoryType(str, Enum):
    PREFERENCE = "preference"  # 用户偏好   → 检索档
    FACT = "fact"  # 稳定事实   → 检索档
    CONSTRAINT = "constraint"  # 硬约束     → 常驻档
    PROJECT = "project_state"  # 项目状态   → 状态缓存档


class Priority(str, Enum):
    NORMAL = "normal"  # 攒着，压缩时合并
    URGENT = "urgent"  # 立即改头部（认掉缓存损失）


class SourceKind(str, Enum):
    USER_INPUT = "user"  # 信任级：高
    TOOL_RESULT = "tool"  # 信任级：中
    EXTERNAL = "external"  # 信任级：低（不进记忆）


class Tier(str, Enum):
    RESIDENT = "resident"  # 常驻档 —— 每轮都在，不检索
    STATE = "state"  # 状态缓存档 —— 项目/任务状态
    RETRIEVED = "retrieved"  # 检索档 —— 按需召回


@dataclass(slots=True)
class Memory:
    id: str
    owner: str
    what: str
    type: MemoryType
    when: datetime
    scope: str = "global"
    source: Optional[str] = None  # "session:abc#turn7"
    source_kind: SourceKind = SourceKind.USER_INPUT
    confidence: float = 0.5
    importance: float = 0.5
    expires_at: Optional[datetime] = None
    hit_count: int = 0  # 被召回次数
    used_count: int = 0  # 召回后【确实被用上】的次数（升档判据）
    tier: Tier = Tier.RETRIEVED  # 当前档位（自适应调整）
    last_access: Optional[datetime] = None
    superseded_by: Optional[str] = None
    embedding: Optional[list[float]] = None  # 不进 SQL 时用它传


@dataclass(slots=True)
class Document:  # 知识域的 chunk
    id: str
    collection: str
    source_doc: str
    text: str
    acl: list[str]
    doc_version: Optional[str] = None
    effective_date: Optional[date] = None
    section: Optional[str] = None
    chunk_index: int = 0
    parent_id: Optional[str] = None  # ★ 只有【子块】填它；父块为 None
    is_parent: bool = False  # ★ 区分父子（检索只查子块）
    embedding_model: str = ""
    embedding: Optional[list[float]] = None


@dataclass(slots=True)
class Patch:
    id: int
    owner: str
    text: str
    priority: Priority
    at: datetime


@dataclass(slots=True)
class RecalledContext:  # 召回的最终产物（三层分开）
    resident: list[Memory] = field(default_factory=list)
    state: list[Memory] = field(default_factory=list)
    retrieved: list[Memory] = field(default_factory=list)
    documents: list[Document] = field(default_factory=list)
    # ★ 因常驻档 token 预算超限而被【退回检索档】的条目。
    #   单独记一份是为了让这条降级路径**可观测** ——
    #   否则「退回」在结果里看不出来（它们既不在 resident，也未必命中检索），
    #   「退回不是删除」这句规格约束就无法验证。
    demoted: list[Memory] = field(default_factory=list)


# --------------------------------------------------------------------------
# 规则参数（纯数据，供 domain 的纯函数与 cli.py 装配共用）
# 放在 L1 是为了让 policy / scoring 保持「零内部依赖」——
# 它们只吃这些不带行为的配置载体。
# --------------------------------------------------------------------------


@dataclass(slots=True)
class TierConfig:
    """分档与常驻预算。预算单位是 **token**，不是条数。"""

    resident_token_budget: int = 800
    promote_threshold: int = 3  # used_count 累计到多少 → 升常驻
    demote_rounds: int = 20  # 连续多少轮没被用到 → 降档
    resident_importance: float = 0.9  # importance >= 此值 → RESIDENT


@dataclass(slots=True)
class ScoringConfig:
    """衰减与淘汰。"""

    half_life_days_default: float = 90.0
    half_life_preference: float = 180.0  # 偏好：数月
    half_life_fact: float = 90.0
    half_life_project: float = 21.0  # 项目状态：数周
    half_life_constraint: float = float("inf")  # 硬约束：不衰减
    evict_threshold: float = 0.05  # eviction_score 低于此值可淘汰
    low_value_min_hits: int = 10  # 「召回 10 次」的 10
    low_value_max_used: int = 0  # 「用上 0 次」的 0
    stale_after_days: float = 180.0  # 超过则视为 stale


@dataclass(slots=True)
class PolicyConfig:
    """写入阶段的判断参数。"""

    confidence_reject_margin: float = 0.25  # 新条目置信度低这么多 → reject
    duplicate_overlap: float = 0.85  # 文本重合度超过此值视为重复
    max_text_chars: int = 4000  # 单条记忆的最大字符数
    stale_after_days: float = 180.0


@dataclass(slots=True)
class RecallConfig:
    """召回管道参数。"""

    recall_k: int = 50  # 召回候选数
    recall_final_k: int = 5  # 最终注入条数
    rerank_max_candidates: int = 50  # ★ 延迟预算
    rrf_k: int = 60
    bm25_k: int = 50
    ann_oversample: int = 4  # 应用层放大 k，缓解 pgvector 后过滤塌陷
    # ★ 相关性下限：低于它的候选**不返回**（返回空 = 上层据此拒答）。
    #   没有这个阈值时，ANN 的 top-k **永远**返回 k 条 ——
    #   余弦相似度对任何输入都有值，「库里没有」和「库里有一条勉强相关」
    #   在结果上完全一样，于是「拒答」根本无法实现。
    #   默认 0.01 是本机实测出来的：cross-encoder 对无关 query 的最高分是 0.001，
    #   对相关 query 的最高分最低是 0.06（见汇报 F 节）。
    min_relevance: float = 0.01
    # ★ 尾部补丁的上限：超过就把它们合并进常驻区并清空缓冲。
    #   没有这个上限时补丁会无限增长（每次 remember 都往缓冲里塞一条），
    #   于是**每一次** /recall 都把全部历史补丁重放一遍 ——
    #   上下文被撑爆，而且越用越糟。规格只写了「压缩时合并」，
    #   但没规定那个时刻由谁触发；这里把它落成一条确定性规则。
    patch_flush_threshold: int = 20


@dataclass(slots=True)
class ForgetConfig:
    """容量治理参数。"""

    max_active: int = 2000  # 单 owner 的 active 记忆上限
    utilization_threshold: float = 0.8  # 利用率低于此值什么都不做
    batch: int = 200  # 单轮最多处理多少条


@dataclass(slots=True)
class HybridConfig:
    """混合检索参数。"""

    rrf_k: int = 60
    bm25_k: int = 50


def utc_now() -> datetime:
    """统一的时间源：**永远是 aware UTC**。

    这样从 PG 的 TIMESTAMPTZ 读回来的 aware datetime 与领域层自造的 now 可以直接相减，
    不会踩 naive/aware 混用的 TypeError。
    """
    return datetime.now(timezone.utc)
