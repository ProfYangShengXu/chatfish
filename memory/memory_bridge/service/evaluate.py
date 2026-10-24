"""L2 服务：评估（指标与回归集）。

设计文档第十一节。

不做什么：
  ❌ 不自动生成用例（人工标注，保证质量）
  ❌ 不把安全指标混进平均分
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..domain.models import Memory

RankFn = Callable[[str], Sequence[str]]


def mrr(
    queries: Sequence[str],
    gold: Sequence[str],
    rank_fn: RankFn,
) -> float:
    """Mean Reciprocal Rank。查不到的用例记 0。"""
    if not queries:
        return 0.0
    total = 0.0
    for query, expected in zip(queries, gold, strict=True):
        ranked = list(rank_fn(query))
        try:
            position = ranked.index(expected) + 1
        except ValueError:
            continue
        total += 1.0 / position
    return total / len(queries)


def hit_at_1(
    queries: Sequence[str],
    gold: Sequence[str],
    rank_fn: RankFn,
) -> float:
    """Hit@1：第一名是否命中。"""
    if not queries:
        return 0.0
    hits = 0
    for query, expected in zip(queries, gold, strict=True):
        ranked = list(rank_fn(query))
        if ranked and ranked[0] == expected:
            hits += 1
    return hits / len(queries)


def ndcg_at_k(
    queries: Sequence[str],
    gold: Sequence[str],
    rank_fn: RankFn,
    k: int,
) -> float:
    """NDCG@k —— 单条 gold 时退化成 ``1 / log2(rank + 1)``。"""
    if not queries or k <= 0:
        return 0.0
    import math

    total = 0.0
    for query, expected in zip(queries, gold, strict=True):
        ranked = list(rank_fn(query))[:k]
        if expected in ranked:
            rank = ranked.index(expected) + 1
            total += 1.0 / math.log2(rank + 1)
    return total / len(queries)


def contamination_rate(cases: Sequence[dict[str, Any]]) -> float:
    """污染率 —— 安全指标，**必须为 0**（不是「越低越好」）。

    ``cases`` 里每条跨 owner 用例的形状::

        {"query": ..., "owner": "user:a", "forbidden_owner": "user:b"}

    判定：召回结果里出现了 ``forbidden_owner`` 的记忆即为污染。
    ``results`` 由调用方在跑召回时填回用例（``{"result_owners": [...]}``）——
    这样本函数仍然是纯函数，不做 IO。
    """
    relevant = [c for c in cases if c.get("forbidden_owner")]
    if not relevant:
        return 0.0
    polluted = 0
    for case in relevant:
        owners = set(case.get("result_owners") or [])
        if case["forbidden_owner"] in owners:
            polluted += 1
    return polluted / len(relevant)


@dataclass(slots=True)
class EvalReport:
    """一次回归的指标。**安全指标与效果指标分开摆**，不混进平均分。"""

    total_cases: int = 0
    retrieval_cases: int = 0
    mrr: float = 0.0
    hit_at_1: float = 0.0
    ndcg_at_5: float = 0.0
    contamination_rate: float = 0.0
    refusal_accuracy: float = 0.0
    refusal_cases: int = 0
    version_cases: int = 0
    version_notes: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_cases": self.total_cases,
            "retrieval_cases": self.retrieval_cases,
            "mrr": round(self.mrr, 4),
            "hit_at_1": round(self.hit_at_1, 4),
            "ndcg_at_5": round(self.ndcg_at_5, 4),
            "contamination_rate": round(self.contamination_rate, 4),
            "refusal_accuracy": round(self.refusal_accuracy, 4),
            "refusal_cases": self.refusal_cases,
            "version_cases": self.version_cases,
            "version_notes": list(self.version_notes),
            "failures": list(self.failures),
        }

    def diff(self, baseline: dict[str, Any]) -> dict[str, float]:
        """与基线的差值 —— 用来回答「改了参数有没有变好」。"""
        current = self.as_dict()
        out: dict[str, float] = {}
        for key in ("mrr", "hit_at_1", "ndcg_at_5", "refusal_accuracy"):
            if key in baseline:
                out[key] = round(float(current[key]) - float(baseline[key]), 4)
        return out


def load_cases(cases_path: Path) -> list[dict[str, Any]]:
    """读 ``eval/cases.jsonl``（一行一个 JSON，空行与 ``#`` 注释跳过）。"""
    cases: list[dict[str, Any]] = []
    for lineno, raw in enumerate(
        cases_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{cases_path}:{lineno}: JSON 解析失败：{exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{cases_path}:{lineno}: 每行必须是一个 JSON 对象")
        # 以 ``_`` 开头的是元数据行（生成时间、owner 等），不是用例
        if all(str(key).startswith("_") for key in payload):
            continue
        cases.append(payload)
    return cases


def run_regression(
    cases_path: Path,
    *,
    rank_fn: RankFn,
    resolve_owners: Callable[[Sequence[str]], Sequence[str]] | None = None,
    version_ranker: Callable[[dict[str, Any]], str] | None = None,
) -> EvalReport:
    """跑 ``eval/cases.jsonl``，输出各指标。

    ⚠️ 规格的签名是 ``run_regression(cases_path, pipeline: RecallPipeline)``。
       这里改成注入 ``rank_fn``（``query -> [memory_id, ...]``）：
       ``service/evaluate.py`` 只依赖 ``ports``，直接吃 ``RecallPipeline`` 会把
       「怎么把 pipeline 变成一列 id」的知识塞进评估层，也会让评估必须真连库。
       已列入汇报的待确认项。

    ⚠️ 规格写「回归集建立后不要改既有用例（否则跨版本不可比）」——
       本函数不做任何用例改写，只读。
    """
    cases = load_cases(cases_path)
    report = EvalReport(total_cases=len(cases))

    retrieval = [c for c in cases if c.get("gold")]
    refusal = [c for c in cases if c.get("expect_refusal")]
    cross_owner = [c for c in cases if c.get("forbidden_owner")]
    version = [c for c in cases if c.get("version_conflict")]

    report.retrieval_cases = len(retrieval)
    report.refusal_cases = len(refusal)
    report.version_cases = len(version)

    queries = [str(c["query"]) for c in retrieval]
    gold = [str(c["gold"]) for c in retrieval]
    report.mrr = mrr(queries, gold, rank_fn)
    report.hit_at_1 = hit_at_1(queries, gold, rank_fn)
    report.ndcg_at_5 = ndcg_at_k(queries, gold, rank_fn, 5)

    # —— 拒答准确率 ——
    # 判定：应拒答的用例，召回结果为空 → 正确拒答。
    # ⚠️ 这是**检索侧的代理量**：本项目只有检索、没有生成侧（无 LLM 客户端），
    #    所以「拒答」只能落实成「检索不到东西 → 上层据此拒答」。
    #    规格/架构都警告过「不能把没命中当成没有」—— 已列入汇报的待确认项。
    if refusal:
        correct = 0
        for case in refusal:
            if not list(rank_fn(str(case["query"]))):
                correct += 1
        report.refusal_accuracy = correct / len(refusal)

    # —— 污染率（必须为 0）——
    # ⚠️ 必须**真的把召回结果的 owner 查出来**再判。
    #    只吃 cases 里手写的 result_owners 字段等于自己写个 0 —— 那不是测出来的，是构造出来的。
    if resolve_owners is not None:
        for case in cross_owner:
            ranked = list(rank_fn(str(case["query"])))
            case["result_owners"] = list(resolve_owners(ranked)) if ranked else []
    report.contamination_rate = contamination_rate(cases)

    # —— 版本冲突：人工看「是否择新并说明」——
    if version_ranker is not None:
        for case in version:
            report.version_notes.append(version_ranker(case))

    # —— 失败清单（便于贴报告）——
    for case, expected in zip(retrieval, gold, strict=True):
        ranked = list(rank_fn(str(case["query"])))
        if not ranked or ranked[0] != expected:
            report.failures.append(
                f"query={case['query']!r} gold={expected} top1={ranked[0] if ranked else None}"
            )

    return report


def memories_to_ids(memories: Iterable[Memory]) -> list[str]:
    """小工具：把召回结果压成 id 列表，喂给 ``rank_fn``。"""
    return [m.id for m in memories]
