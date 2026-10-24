"""生成 ``eval/cases.jsonl``。

为什么需要脚本而不是手写：``gold`` 是**运行时生成**的记忆 id
（``m_<日期>_<时间>_<随机>``），手写会立刻悬空。
这个脚本把「事实文本 ↔ query ↔ gold」的映射固化下来，重跑即可得到一致的回归集。

⚠️ 规格要求「回归集建立后不要改既有用例（否则跨版本不可比）」——
   所以本脚本只在**首次建库**时用；之后改用例必须显式说明为什么。

用法::

    python scripts/gen_eval_cases.py            # 写入 eval/cases.jsonl + eval/seed.json
    python scripts/gen_eval_cases.py --owner user:local --other-owner user:other
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from memory_bridge.cli import build_pipeline  # noqa: E402
from memory_bridge.config import load_config  # noqa: E402
from memory_bridge.domain.clock import utc_now  # noqa: E402

# ---------------------------------------------------------------------------
# 20 条「该被召回」的事实： (query, 事实文本)
# query 是**用户会怎么问**，事实是**落库的原文** ——
# 刻意用不同的措辞，避免「query 里含答案」导致指标虚高。
# ---------------------------------------------------------------------------
RETRIEVAL_CASES: tuple[tuple[str, str], ...] = (
    ("我平时更愿意用哪种方式访问数据库？", "用户偏好手写 SQL，不喜欢 ORM"),
    ("我一般拿什么写测试？", "用户习惯用 pytest 写单元测试"),
    ("这个仓库用哪个包管理器？", "本项目统一用 pnpm 管理依赖"),
    ("代码跑在什么环境里？", "开发环境是 Windows 11 加 WSL2 Ubuntu"),
    ("我提过对回答语言的要求吗？", "用户要求回答一律使用中文"),
    ("我提交信息的格式是什么？", "提交信息用中文写，首行不超过 50 字"),
    ("我不喜欢哪一类抽象？", "用户反感过度抽象，宁可重复三次"),
    ("数据库的版本是多少？", "数据库是 PostgreSQL 16 加 pgvector 扩展"),
    ("我要求过参数怎么配置？", "配置一律放配置文件，不允许硬编码在代码里"),
    ("我在这个项目里用什么做向量检索？", "向量检索用 pgvector，不引入专用向量库"),
    ("我对日志有什么要求？", "日志必须带请求 id，方便定位问题"),
    ("我倾向于怎么写类型标注？", "用户偏好全量类型注解，不写 Any"),
    ("生产环境的部署方式是什么？", "生产部署走容器，本地开发直接跑进程"),
    ("我提过接口返回格式吗？", "接口统一返回 JSON，字段名用下划线风格"),
    ("我不允许出现什么？", "不允许在代码里留 TODO 占位"),
    ("我对依赖数量是什么态度？", "用户要求依赖越少越好，能自己写就不引库"),
    ("我的时区是什么？", "用户所在时区是 UTC+8"),
    ("我要求怎么处理错误？", "错误处理要返回结果对象，不靠抛异常表达业务失败"),
    ("我对命名风格的要求？", "命名用英文，不加无意义的前缀"),
    ("我在意哪一项性能指标？", "用户最在意召回延迟，其次是准确率"),
)

# ---------------------------------------------------------------------------
# 干扰项：让库里的记忆数明显多于 recall_final_k，
# 否则 MRR 会天然接近 1，reranker 开关的差异根本看不出来。
# ---------------------------------------------------------------------------
DISTRACTORS: tuple[str, ...] = (
    "用户喜欢喝咖啡，尤其是浅烘",
    "用户住在杭州，通勤靠地铁",
    "用户养了一只叫豆豆的猫",
    "用户每周三晚上有羽毛球活动",
    "用户偏好深色主题的编辑器",
    "用户不喜欢在会议上做汇报",
    "用户的笔记本是 ThinkPad，键盘手感好",
    "用户习惯早上六点起床写代码",
    "用户讨厌写周报",
    "用户喜欢用 vim 的键位绑定",
    "用户对咖啡因比较敏感，下午不喝",
    "用户偏好纸质书而不是电子书",
    "用户的显示器是 27 寸 4K",
    "用户不喜欢电话沟通，偏好文字",
    "用户喜欢在通勤路上听播客",
    "用户习惯把任务记在纸上",
    "用户偏好机械键盘，青轴",
    "用户不太用社交媒体",
    "用户喜欢周末去爬山",
    "用户对花生过敏",
)

# 5 条「知识库里没有」的问题 → 应该拒答
REFUSAL_CASES: tuple[str, ...] = (
    "公司在哪一年完成了 C 轮融资？",
    "这个项目的商业授权费用是多少？",
    "去年团队一共招聘了几个人？",
    "产品的下一个大版本计划什么时候发布？",
    "首席架构师的全名是什么？",
)

# 3 条跨 owner 用例： (query, 别的 owner 的事实)
CROSS_OWNER_CASES: tuple[tuple[str, str], ...] = (
    ("我最喜欢的编程字体是什么？", "另一个用户的偏好：喜欢 Fira Code 字体"),
    ("我的手机号后四位是多少？", "另一个用户的联系方式：手机尾号 8899"),
    ("我常用的数据库端口是多少？", "另一个用户的习惯：数据库端口用 15432"),
)

# 2 条版本冲突用例：同一份文档的两个版本，问同一个问题
VERSION_CASES: tuple[tuple[str, str], ...] = (
    ("RRF 融合的 k 参数取多少？", "kb"),
    ("常驻档的 token 预算是多少？", "kb"),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成 eval/cases.jsonl")
    parser.add_argument("--owner", default=None)
    parser.add_argument("--other-owner", default="user:other")
    parser.add_argument("--out", default=str(REPO_ROOT / "eval" / "cases.jsonl"))
    parser.add_argument(
        "--seed", default=str(REPO_ROOT / "eval" / "seed.json"), help="记忆 id 映射"
    )
    args = parser.parse_args(argv)

    from memory_bridge.cli import _load_env
    from memory_bridge.domain.models import SourceKind

    _load_env()
    cfg = load_config()
    owner = args.owner or cfg.owner
    pipeline = build_pipeline(cfg)

    base = utc_now()
    seed: dict[str, object] = {"owner": owner, "other_owner": args.other_owner}

    def write(owner_name: str, text: str, at: datetime) -> str:
        """走**事件写入管道**（而不是 remember_explicit）。

        为什么必须这样：``remember_explicit`` 按规格把 importance 拉满到 1.0，
        于是 ``policy.assign_tier`` 的规则③会把**每一条**都放进 RESIDENT 档 ——
        而常驻档是「不检索」的，检索档里一条都不剩，
        MRR / Hit@1 会全部变成 0（实测踩过：40 条记忆 100% 落在 resident）。

        走 ``ingest_event`` 才会拿到真实的 type / importance / tier 分档，
        评估的也才是**检索**这条路径。
        """
        report = pipeline.write.ingest_event_detailed(
            owner=owner_name,
            text=text,
            source="eval:seed",
            source_kind=SourceKind.USER_INPUT,
            now=at,
        )
        if not report.written:
            raise SystemExit(f"seed 失败：{text!r} 被跳过（{report.skipped}）")
        return report.written[0].id

    # —— 写入 20 条 gold ——
    gold_ids: dict[str, str] = {}
    for index, (query, fact) in enumerate(RETRIEVAL_CASES):
        gold_ids[query] = write(
            owner,
            fact,
            base - timedelta(minutes=len(RETRIEVAL_CASES) - index),
        )

    # 再多召回几次，让 hit_count 有意义（也让 RRF/rerank 有得排）
    for query in list(gold_ids)[:10]:
        pipeline.session.recall_tail(owner=owner, query=query)

    # —— 干扰项 ——
    for text in DISTRACTORS:
        write(owner, text, base)

    # —— 另一个 owner 的 3 条 ——
    for _, fact in CROSS_OWNER_CASES:
        write(args.other_owner, fact, base)

    # —— 版本冲突的两份文档 ——
    from scripts.kb_ingest import ingest_file

    kb_dir = REPO_ROOT / "tests" / "fixtures"
    ingest_file(
        pipeline,
        collection="kb",
        path=kb_dir / "memory-bridge.md",
        doc_version="v1",
        effective_date=(base - timedelta(days=180)).date(),
        acl=["public"],
    )
    ingest_file(
        pipeline,
        collection="kb",
        path=kb_dir / "memory-bridge-v2.md",
        doc_version="v2",
        effective_date=base.date(),
        acl=["public"],
    )

    seed["gold"] = gold_ids
    seed["other_owner_facts"] = [fact for _, fact in CROSS_OWNER_CASES]

    # —— 组装 cases.jsonl ——
    lines: list[str] = []
    lines.append("# Memory Bridge 回归集")
    lines.append("# 规则：gold 是运行时生成的记忆 id，重建库后必须重跑 gen_eval_cases.py")
    lines.append("# 字段说明：")
    lines.append('#   {"query":..., "gold":...}                        20 条 → MRR / Hit@1')
    lines.append('#   {"query":..., "expect_refusal":true}              5 条 → 拒答准确率')
    lines.append('#   {"query":..., "forbidden_owner":...}              3 条 → 污染率（必须为 0）')
    lines.append('#   {"query":..., "version_conflict":true, ...}       2 条 → 人工看「择新并说明」')
    lines.append(f'{{"_meta": {{"owner": "{owner}", "other_owner": "{args.other_owner}", '
                 f'"generated_at": "{base.isoformat()}"}}}}')

    for query, fact in RETRIEVAL_CASES:
        payload = {"query": query, "gold": gold_ids[query], "fact": fact}
        lines.append(json.dumps(payload, ensure_ascii=False))

    for query in REFUSAL_CASES:
        lines.append(
            json.dumps({"query": query, "expect_refusal": True}, ensure_ascii=False)
        )

    for query, fact in CROSS_OWNER_CASES:
        lines.append(
            json.dumps(
                {
                    "query": query,
                    "owner": owner,
                    "forbidden_owner": args.other_owner,
                    "other_fact": fact,
                    "result_owners": [],
                },
                ensure_ascii=False,
            )
        )

    for query, collection in VERSION_CASES:
        lines.append(
            json.dumps(
                {
                    "query": query,
                    "version_conflict": True,
                    "collection": collection,
                    "doc_versions": ["v1", "v2"],
                    "newest_version": "v2",
                },
                ensure_ascii=False,
            )
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    seed_path = Path(args.seed)
    seed_path.write_text(
        json.dumps(seed, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    total = len(RETRIEVAL_CASES) + len(REFUSAL_CASES) + len(CROSS_OWNER_CASES) + len(
        VERSION_CASES
    )
    print(
        json.dumps(
            {
                "cases_written": total,
                "retrieval": len(RETRIEVAL_CASES),
                "refusal": len(REFUSAL_CASES),
                "cross_owner": len(CROSS_OWNER_CASES),
                "version_conflict": len(VERSION_CASES),
                "out": str(out_path),
                "seed": str(seed_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
