"""★★ 唯一装配点：全项目唯一 ``new`` 具体对象的地方。

换实现只改这个文件。

⚠️ 两个纪律：
  1. **``service/`` 里不允许出现 ``PgMemoryStore(`` / ``BgeSmallZhEmbedder(`` 这种构造调用**
     —— 那意味着分层被打破（三条 grep 自检会抓到）
  2. **配置从 ``config.yaml`` 读，不写死在代码里**（除模型名和维度这类结构常量）
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from collections.abc import Sequence
from pathlib import Path

from .bridge.context_file import ContextFileWriter
from .bridge.session import SessionLifecycle
from .bridge.transcript import FileTranscriptReader
from .config import Config, load_config, load_env_file
from .domain.clock import utc_now
from .domain.models import ForgetConfig, RecallConfig, SourceKind
from .impl.embedder import BgeSmallZhEmbedder, HashEmbedder
from .impl.hybrid import Bm25Index, HybridRetriever
from .impl.pg_store import PgMemoryStore
from .impl.reranker import CrossEncoderReranker, NoopAttributor
from .ports import (
    Embedder,
    MemoryStore,
    NoopReranker,
    Reranker,
    Retriever,
    UsageAttributor,
)
from .service.evaluate import run_regression
from .service.forget import ForgetPipeline
from .service.knowledge import KnowledgeService
from .service.recall import RecallPipeline
from .service.write import WritePipeline

logger = logging.getLogger("memory_bridge")


@dataclass(slots=True)
class Pipeline:
    """装配好的对象图。``cli`` 与 ``bridge/server.py`` 都只认这个。"""

    cfg: Config
    store: MemoryStore
    embedder: Embedder
    retriever: Retriever
    reranker: Reranker
    attributor: UsageAttributor
    write: WritePipeline
    recall: RecallPipeline
    forget: ForgetPipeline
    knowledge: KnowledgeService
    session: SessionLifecycle
    loaded_models: list[str] = field(default_factory=list)
    missing_models: list[str] = field(default_factory=list)


def build_embedder(cfg: Config) -> tuple[Embedder, bool]:
    """建 embedder。返回 ``(实现, 是否用了真模型)``。

    ⚠️ 模型缺失时**不静默降级成假实现**：报告在 ``missing_models`` 里，
       由 `/health` 与启动日志暴露。但为了让「没下模型也能把管道接线跑通」，
       这里退到 ``HashEmbedder`` —— 它是**测试替身**，不是第二套存储实现。
    """
    model_dir = cfg.embedder_model_dir
    if not model_dir.exists():
        logger.warning("embedding 模型缺失：%s", model_dir)
        return HashEmbedder(dim=cfg.embedding_dim), False
    return BgeSmallZhEmbedder(model_dir, device="cpu"), True


def build_reranker(cfg: Config) -> tuple[Reranker, bool]:
    """建 reranker。``rerank_enabled`` 为假时用 ``NoopReranker``。

    —— 它让「不用精排」成为一个**配置**，而不是「缺了个模块」。
    """
    if not cfg.rerank_enabled:
        return NoopReranker(), False
    model_dir = cfg.reranker_model_dir
    if not model_dir.exists():
        logger.warning("reranker 模型缺失：%s → 退回 NoopReranker", model_dir)
        return NoopReranker(), False
    return (
        CrossEncoderReranker(model_dir, max_candidates=cfg.rerank_max_candidates),
        True,
    )


def build_pipeline(cfg: Config) -> Pipeline:
    """装配（这就是 Composition Root）。"""
    if not cfg.pg_dsn:
        raise ValueError(
            "缺少数据库 DSN。请在 .env 里设置 MEMORY_BRIDGE_PG_DSN"
            "（参照 .env.example）。"
        )

    store = PgMemoryStore(dsn=cfg.pg_dsn, embedding_dim=cfg.embedding_dim)

    loaded: list[str] = []
    missing: list[str] = []

    embedder, embedder_real = build_embedder(cfg)
    (loaded if embedder_real else missing).append(
        cfg.embedder_model_dir.name if embedder_real else "bge-small-zh-v1.5"
    )

    reranker, reranker_real = build_reranker(cfg)
    if cfg.rerank_enabled:
        (loaded if reranker_real else missing).append(
            cfg.reranker_model_dir.name if reranker_real else "bge-reranker-base"
        )

    retriever = HybridRetriever(
        store=store,  # HybridRetriever 需要 pg 专有方法（见 impl/hybrid.py）
        embedder=embedder,
        bm25_index=Bm25Index(),
        cfg=cfg.hybrid,
    )
    write = WritePipeline(store, embedder, cfg.policy, cfg.tier)
    recall = RecallPipeline(store, retriever, reranker, cfg.recall, cfg.tier)
    forget = ForgetPipeline(store, cfg.forget, cfg.scoring, cfg.tier)
    knowledge = KnowledgeService(store, embedder)
    context_file = ContextFileWriter(cfg.context_files)
    attributor: UsageAttributor = NoopAttributor()

    session = SessionLifecycle(
        write=write,
        recall=recall,
        store=store,
        context_file=context_file,
        forget=forget,
        tier_cfg=cfg.tier,
        forget_cfg=cfg.forget,
        reranker=reranker,
        retriever=retriever,
        attributor=attributor,
        transcript_reader=FileTranscriptReader(),
        recall_cfg_patch_threshold=cfg.recall.patch_flush_threshold,
    )

    return Pipeline(
        cfg=cfg,
        store=store,
        embedder=embedder,
        retriever=retriever,
        reranker=reranker,
        attributor=attributor,
        write=write,
        recall=recall,
        forget=forget,
        knowledge=knowledge,
        session=session,
        loaded_models=loaded,
        missing_models=missing,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_USAGE = """memory —— Memory Bridge 的命令行

  memory session-start [--cwd DIR]        输出常驻文本（并写入 CLAUDE.md 记忆区）
  memory remember "<text>"                显式写入一条记忆
  memory recall "<query>"                 输出尾部文本（给 agent 追加）
  memory session-end <transcript>         会话结束：提取 + 容量治理
  memory mark-used <id> [<id> ...]        标记「这些记忆本轮确实被用上了」
  memory flush                            把尾部补丁合并进常驻区并清空缓冲
  memory forget [--purge] [--older-than D] 容量治理 / 用户行使删除权
  memory kb-ingest <collection> <file>    知识库导入
  memory eval [--cases PATH] [--baseline PATH]  跑回归集
  memory init-db                          执行 sql/001_schema.sql
  memory health                           查常驻服务状态
"""


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--owner", default=None, help="默认取配置里的 owner")
    parser.add_argument("--cwd", default=None, help="当前工作目录（scope 过滤用）")
    parser.add_argument(
        "--no-server", action="store_true", help="不探测/拉起常驻服务，直接就地装配"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="memory", description="Memory Bridge", add_help=True
    )
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("session-start", help="输出常驻文本")
    _add_common(p)

    p = sub.add_parser("remember", help="显式写入一条记忆")
    p.add_argument("text")
    _add_common(p)

    p = sub.add_parser("recall", help="输出尾部文本")
    p.add_argument("query")
    _add_common(p)

    p = sub.add_parser("session-end", help="会话结束处理")
    p.add_argument("transcript", nargs="?", default=None)
    p.add_argument("--text", default=None, help="直接给会话文本（不给文件路径时用）")
    p.add_argument("--agent", default="claude")
    _add_common(p)

    p = sub.add_parser("mark-used", help="标记记忆确实被用上")
    p.add_argument("ids", nargs="+")
    _add_common(p)

    p = sub.add_parser("flush", help="合并尾部补丁进常驻区")
    _add_common(p)

    p = sub.add_parser("forget", help="容量治理 / 删除权")
    p.add_argument("--purge", action="store_true", help="真删除（用户行使删除权）")
    p.add_argument("--older-than", default=None, help="ISO 日期；只处理早于它的条目")
    p.add_argument("--purge-id", action="append", default=[], help="只删指定 id（可重复）")
    _add_common(p)

    p = sub.add_parser("kb-ingest", help="知识库导入")
    p.add_argument("collection")
    p.add_argument("file")
    p.add_argument("--version", default="v1")
    p.add_argument("--effective-date", default=None)
    p.add_argument("--acl", default="public")

    p = sub.add_parser("eval", help="跑回归集")
    p.add_argument("--cases", default=None)
    p.add_argument("--baseline", default=None, help="基线 JSON，用于对比")
    p.add_argument("--save-baseline", default=None)
    _add_common(p)

    sub.add_parser("init-db", help="执行 DDL")
    sub.add_parser("health", help="查服务状态")
    return parser


def _resolve_owner(args: argparse.Namespace, cfg: Config) -> str:
    owner = getattr(args, "owner", None) or cfg.owner
    if not owner.strip():
        raise SystemExit("owner 不能为空")
    return owner


def _load_env() -> None:
    """把 ``.env`` 灌进 ``os.environ``（**已存在的环境变量优先**）。

    手工解析 + 手工 setdefault，不用 python-dotenv ——
    规格第〇节的依赖清单里没有它，而「不引入新依赖」是硬约束。
    """
    from .config import REPO_ROOT

    for candidate in (Path.cwd() / ".env", REPO_ROOT / ".env"):
        for key, value in load_env_file(candidate).items():
            os.environ.setdefault(key, value)


def _load_cfg(args: argparse.Namespace) -> Config:
    _load_env()
    return load_config()


def _print(text: str) -> None:
    sys.stdout.write(text)
    if text and not text.endswith("\n"):
        sys.stdout.write("\n")
    sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s | %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    command = args.command
    if command is None:
        _print(_USAGE)
        return 0

    cfg = _load_cfg(args)

    if command == "health":
        return _cmd_health(cfg)
    if command == "init-db":
        return _cmd_init_db(cfg)

    pipeline = build_pipeline(cfg)
    try:
        return _dispatch(command, args, cfg, pipeline)
    finally:
        store = pipeline.store
        dispose = getattr(store, "dispose", None)
        if callable(dispose):
            dispose()


def _dispatch(
    command: str, args: argparse.Namespace, cfg: Config, pipeline: Pipeline
) -> int:
    owner = _resolve_owner(args, cfg)

    if command == "session-start":
        cwd = Path(args.cwd) if args.cwd else Path.cwd()
        text = pipeline.session.on_session_start(owner=owner, cwd=cwd)
        _print(text)
        return 0

    if command == "remember":
        try:
            memory = pipeline.write.remember_explicit(
                owner=owner, text=args.text, now=utc_now()
            )
        except ValueError as exc:
            # 领域失败：空文本等 —— 报错而不是抛回溯
            raise SystemExit(f"remember 失败：{exc}") from exc
        _print(json.dumps(_memory_json(memory), ensure_ascii=False))
        return 0

    if command == "recall":
        text = pipeline.session.recall_tail(owner=owner, query=args.query)
        _print(text)
        return 0

    if command == "mark-used":
        count = pipeline.session.mark_used(owner=owner, memory_ids=list(args.ids))
        _print(json.dumps({"marked": count, "ids": list(args.ids)}, ensure_ascii=False))
        return 0

    if command == "flush":
        count = pipeline.session.flush_patches(owner=owner)
        _print(json.dumps({"flushed": count}, ensure_ascii=False))
        return 0

    if command == "session-end":
        return _cmd_session_end(args, pipeline, owner)

    if command == "forget":
        return _cmd_forget(args, pipeline, owner)

    if command == "kb-ingest":
        return _cmd_kb_ingest(args, pipeline)

    if command == "eval":
        return _cmd_eval(args, pipeline, owner)

    parser_error = f"未知命令：{command}"
    raise SystemExit(parser_error)


def _memory_json(memory: object) -> dict[str, object]:
    payload = {
        "id": getattr(memory, "id", None),
        "owner": getattr(memory, "owner", None),
        "what": getattr(memory, "what", None),
        "type": getattr(getattr(memory, "type", None), "value", None),
        "tier": getattr(getattr(memory, "tier", None), "value", None),
        "importance": getattr(memory, "importance", None),
    }
    return payload


def _parse_dt(raw: str | None) -> datetime | None:
    if not raw:
        return None
    text = raw.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError as exc:
        raise SystemExit(f"--older-than 不是合法 ISO 日期：{raw!r}") from exc


def _cmd_session_end(args: argparse.Namespace, pipeline: Pipeline, owner: str) -> int:
    transcript_path: Path | None = None
    if args.text is not None:
        report = pipeline.session.on_session_end(
            owner=owner, transcript_text=args.text, now=utc_now()
        )
    else:
        if args.transcript:
            transcript_path = Path(args.transcript)
        else:
            from .bridge.transcript import find_latest_session

            transcript_path = find_latest_session(args.agent, Path.cwd())
            if transcript_path is None:
                raise SystemExit("找不到最近的会话记录，请显式给出 <transcript> 路径")
        report = pipeline.session.on_session_end(
            owner=owner, transcript_path=transcript_path, now=utc_now()
        )

    _print(
        json.dumps(
            {
                "written": [m.id for m in report.written],
                "archived": report.archived,
                "promoted": report.promoted,
                "demoted": report.demoted,
                "need_manual": report.need_manual,
                "note": report.note,
            },
            ensure_ascii=False,
        )
    )
    if report.need_manual:
        sys.stderr.write(
            "⚠️ 记忆容量超过上限且无法自动归档：请当场整合或删除若干条记忆。\n"
        )
        return 3
    return 0


def _cmd_forget(args: argparse.Namespace, pipeline: Pipeline, owner: str) -> int:
    cutoff = _parse_dt(args.older_than)

    if not args.purge and not args.purge_id:
        report = pipeline.forget.run(owner=owner, now=utc_now())
        _print(
            json.dumps(
                {
                    "scanned": report.scanned,
                    "archived": report.archived,
                    "remaining": report.remaining,
                    "need_manual": report.need_manual,
                    "reason": report.reason,
                },
                ensure_ascii=False,
            )
        )
        return 3 if report.need_manual else 0

    if args.purge_id:
        counter = getattr(pipeline.store, "purge_memories", None)
        if not callable(counter):
            raise SystemExit("当前 store 不支持按 id 删除")
        removed = int(counter(list(args.purge_id)))
    else:
        removed = pipeline.forget.purge(owner, older_than=cutoff)

    counters = getattr(pipeline.store, "count_memories", None)
    stats = counters(owner) if callable(counters) else {}
    _print(json.dumps({"deleted": removed, "remaining": stats}, ensure_ascii=False))
    return 0


def _cmd_kb_ingest(args: argparse.Namespace, pipeline: Pipeline) -> int:
    from scripts.kb_ingest import ingest_file

    path = Path(args.file)
    if not path.exists():
        raise SystemExit(f"文件不存在：{path}")
    effective = _parse_dt(args.effective_date)
    acl = [item.strip() for item in args.acl.split(",") if item.strip()]
    count = ingest_file(
        pipeline,
        collection=args.collection,
        path=path,
        doc_version=args.version,
        effective_date=(effective.date() if effective else utc_now().date()),
        acl=acl,
    )
    counters = getattr(pipeline.store, "count_chunks", None)
    stats = counters(str(path)) if callable(counters) else {}
    _print(json.dumps({"chunks": count, "table": stats}, ensure_ascii=False))
    return 0


def _cmd_eval(args: argparse.Namespace, pipeline: Pipeline, owner: str) -> int:
    from .domain.models import RecalledContext  # noqa: F401 - 保持类型可见性

    cases_path = Path(args.cases) if args.cases else Path("eval/cases.jsonl")
    if not cases_path.exists():
        raise SystemExit(f"回归集不存在：{cases_path}")

    final_k = pipeline.cfg.recall_final_k

    def rank_fn(query: str) -> list[str]:
        ctx = pipeline.recall.recall(owner=owner, query=query, now=utc_now())
        return [m.id for m in ctx.retrieved][:final_k]

    def resolve_owners(ids: Sequence[str]) -> list[str]:
        """把召回到的 id 查回 owner —— 污染率必须**真的算**，不能手填。"""
        getter = getattr(pipeline.store, "get_memory", None)
        if not callable(getter):
            return []
        owners: list[str] = []
        for memory_id in ids:
            memory = getter(memory_id)
            if memory is not None:
                owners.append(str(memory.owner))
        return owners

    def version_ranker(case: dict[str, object]) -> str:
        """版本冲突用例：走知识域召回 + 择新说明，人工看结论。"""
        collection = str(case.get("collection", "kb"))
        docs = pipeline.recall.recall_knowledge(
            query=str(case["query"]), collection=collection, acl=["public"], k=5
        )
        if not docs:
            return f"{case['query']} → 知识域没召回到任何版本（无法判断择新）"
        rendered = pipeline.knowledge.resolve_version_conflict(docs)
        versions = sorted({d.doc_version or "未标注" for d in docs})
        return (
            f"{case['query']} → 召回 {len(docs)} 个父块，版本 {versions}；"
            f"{rendered.splitlines()[0] if rendered else '单版本，无需择新'}"
        )

    report = run_regression(
        cases_path,
        rank_fn=rank_fn,
        resolve_owners=resolve_owners,
        version_ranker=version_ranker,
    )
    payload = report.as_dict()

    if args.baseline:
        baseline_path = Path(args.baseline)
        if baseline_path.exists():
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
            payload["diff_vs_baseline"] = report.diff(baseline)
    if args.save_baseline:
        Path(args.save_baseline).write_text(
            json.dumps(report.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    _print(json.dumps(payload, ensure_ascii=False, indent=2))
    # 安全指标不混进平均分：污染率不为 0 直接失败
    return 1 if report.contamination_rate > 0 else 0


def _cmd_health(cfg: Config) -> int:
    from .bridge.server import MemoryBridgeService  # noqa: F401 - 类型可见性
    from .bridge.server import _health_ok  # noqa: PLC2701

    ok = _health_ok(cfg.host, cfg.port)
    _print(json.dumps({"running": ok, "url": f"http://{cfg.host}:{cfg.port}"}))
    return 0 if ok else 1


def _cmd_init_db(cfg: Config) -> int:
    store = PgMemoryStore(dsn=cfg.pg_dsn, embedding_dim=cfg.embedding_dim)
    try:
        store.apply_schema(cfg.schema_path)
    finally:
        store.dispose()
    _print(json.dumps({"applied": str(cfg.schema_path)}, ensure_ascii=False))
    return 0


def entrypoint() -> None:
    """``pyproject.toml`` 里的 console script 入口。"""
    raise SystemExit(main())


if __name__ == "__main__":
    raise SystemExit(main())
