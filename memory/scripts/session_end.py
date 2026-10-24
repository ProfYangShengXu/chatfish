#!/usr/bin/env python3
"""会话结束钩子 —— **不依赖模型配合的兜底触发**。

两种部署方式：

① **agent 原生 hook**（如 Claude Code 的 hooks）→ 直接调用本脚本::

       python scripts/session_end.py --transcript /path/to/session.jsonl

② **无 hook 时** → 由 cron / 后台进程定期扫描 transcript 目录::

       python scripts/session_end.py --scan          # 扫一次，处理「已结束」的会话

   判定「会话已结束」：文件 **mtime 超过 N 分钟** 且 **非空**。
   处理过的文件记进 ``<data_dir>/processed.json``，避免重复提取。

⚠️ 已知取舍：mtime 阈值会把「用户中途思考超过 N 分钟」误判成会话结束，
   于是会话尾部可能被漏掉。这里用两个手段缓解：
   - 默认阈值取 **30 分钟**（而不是 5 分钟），宁可晚提取也不早提取；
   - 提取**不修改** transcript，且在 ``--scan`` 模式下反复扫描时
     「已处理」记录里存的是文件的 ``(mtime, size)`` —— 文件**继续增长**时会被重新处理。
   这一条已列入汇报的待确认项。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_IDLE_MINUTES = 30
DEFAULT_AGENTS = ("claude", "dsh", "codex")


def _load_processed(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _save_processed(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _signature(path: Path) -> str:
    """文件的「内容签名」：mtime + size。

    用它而不是只记路径 —— 文件继续增长时签名会变，于是会被重新提取，
    补上「会话其实还没结束」漏掉的那段。
    """
    stat = path.stat()
    return f"{int(stat.st_mtime)}:{stat.st_size}"


def process_one(
    pipeline: Any,
    *,
    owner: str,
    transcript_path: Path,
    now: datetime | None = None,
) -> dict[str, Any]:
    from memory_bridge.domain.clock import utc_now

    report = pipeline.session.on_session_end(
        owner=owner,
        transcript_path=transcript_path,
        now=now or utc_now(),
    )
    return {
        "transcript": str(transcript_path),
        "written": [m.id for m in report.written],
        "archived": report.archived,
        "promoted": report.promoted,
        "demoted": report.demoted,
        "need_manual": report.need_manual,
        "note": report.note,
    }


def scan_once(
    pipeline: Any,
    *,
    owner: str,
    agents: tuple[str, ...],
    idle_minutes: int,
    processed_path: Path,
    cwd: Path | None = None,
) -> list[dict[str, Any]]:
    """扫一次 transcript 目录，处理「已结束」的会话。"""
    from memory_bridge.bridge.transcript import find_latest_session

    processed = _load_processed(processed_path)
    results: list[dict[str, Any]] = []
    cutoff = time.time() - idle_minutes * 60

    for agent in agents:
        candidate = find_latest_session(agent, cwd or Path.cwd())
        if candidate is None or not candidate.exists():
            continue
        try:
            stat = candidate.stat()
        except OSError:
            continue
        if stat.st_size == 0:
            continue
        if stat.st_mtime > cutoff:
            # 还在写 → 判定会话未结束
            continue
        signature = _signature(candidate)
        if processed.get(str(candidate)) == signature:
            continue
        results.append(process_one(pipeline, owner=owner, transcript_path=candidate))
        processed[str(candidate)] = signature

    if results:
        _save_processed(processed_path, processed)
    return results


def main(argv: list[str] | None = None) -> int:
    from memory_bridge.cli import build_pipeline
    from memory_bridge.config import load_config, load_env_file

    parser = argparse.ArgumentParser(description="会话结束钩子")
    parser.add_argument("--transcript", default=None, help="transcript 文件路径")
    parser.add_argument("--scan", action="store_true", help="扫描模式（cron 用）")
    parser.add_argument("--agent", default="claude", choices=[*DEFAULT_AGENTS, "all"])
    parser.add_argument("--idle-minutes", type=int, default=DEFAULT_IDLE_MINUTES)
    parser.add_argument("--owner", default=None)
    parser.add_argument("--stdin", action="store_true", help="从 stdin 读会话文本")
    args = parser.parse_args(argv)

    load_env_file(REPO_ROOT / ".env")
    cfg = load_config()
    owner = args.owner or cfg.owner

    # ★ cron 场景：把日志落到文件，否则失败完全不可见
    log_path = cfg.data_dir / "session_end.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("a", encoding="utf-8")

    try:
        pipeline = build_pipeline(cfg)
    except Exception as exc:  # noqa: BLE001 - 钩子失败必须留痕
        log_handle.write(
            f"{datetime.now().isoformat()} FATAL build_pipeline: "
            f"{type(exc).__name__}: {exc}\n"
        )
        log_handle.close()
        print(f"装配失败：{exc}", file=sys.stderr)
        return 1

    try:
        if args.stdin:
            text = sys.stdin.read()
            report = pipeline.session.on_session_end(owner=owner, transcript_text=text)
            payload: Any = {"written": [m.id for m in report.written]}
            results = [payload]
        elif args.scan:
            agents = DEFAULT_AGENTS if args.agent == "all" else (args.agent,)
            results = scan_once(
                pipeline,
                owner=owner,
                agents=agents,
                idle_minutes=args.idle_minutes,
                processed_path=cfg.processed_path,
            )
        else:
            if not args.transcript:
                parser.error("需要 --transcript / --scan / --stdin 之一")
            results = [
                process_one(
                    pipeline, owner=owner, transcript_path=Path(args.transcript)
                )
            ]
    except Exception as exc:  # noqa: BLE001 - 钩子失败必须留痕
        log_handle.write(
            f"{datetime.now().isoformat()} ERROR: {type(exc).__name__}: {exc}\n"
        )
        log_handle.close()
        print(f"处理失败：{exc}", file=sys.stderr)
        return 1

    line = json.dumps(results, ensure_ascii=False)
    log_handle.write(f"{datetime.now().isoformat()} OK: {line}\n")
    log_handle.close()
    print(line)

    if any(item.get("need_manual") for item in results):
        print(
            "⚠️ 记忆容量超过上限且无法自动归档：请当场整合或删除若干条记忆。",
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
