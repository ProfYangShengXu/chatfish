"""L3 桥：常驻进程。

持有模型、提供本地 HTTP 接口。**它存在的唯一理由是「模型加载慢」。**

不做什么：
  ❌ 不监听 0.0.0.0（**记忆是隐私数据，只走本机回环**）
  ❌ 不做鉴权（回环 + 单用户 —— README 里已写明这一点）
"""

from __future__ import annotations

import json
import logging
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy.exc import IntegrityError

from ..config import Config
from ..domain.clock import utc_now

if TYPE_CHECKING:  # pragma: no cover
    from ..cli import Pipeline

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})

logger = logging.getLogger("memory_bridge.server")


class MemoryBridgeService:
    """把已装配好的管道包成服务。**不认识 FastAPI**，便于单测。"""

    def __init__(self, pipeline: Pipeline, cfg: Config) -> None:
        self._pipeline = pipeline
        self._cfg = cfg
        self._started_at = time.time()

    @property
    def pipeline(self) -> Pipeline:
        return self._pipeline

    def health(self) -> dict[str, Any]:
        """★ 真探活：**真的碰一次数据库**，不是返回字面量 ``ok: true``。

        规格写的是 ``-> {"ok": true, "models": [...]}``；一个恒真的 /health
        会让「PG 挂了」这种故障完全不可见。所以这里加上 ``db`` 与 ``models`` 的真实状态。
        """
        db_ok = False
        db_error = ""
        try:
            self._pipeline.store.ping()  # type: ignore[attr-defined]
            db_ok = True
        except Exception as exc:  # noqa: BLE001 - 探活要把失败原因带出来
            db_error = f"{type(exc).__name__}"

        embedder_missing = list(self._pipeline.missing_models)
        return {
            "ok": db_ok and not embedder_missing,
            "db": db_ok,
            "db_error": db_error,
            "models": list(self._pipeline.loaded_models),
            "models_missing": embedder_missing,
            "rerank_enabled": self._cfg.rerank_enabled,
            "uptime_s": round(time.time() - self._started_at, 1),
        }

    def recall(self, *, owner: str, query: str, scope: str | None = None) -> str:
        return self._pipeline.session.recall_tail(owner=owner, query=query)

    def remember(self, *, owner: str, text: str, now: datetime | None = None) -> str:
        memory = self._pipeline.write.remember_explicit(
            owner=owner, text=text, now=now or utc_now()
        )
        return memory.id

    def session_start(self, *, owner: str, cwd: str | None = None) -> str:
        path = Path(cwd) if cwd else None
        return self._pipeline.session.on_session_start(owner=owner, cwd=path)

    def session_end(self, *, owner: str, transcript: str) -> dict[str, Any]:
        report = self._pipeline.session.on_session_end(
            owner=owner, transcript_text=transcript
        )
        return {
            "written": [m.id for m in report.written],
            "archived": report.archived,
            "promoted": report.promoted,
            "demoted": report.demoted,
            "need_manual": report.need_manual,
            "note": report.note,
        }


class RecallBody(BaseModel):
    owner: str
    query: str
    scope: str | None = None


class RememberBody(BaseModel):
    owner: str
    text: str


class SessionStartBody(BaseModel):
    owner: str
    cwd: str | None = None


class SessionEndBody(BaseModel):
    owner: str
    transcript: str


def create_app(service: MemoryBridgeService) -> Any:
    """构建 FastAPI 应用。

    ★ 绑定检查写在最前面：**只允许回环地址**。
      非回环直接在启动时抛错，而不是等到监听之后 —— 隐私数据不出本机是硬约束。

    ⚠️ 请求体模型定义在**模块级**，不在 ``create_app`` 里面。
       本文件有 ``from __future__ import annotations``，注解在运行时是字符串；
       如果把模型类定义在函数内、又用 ``Body(...)`` 当默认值，
       Pydantic 解析不到那个前向引用，会在**第一次请求时**抛
       ``PydanticUserError: not fully defined``（实测踩过）。
       模块级定义让类在模块加载时就已存在，问题消失。
    """
    cfg = service._cfg  # noqa: SLF001 - 同包内的装配细节
    if cfg.host not in LOOPBACK_HOSTS:
        raise ValueError(
            f"拒绝绑定非回环地址 {cfg.host!r}：记忆是隐私数据，只允许 127.0.0.1 / ::1"
        )

    app = FastAPI(title="memory-bridge", version="0.1.0", docs_url=None, redoc_url=None)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return service.health()

    @app.post("/recall")
    def recall(body: RecallBody = Body(...)) -> dict[str, str]:
        return {
            "text": service.recall(owner=body.owner, query=body.query, scope=body.scope)
        }

    @app.post("/remember")
    def remember(body: RememberBody = Body(...)) -> dict[str, str]:
        try:
            memory_id = service.remember(owner=body.owner, text=body.text)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except IntegrityError as exc:
            # 并发写入撞上部分唯一索引 —— 是客户端可理解的冲突，不是 500
            raise HTTPException(status_code=409, detail="该事实已存在") from exc
        return {"id": memory_id}

    @app.post("/session/start")
    def session_start(body: SessionStartBody = Body(...)) -> dict[str, str]:
        return {"text": service.session_start(owner=body.owner, cwd=body.cwd)}

    @app.post("/session/end")
    def session_end(body: SessionEndBody = Body(...)) -> dict[str, Any]:
        return service.session_end(owner=body.owner, transcript=body.transcript)

    return app


def _port_open(host: str, port: int, timeout: float = 0.4) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


def _health_ok(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(  # noqa: S310 - 固定回环地址
            f"http://{host}:{port}/health", timeout=timeout
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, json.JSONDecodeError, TimeoutError):
        return False
    return bool(payload.get("ok"))


def ensure_running(cfg: Config, *, wait_s: float = 120.0) -> bool:
    """CLI 调用前先探测：端口没响应就后台拉起（类似 LSP 客户端的做法）。

    ⚠️ 用**文件锁**串行化「冷启动」：
       没有锁时并发两条 CLI 会各拉起一个 server（各载 ~1.2GB 模型），
       第二个绑端口失败且 stderr 无处可去 → 静默失败。
    """
    if _health_ok(cfg.host, cfg.port):
        return True

    lock_path = cfg.data_dir / "server.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with _file_lock(lock_path):
        # 拿锁之后再探一次：可能别的进程刚刚拉起来了
        if _health_ok(cfg.host, cfg.port):
            return True

        log_path = cfg.data_dir / "server.log"
        with log_path.open("ab") as log_handle:
            subprocess.Popen(  # noqa: S603 - 固定模块路径，无用户输入
                [sys.executable, "-m", "memory_bridge.bridge.server"],
                stdout=log_handle,
                stderr=log_handle,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )

        deadline = time.time() + wait_s
        while time.time() < deadline:
            if _health_ok(cfg.host, cfg.port):
                return True
            time.sleep(0.5)

    logger.warning("服务在 %.0fs 内没有就绪，详见 %s", wait_s, log_path)
    return False


class _file_lock:
    """极简的跨进程文件锁（``fcntl.flock``）。

    ⚠️ 用 ``flock`` 而不是「创建锁文件」—— 后者在进程崩溃后会留下死锁文件。
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._handle: Any = None

    def __enter__(self) -> _file_lock:
        import fcntl

        self._handle = self._path.open("a+")
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc: object) -> None:
        import fcntl

        if self._handle is not None:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            self._handle.close()
            self._handle = None


def run(cfg: Config) -> int:
    """启动常驻服务（前台，供 ``python -m memory_bridge.bridge.server`` 用）。"""
    import uvicorn

    from ..cli import build_pipeline

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    if cfg.host not in LOOPBACK_HOSTS:
        logger.error("拒绝绑定非回环地址 %s", cfg.host)
        return 2

    logger.info("memory-bridge 启动中（models_dir=%s）", cfg.models_dir)
    pipeline = build_pipeline(cfg)

    if pipeline.missing_models:
        logger.warning(
            "以下模型未加载（缺失文件）：%s —— 相关能力不可用",
            ", ".join(pipeline.missing_models),
        )
    if pipeline.loaded_models:
        logger.info("模型加载成功：%s", ", ".join(pipeline.loaded_models))
    else:
        logger.warning("没有任何模型被加载")

    service = MemoryBridgeService(pipeline, cfg)
    app = create_app(service)

    logger.info("监听 http://%s:%d （仅回环）", cfg.host, cfg.port)
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="info", workers=1)
    return 0


def _install_signal_handlers() -> None:
    def _handler(signum: int, _frame: object) -> None:
        logger.info("收到信号 %d，退出", signum)
        raise SystemExit(0)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handler)
        except ValueError:  # pragma: no cover - 非主线程
            pass


def main() -> int:
    from ..cli import _load_env
    from ..config import load_config

    _load_env()
    cfg = load_config()
    _install_signal_handlers()
    return run(cfg)


if __name__ == "__main__":
    raise SystemExit(main())


# 供测试用：确认没有把 0.0.0.0 写进任何默认值
def is_loopback(host: str) -> bool:
    return host in LOOPBACK_HOSTS


