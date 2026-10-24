"""装配层：配置加载与默认值。

**没有任何导入副作用** —— ``import memory_bridge`` 不读文件、不连库、不载模型。

配置来源（优先级从低到高）：

1. 本文件里的默认值
2. ``config.yaml``（``MEMORY_BRIDGE_CONFIG`` 指向，默认 ``<repo>/config.yaml``）
3. 环境变量（含 ``.env``，由 ``load_env_file`` 显式加载）
   - ``MEMORY_BRIDGE_PG_DSN``   ← 必填，带密码（host 认证）
   - ``MEMORY_BRIDGE_MODELS_DIR``
   - ``MEMORY_BRIDGE_DATA_DIR``
   - ``MEMORY_BRIDGE_OWNER``
   - ``MEMORY_BRIDGE_CONTEXT_FILES``（os.pathsep 分隔）
   - ``MEMORY_BRIDGE_PORT``
   - ``MEMORY_BRIDGE_RERANK_ENABLED``

⚠️ 密码只存在于 ``.env`` 与 ``pg_dsn`` 里；``Config`` 覆写了 ``__repr__`` 做掩码，
   避免 ``print(cfg)`` / 异常回溯把 DSN 里的密码带进日志或汇报。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Union

from .domain.models import (
    ForgetConfig,
    HybridConfig,
    PolicyConfig,
    RecallConfig,
    ScoringConfig,
    TierConfig,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

_SUB_CONFIG_TYPES: dict[str, type[Any]] = {
    "tier": TierConfig,
    "scoring": ScoringConfig,
    "policy": PolicyConfig,
    "recall": RecallConfig,
    "forget": ForgetConfig,
    "hybrid": HybridConfig,
}


@dataclass(slots=True)
class Config:
    """全部配置。``pg_dsn`` 必须带密码（WSL 的 host 认证走 scram-sha-256）。"""

    pg_dsn: str = ""
    models_dir: Path = field(default_factory=lambda: REPO_ROOT / "models")
    data_dir: Path = field(default_factory=lambda: REPO_ROOT / "var")
    owner: str = "user:local"
    context_files: list[Path] = field(default_factory=list)
    transcript_roots: list[Path] = field(default_factory=list)

    # —— 召回 ——
    recall_k: int = 50
    recall_final_k: int = 5
    rerank_enabled: bool = True
    rerank_max_candidates: int = 50

    # —— 服务 ——
    host: str = "127.0.0.1"  # ★ 只绑回环
    port: int = 8765

    # —— 子配置 ——
    tier: TierConfig = field(default_factory=TierConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    recall: RecallConfig = field(default_factory=RecallConfig)
    forget: ForgetConfig = field(default_factory=ForgetConfig)
    hybrid: HybridConfig = field(default_factory=HybridConfig)

    # —— 派生路径 ——
    @property
    def sql_dir(self) -> Path:
        return REPO_ROOT / "sql"

    @property
    def schema_path(self) -> Path:
        return self.sql_dir / "001_schema.sql"

    @property
    def bm25_cache_path(self) -> Path:
        return self.data_dir / "bm25_cache.json"

    @property
    def processed_path(self) -> Path:
        return self.data_dir / "processed.json"

    @property
    def embedding_dim(self) -> int:
        """bge-small-zh-v1.5 的输出维度（结构常量，不是可调项）。"""
        return 512

    @property
    def embedder_model_dir(self) -> Path:
        return self.models_dir / "bge-small-zh-v1.5"

    @property
    def reranker_model_dir(self) -> Path:
        return self.models_dir / "bge-reranker-base"

    def __repr__(self) -> str:
        """掩码 DSN 里的密码 —— 密码不得出现在日志 / 汇报 / 回溯里。"""
        return (
            f"Config(pg_dsn={mask_dsn(self.pg_dsn)!r}, models_dir={str(self.models_dir)!r}, "
            f"data_dir={str(self.data_dir)!r}, owner={self.owner!r}, "
            f"context_files={[str(p) for p in self.context_files]!r}, "
            f"rerank_enabled={self.rerank_enabled!r}, port={self.port!r})"
        )

    __str__ = __repr__


_PASSWORD_RE = re.compile(r"(password\s*=\s*)(\S+)", re.IGNORECASE)


def mask_dsn(dsn: str) -> str:
    """把 DSN 里的密码换成 ``***``。"""
    masked = _PASSWORD_RE.sub(r"\1***", dsn)
    # 兼容 postgresql://user:pw@host/db 形式
    return re.sub(r"(://[^:/@\s]+:)[^@/\s]+(@)", r"\1***\2", masked)


def _parse_scalar(raw: str) -> Union[str, bool, int, float]:
    value = raw.strip().strip('"').strip("'")
    lowered = value.lower()
    if lowered in ("true", "yes", "on"):
        return True
    if lowered in ("false", "no", "off"):
        return False
    if lowered in ("null", "none", "~"):
        return ""
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def parse_flat_yaml(text: str) -> dict[str, Any]:
    """解析本项目用到的那一小撮 YAML 子集。

    支持：``key: value``、``section:`` + 缩进的 ``key: value``、``#`` 注释、
    和 ``- item`` 列表。不支持锚点 / 多行标量 / 流式集合 —— 遇到就抛错，
    **不静默忽略**（静默忽略会让人以为配置生效了）。

    实现要点：``key:`` 后面到底是「映射」还是「列表」，只能由**下一行**决定
    （``- `` 开头 → 列表）。所以空值键先挂一个哨兵，等下一行再物化。
    """
    result: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any] | list[Any]]] = [(-1, result)]
    # 已遇到、但还不知道是映射还是列表的键
    pending: tuple[dict[str, Any], str, int] | None = None

    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())

        if stripped.startswith("- "):
            if pending is not None:
                # 上一行的空值键在此刻确定是【列表】
                parent, key, key_indent = pending
                items: list[Any] = []
                parent[key] = items
                stack.append((key_indent, items))
                pending = None
            # 连续列表项：栈顶就是那个列表
            while len(stack) > 1 and indent < stack[-1][0]:
                stack.pop()
            if not isinstance(stack[-1][1], list):
                raise ValueError(
                    f"config.yaml:{lineno}: 列表项没有归属的键：{stripped!r}"
                )
            stack[-1][1].append(_parse_scalar(stripped[2:]))
            continue

        if ":" not in stripped:
            raise ValueError(f"config.yaml:{lineno}: 不是 key: value 形式：{stripped!r}")

        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        container = stack[-1][1]
        if not isinstance(container, dict):
            raise ValueError(f"config.yaml:{lineno}: 缩进层级不明：{stripped!r}")

        # 上一行的空值键在此刻物化：有缩进 → 映射
        if pending is not None:
            parent, key, key_indent = pending
            if indent <= key_indent:
                raise ValueError(
                    f"config.yaml:{lineno}: 键 {key!r} 声明为空但下面没有缩进内容"
                )
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((key_indent, child))
            container = child
            pending = None

        key, _, raw_value = stripped.partition(":")
        key = key.strip()
        raw_value = raw_value.strip()

        if raw_value:
            container[key] = _parse_scalar(raw_value)
        else:
            # 物化推迟到下一行（可能是映射，也可能是列表）
            container[key] = None
            pending = (container, key, indent)

    if pending is not None:
        _, key, _ = pending
        raise ValueError(f"config.yaml: 键 {key!r} 声明为空但下面没有内容")

    return result


def load_env_file(path: Path) -> dict[str, str]:
    """读 ``.env``（``KEY=VALUE``，支持 ``#`` 注释与引号）。

    手工解析而不是依赖 python-dotenv —— 规格第〇节的依赖清单里没有它，
    而「不引入新依赖」是硬约束。格式够简单，手写反而少一个依赖。
    """
    if not path.exists():
        return {}
    env: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def _coerce_sub_config(name: str, data: dict[str, Any]) -> Any:
    cls = _SUB_CONFIG_TYPES[name]
    allowed = {f.name for f in fields(cls)}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"config.yaml: [{name}] 里有未知字段：{sorted(unknown)}")
    return cls(**data)


def load_config(
    *,
    config_path: Path | None = None,
    env: dict[str, str] | None = None,
) -> Config:
    """装配出 ``Config``。**不连库、不载模型。**"""
    env_map = dict(os.environ)
    env_map.update(env or {})

    explicit = config_path or (
        Path(env_map["MEMORY_BRIDGE_CONFIG"])
        if env_map.get("MEMORY_BRIDGE_CONFIG")
        else REPO_ROOT / "config.yaml"
    )

    raw: dict[str, Any] = {}
    if explicit.exists():
        raw = parse_flat_yaml(explicit.read_text(encoding="utf-8"))

    sub_configs: dict[str, Any] = {}
    for name in _SUB_CONFIG_TYPES:
        section = raw.pop(name, None)
        if section is None:
            sub_configs[name] = _SUB_CONFIG_TYPES[name]()
            continue
        if not isinstance(section, dict):
            raise ValueError(f"config.yaml: [{name}] 必须是一个映射，实际是 {type(section)}")
        sub_configs[name] = _coerce_sub_config(name, section)

    data_dir = Path(
        str(
            env_map.get("MEMORY_BRIDGE_DATA_DIR")
            or raw.pop("data_dir", None)
            or REPO_ROOT / "var"
        )
    )
    models_dir = Path(
        str(
            env_map.get("MEMORY_BRIDGE_MODELS_DIR")
            or raw.pop("models_dir", None)
            or data_dir / "models"
        )
    )

    context_files_raw = env_map.get("MEMORY_BRIDGE_CONTEXT_FILES")
    if context_files_raw:
        context_files = [
            Path(p) for p in context_files_raw.split(os.pathsep) if p.strip()
        ]
    elif "context_files" in raw:
        value = raw.pop("context_files")
        if not isinstance(value, list):
            raise ValueError("config.yaml: context_files 必须是列表")
        context_files = [Path(str(p)) for p in value]
    else:
        context_files = []

    transcript_roots_raw = raw.pop("transcript_roots", None)
    transcript_roots = (
        [Path(str(p)) for p in transcript_roots_raw]
        if isinstance(transcript_roots_raw, list)
        else []
    )

    recall_k = int(
        env_map.get("MEMORY_BRIDGE_RECALL_K")
        or raw.pop("recall_k", sub_configs["recall"].recall_k)
    )
    recall_final_k = int(
        env_map.get("MEMORY_BRIDGE_RECALL_FINAL_K")
        or raw.pop("recall_final_k", sub_configs["recall"].recall_final_k)
    )
    rerank_enabled = _as_bool(
        env_map.get("MEMORY_BRIDGE_RERANK_ENABLED"),
        default=bool(raw.pop("rerank_enabled", True)),
    )
    rerank_max_candidates = int(
        env_map.get("MEMORY_BRIDGE_RERANK_MAX_CANDIDATES")
        or raw.pop("rerank_max_candidates", sub_configs["recall"].rerank_max_candidates)
    )

    # 把扁平字段回灌进子配置，保证两处口径一致
    sub_configs["recall"].recall_k = recall_k
    sub_configs["recall"].recall_final_k = recall_final_k
    sub_configs["recall"].rerank_max_candidates = rerank_max_candidates

    cfg = Config(
        pg_dsn=str(
            env_map.get("MEMORY_BRIDGE_PG_DSN") or raw.pop("pg_dsn", "") or ""
        ),
        models_dir=models_dir,
        data_dir=data_dir,
        owner=str(
            env_map.get("MEMORY_BRIDGE_OWNER")
            or raw.pop("owner", None)
            or "user:local"
        ),
        context_files=context_files,
        transcript_roots=transcript_roots,
        recall_k=recall_k,
        recall_final_k=recall_final_k,
        rerank_enabled=rerank_enabled,
        rerank_max_candidates=rerank_max_candidates,
        host=str(raw.pop("host", "127.0.0.1")),
        port=int(env_map.get("MEMORY_BRIDGE_PORT") or raw.pop("port", 8765)),
        tier=sub_configs["tier"],
        scoring=sub_configs["scoring"],
        policy=sub_configs["policy"],
        recall=sub_configs["recall"],
        forget=sub_configs["forget"],
        hybrid=sub_configs["hybrid"],
    )

    if raw:
        raise ValueError(f"config.yaml 里有未消费的顶层字段：{sorted(raw)}")
    return cfg


def _as_bool(raw: str | None, *, default: bool) -> bool:
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


__all__ = [
    "Config",
    "REPO_ROOT",
    "load_config",
    "load_env_file",
    "mask_dsn",
    "parse_flat_yaml",
]
