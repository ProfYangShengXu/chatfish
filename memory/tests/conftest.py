"""测试全局夹具。

⭐ 这里解决一个【静默失效】：tests/test_impl.py 读的是
   ``MEMORY_BRIDGE_TEST_DSN``，而 .env 里存的是 ``MEMORY_BRIDGE_PG_DSN`` ——
   名字不同、且 .env 是 ``KEY=a b c``（带空格）格式，shell 的 source 解析不了。
   结果：42 个真连库用例【默认全部 skip】，跑出来像"全绿"。

   本文件在收集阶段就用【应用自己的】 load_env_file 加载 .env，
   并把 PG_DSN 映射成 TEST_DSN —— 让那 42 个默认就能跑。

   如果 PostgreSQL 不可达，就【主动清掉】 TEST_DSN，让它们退回 skip
   （而不是 fail）—— 因为本地没起库是正常情况，不该让 CI 变红。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _bootstrap_dsn() -> None:
    """用应用自己的 loader 把 .env 喂进 os.environ。"""
    try:
        from memory_bridge.config import load_env_file
    except Exception:
        return

    env_file = ROOT / ".env"
    if not env_file.exists():
        return

    try:
        env = load_env_file(env_file)
    except Exception:
        return

    # ⚠️ 用 setdefault：显式传给 pytest 的环境变量优先，不要被 .env 覆盖
    for k, v in env.items():
        os.environ.setdefault(k, v)

    if not os.environ.get("MEMORY_BRIDGE_TEST_DSN"):
        dsn = os.environ.get("MEMORY_BRIDGE_PG_DSN") or env.get("MEMORY_BRIDGE_PG_DSN")
        if dsn:
            os.environ["MEMORY_BRIDGE_TEST_DSN"] = dsn


def _probe_db() -> bool:
    """真连一下。连不上就返回 False（让真连库用例退回 skip）。"""
    dsn = os.environ.get("MEMORY_BRIDGE_TEST_DSN", "")
    if not dsn:
        return False
    try:
        import psycopg

        with psycopg.connect(dsn, connect_timeout=3):
            return True
    except Exception:
        return False


_bootstrap_dsn()

if not _probe_db():
    # ⭐ 关键：连不上就【撤掉】变量，让 requires_db 的测试 skip 而不是 fail。
    #    否则本地没起 PG 时整个测试套会红，掩盖真正的失败。
    os.environ.pop("MEMORY_BRIDGE_TEST_DSN", None)
