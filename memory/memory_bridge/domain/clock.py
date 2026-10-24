"""L1 时间约定：**全项目统一用 naive-UTC 表示瞬时时刻**。

为什么必须有一条硬约定：

* PostgreSQL 的 ``TIMESTAMP WITHOUT TIME ZONE`` 在 psycopg 里读回来是 **naive**；
* 而 ``datetime.now(timezone.utc)`` 造出来是 **aware**；
* 两者相减直接抛 ``TypeError`` —— 单测（naive）全绿、真连库时炸。

所以：**写入前一律 ``to_naive_utc()``，读出来一律当 naive-UTC 解释。**
带时区语义的「本地日期」需求（``scrub.absolutize_time``）另行用 ``astimezone()`` 处理，
不走这条约定。
"""

from __future__ import annotations

from datetime import datetime, timezone

__all__ = ["to_naive_utc", "utc_now", "from_naive_utc"]


def to_naive_utc(value: datetime) -> datetime:
    """把 aware 转成 UTC 的 naive；naive 原样返回（约定上它已经是 UTC）。"""
    if value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def from_naive_utc(value: datetime) -> datetime:
    """naive → aware UTC（需要跨系统传递时用）。"""
    if value.tzinfo is not None:
        return value
    return value.replace(tzinfo=timezone.utc)


def utc_now() -> datetime:
    """当前时刻，naive-UTC。"""
    return datetime.now(timezone.utc).replace(tzinfo=None)
