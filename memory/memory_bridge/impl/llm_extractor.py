"""会话结束时的 LLM 事实提炼。

为什么有这个文件
----------------
原始实现用**规则抽取**兜底（抓「决定 / 采用 / 最终 / 结论」这类词，
外加 ``len(line) <= 120`` 就收）。原因写在 ``service/write.py`` 的注释里：
规格的依赖清单里没有 LLM 客户端。

实测后果（2026-10-23 第一次跑通抽取链路时验证）：
抽出来的全是对话原文切片，连机器人自己的台词都被当成「关于用户的事实」
入库 —— 例如 ``"assistant： 喵？ 你搁这学猫叫呢 怎么，有事说事"``。
这种「记忆」召回时不会提供任何事实，只会污染上下文。

本文件补上真提炼：把 transcript 交给模型，只让它输出
「关于用户、且以后还用得上的事实」，必须是陈述句。

没有配置 key 时 ``extract`` 返 ``None``，调用方回退到原有规则抽取，
不改变任何既有行为。
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import urllib.error
import urllib.request

# 默认从 Hermes 的 secrets 目录读 key（本项目与 Hermes 同机部署）
_DEFAULT_ENV_FILE = pathlib.Path.home() / "AppData/Local/hermes/secrets/deepseek.env"

PROMPT = """从下面这段聊天记录里，提取关于【主人/用户】的、以后还会用到的**事实**。

只提取三类：
1. 偏好：他喜欢什么、讨厌什么、说话方式、审美取向
2. 决定：他已经拍板的事（用什么技术、怎么做、明确不做什么）
3. 环境：他的机器、工具链、账号类型等稳定客观信息

硬性要求：
- 只写关于**用户**的内容，绝不写机器人自己说过的话、做过的事
- 每条一句话，陈述句，20 到 40 字，不要复述对话原话
- 只要以后还用得上的；一次性的寒暄、情绪、临时状态一律不要
- 实在没有可提取的，返回空数组
- 最多 5 条，宁少勿滥

输出格式：严格的 JSON 数组，例如
["用户偏好用 uv 管理 Python 依赖", "用户决定记忆桥用 PostgreSQL 而不是 SQLite"]
不要输出任何其他文字，不要加 Markdown 代码块。"""


def _read_env_file(path: pathlib.Path) -> dict[str, str] | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def _credentials() -> tuple[str, str] | None:
    base = os.environ.get("DEEPSEEK_BASE_URL")
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not (base and key):
        env_path = pathlib.Path(
            os.environ.get("MEMORY_BRIDGE_DEEPSEEK_ENV", str(_DEFAULT_ENV_FILE))
        )
        env = _read_env_file(env_path)
        if env:
            base = base or env.get("DEEPSEEK_BASE_URL")
            key = key or env.get("DEEPSEEK_API_KEY")
    if not (base and key):
        return None
    return base.rstrip("/"), key


def _post_chat(base: str, key: str, prompt: str, model: str, timeout: float) -> str:
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 600,
            # DeepSeek 的推理模型默认会思考，思考模式下 content 为空，这里必须关掉
            "thinking": {"type": "disabled"},
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"] or ""


def _parse_facts(raw: str) -> list[str]:
    text = raw.strip()
    # 容忍模型加 ```json 围栏
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", text, re.S)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
    if not isinstance(data, list):
        return []
    facts = []
    for item in data:
        if not isinstance(item, str):
            continue
        s = item.strip().strip("-•*").strip()
        if 4 <= len(s) <= 200:
            facts.append(s)
    return facts[:5]


def extract(
    transcript: str,
    *,
    model: str = "deepseek-chat",
    timeout: float = 45.0,
    max_chars: int = 12000,
) -> list[str] | None:
    """把一段会话提炼成若干条关于用户的事实。

    Returns:
        成功 → 事实列表（可能为空）；未配置 key 或调用失败 → ``None``，
        调用方据此回退到规则抽取。
    """
    if not transcript or not transcript.strip():
        return []
    creds = _credentials()
    if creds is None:
        return None

    base, key = creds
    body = transcript[:max_chars]
    prompt = f"{PROMPT}\n\n----- 聊天记录开始 -----\n{body}\n----- 聊天记录结束 -----"

    try:
        raw = _post_chat(base, key, prompt, model, timeout)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, KeyError):
        return None
    except Exception:
        return None

    return _parse_facts(raw)
