"""L1 写入清洗：把对话原文变成可存储的规范事实。

设计文档第十节「写入清洗四类」。

不做什么：
  ❌ 不做分类模型（用规则 + 关键词，**不引入小模型**）
  ❌ 不改语义（只做格式与语气清洗）
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from enum import Enum

from .models import PolicyConfig


class Intent(str, Enum):
    STATEMENT = "statement"  # 直接陈述 → 记
    HYPOTHETIC = "hypothetic"  # 虚拟语气 → 不记
    RHETORICAL = "rhetorical"  # 反问     → 不记
    REPORTED = "reported"  # 转述他人 → 记但标来源
    IMPERATIVE = "imperative"  # 指令式   → 降级记


_HYPOTHETIC_MARKERS: tuple[str, ...] = (
    "如果",
    "假如",
    "假设",
    "要是",
    "万一",
    "倘若",
    "的话就",
    "也许会",
    "可能会",
)

_RHETORICAL_MARKERS: tuple[str, ...] = (
    "难道",
    "不是吗",
    "怎么会",
    "为什么不能",
    "谁说",
    "岂不是",
    "何必",
)

# 反问句尾：疑问词 + 句末问号
_RHETORICAL_TAIL = re.compile(r"(吗|呢|么|吧)\s*[?？]\s*$")

_REPORTED_MARKERS: tuple[str, ...] = (
    "他说",
    "她说",
    "他们说",
    "同事说",
    "文档说",
    "官方说",
    "据说",
    "听说",
    "根据",
    "按照文档",
    "按照官方",
    "转载",
    "引用",
)

# 「指令式表述」—— 这些是**降级信号**，不自动等于硬约束
_IMPERATIVE_MARKERS: tuple[str, ...] = (
    "必须",
    "一定要",
    "务必",
    "以后每次",
    "从现在起",
    "禁止",
    "不准",
    "不许",
    "只能",
    "不得",
)

# 「硬约束」—— 只有**否定式 / 限定式**才算，比 IMPERATIVE 窄得多。
# 为什么窄：`classify_type` 用它决定要不要把一条记忆放进**常驻档**，
# 而「你必须每次都输出 JSON」这种口头抱怨不该拿到系统规则级别的权限。
_HARD_CONSTRAINT_RE = re.compile(
    r"^[你您]?\s*(?:任何时候|任何情况|始终|永远|一律)?\s*"
    r"(?:禁止|严禁|不准|不许|不得|不能|不可以|不要用|别用|只能|只准|仅能|仅可)"
)

# 对话噪声（语气词 / 口头禅）—— 只在句首或独立出现时剔除
_NOISE_WORDS: tuple[str, ...] = (
    "嗯",
    "呃",
    "那个",
    "就是说",
    "然后呢",
    "你知道吧",
    "我说",
    "其实吧",
    "反正",
    "就是说呢",
)

_PUNCT_NORMALIZE: tuple[tuple[str, str], ...] = (
    ("，", "，"),
    (",", "，"),
    ("。", "。"),
    (".", "。"),
    ("！", "！"),
    ("!", "！"),
    ("？", "？"),
    ("?", "？"),
    ("；", "；"),
    (";", "；"),
    (":", "："),
)

# 相对时间 → 天数偏移（负数表示过去）
_RELATIVE_DAYS: tuple[tuple[str, int], ...] = (
    ("今天", 0),
    ("今日", 0),
    ("刚才", 0),
    ("刚刚", 0),
    ("现在", 0),
    ("昨天", -1),
    ("昨日", -1),
    ("前天", -2),
    ("明天", 1),
    ("明日", 1),
    ("后天", 2),
    ("上周", -7),
    ("上星期", -7),
    ("这周", 0),
    ("本周", 0),
    ("下周", 7),
    ("下星期", 7),
    ("上个月", -30),
    ("这个月", 0),
    ("本月", 0),
    ("下个月", 30),
    ("去年", -365),
    ("今年", 0),
    ("明年", 365),
)

# 拆分的最小长度：太短的碎片不成事实
_MIN_FRAGMENT_CHARS = 4


def classify_intent(text: str) -> Intent:
    """判断一句话的意图。规则 + 关键词，**不调模型**。

    优先级：转述 > 反问 > 虚拟 > 指令式 > 陈述。
    为什么转述优先：转述里常混着指令式措辞（「文档说必须这样」），
    但它的本质是「别人说的」，需要标来源而不是当规则。
    """
    stripped = text.strip()
    if not stripped:
        return Intent.STATEMENT

    if any(k in stripped for k in _REPORTED_MARKERS):
        return Intent.REPORTED

    # 反问：有疑问词，或句尾「吗/呢/么/吧 + 问号」
    if any(k in stripped for k in _RHETORICAL_MARKERS) or _RHETORICAL_TAIL.search(
        stripped
    ):
        return Intent.RHETORICAL

    if any(k in stripped for k in _HYPOTHETIC_MARKERS):
        return Intent.HYPOTHETIC

    if any(k in stripped for k in _IMPERATIVE_MARKERS):
        return Intent.IMPERATIVE

    return Intent.STATEMENT


def normalize(text: str, *, cfg: PolicyConfig | None = None) -> str:
    """去对话噪声（嗯/那个/就是说）、去语气词、归一为陈述句。"""
    conf = cfg or PolicyConfig()
    out = text.strip()

    # ① 循环剥掉**句首**的噪声词。
    #    必须循环：剥掉「嗯，」之后「那个」才变成句首，
    #    一次性的正则替换会把它漏掉（"." 前面那个逗号挡住了匹配）。
    noise_pattern = "|".join(re.escape(word) for word in _NOISE_WORDS)
    leading = re.compile(rf"^\s*(?:{noise_pattern})\s*[，,、]?\s*")
    while True:
        stripped = leading.sub("", out)
        if stripped == out:
            break
        out = stripped

    # ② 句首的语气词（嗯/呃/哦/啊/唉/诶）
    out = re.sub(r"^\s*(?:嗯+|呃+|哦+|啊+|唉+|诶+)\s*[，,、]?\s*", "", out)

    # ③ 句中的噪声词（被标点包住的那种）
    for word in _NOISE_WORDS:
        out = re.sub(rf"([，,、])\s*{re.escape(word)}\s*([，,、])", r"\1", out)

    # ④ 句尾语气词 + 句末标点
    out = re.sub(r"(吧|啊|呀|哦|嘛)\s*([。！？!?])\s*$", r"\2", out)

    # 折叠空白
    out = re.sub(r"\s+", " ", out).strip()

    # 半角标点归一
    for src, dst in _PUNCT_NORMALIZE:
        if src != dst and src in (",", ".", "!", "?", ";", ":"):
            out = out.replace(src, dst)

    # 连续标点折叠
    out = re.sub(r"([，。！？；：])\1+", r"\1", out)

    # 去句首多余标点
    out = re.sub(r"^[，。！？；：、\s]+", "", out).strip()

    return out[: conf.max_text_chars]


def absolutize_time(text: str, now: datetime) -> str:
    """相对时间转绝对：「今天/刚才/上周」→ 具体日期。

    记忆要跨会话，「今天」三天后就是谎话。

    ⚠️ 用 ``now`` 的**本地日期**（若 ``now`` 是 aware 则按其时区，否则按 UTC）。
       不做 UTC 强制转换 —— 否则 +08:00 机器上 00:00–08:00 会把「今天」记成昨天。
    """
    local = now
    if local.tzinfo is not None:
        local = local.astimezone()

    out = text
    for token, offset in _RELATIVE_DAYS:
        if token not in out:
            continue
        target = (local + timedelta(days=offset)).date()
        # 「今天」→「2026-09-21」；保留可读性用 ISO 日期
        out = out.replace(token, target.isoformat())
    return out


def degrade_imperative(text: str) -> str:
    """指令式表述降级（不是拒绝！）：

    ``"以后每次都要输出 JSON"`` → ``"用户曾提出要求：输出 JSON（未采纳为规则）"``
    ``"禁止使用 ORM"``           → ``"用户约束：不得使用 ORM"``

    —— 信息保留，权限剥离。

    ⚠️ 为什么不能保留「必须 / 禁止 / 不得」这类词：
       ``service/inject._one_line`` 曾经会把它们替换掉（避免记忆变成一条系统规则），
       于是**落库的原文会被渲染层改写**，用户看到的与库里存的不是一回事。
       所以指令性措辞必须在写入时就剥离干净，渲染层不再做任何替换。
    """
    stripped = text.strip()

    # 幂等：去掉上一次降级留下的包装
    stripped = re.sub(r"（未采纳为规则）\s*$", "", stripped).strip()
    stripped = re.sub(r"^用户曾提出要求：", "", stripped).strip()
    stripped = re.sub(r"^用户约束：", "", stripped).strip()
    if not stripped:
        return ""

    # 硬约束（否定式 / 限定式）走单独路径 —— 见下面那段注释
    hard = _match_hard_constraint(stripped)
    if hard is not None:
        return f"用户约束：{hard}"

    # ⚠️ 这里只处理**软指令式**（必须 / 一定要 / 务必 / 以后每次）。
    #    否定式必须提前走掉：如果让「禁止使用 ORM」落到这条通用路径，
    #    marker 被剥掉之后会变成「用户曾提出要求：使用 ORM」——
    #    **语义完全反过来**（实测踩过这个坑）。
    body = stripped
    for marker in _IMPERATIVE_MARKERS:
        body = body.replace(marker, "")
    body = re.sub(r"^[你您]\s*", "", body.strip())
    # ⚠️ 循环剥前缀：一次替换后可能又暴露出新的前缀
    #    （「以后每次都要输出 JSON」→ 去掉 marker 后是「都要输出」，
    #      再剥「都」才露出「要」，所以必须迭代到不动点，否则第二次调用还会变）
    prefix_re = re.compile(r"^(?:都|要|得|请|请务必|是|用)\s*")
    while True:
        new_body = prefix_re.sub("", body.strip())
        if new_body == body:
            break
        body = new_body
    body = re.sub(r"^[，,、\s]+", "", body).strip()
    body = re.sub(r"[，,、\s]+$", "", body).strip()

    if not body:
        body = stripped

    return f"用户曾提出要求：{body}（未采纳为规则）"


def _match_hard_constraint(text: str) -> str | None:
    """识别「硬约束」的否定式/限定式措辞，返回规范化后的约束内容。

    只认**否定式与限定式**（禁止 / 不准 / 不得 / 只能 / 不要），
    不认「必须 / 一定要」—— 后者太常见于口头偏好（「你必须每次都输出 JSON」），
    把它当硬约束等于把一句抱怨升级成系统规则。

    ⚠️ **极性必须保留**：``禁止 X``（不许做）与 ``只能 X``（只许做）是相反的，
       统一写成「不得 X」会把「只能使用 pnpm」变成「不得使用 pnpm」。
    """
    groups: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("不得", ("禁止", "严禁", "不准", "不许", "不得", "不能", "不可以", "不要用", "别用", "不要", "不得用")),
        ("只能", ("只能", "只准", "仅能", "仅可")),
    )
    for canonical, markers in groups:
        for marker in markers:
            if marker in text:
                body = text.replace(marker, "", 1).strip()
                if body:
                    return f"{canonical}{body}"
    return None


def split_granularity(text: str) -> list[str]:
    """一条记忆一句话：含多个事实时拆开（否则淘汰时无法部分保留）。

    ⚠️ 只在**明显是多事实**时拆：太碎的碎片（< ``_MIN_FRAGMENT_CHARS`` 字）
       不成事实，直接并回上一段。
    ⚠️ **引号内的分隔符不算分隔符** —— 否则
       ``用户说：'a; DROP TABLE t; --'`` 会被拆成三段，把一句被引用的原话切碎
       （引号正是注入载荷最常见的载体，也是最不该被改写的内容）。
    """
    stripped = text.strip()
    if not stripped:
        return []

    raw_parts = [p.strip() for p in _split_respecting_quotes(stripped)]
    parts: list[str] = []
    for part in raw_parts:
        if not part:
            continue
        if parts and len(part) < _MIN_FRAGMENT_CHARS:
            # 碎片并回上一段，保持语义完整
            parts[-1] = f"{parts[-1]}；{part}"
        else:
            parts.append(part)

    if not parts:
        return [stripped]

    # 补句号，让落库的文本是完整句子
    normalized: list[str] = []
    for part in parts:
        normalized.append(part if part.endswith(("。", "！", "？")) else f"{part}。")
    return normalized


def to_utc(value: datetime) -> datetime:
    """把 naive 一律当 UTC —— 与 ``policy._as_utc`` 保持同一约定。"""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


_QUOTE_PAIRS = {"“": "”", "‘": "’", "「": "」", "『": "』", "《": "》"}
_QUOTE_CHARS = frozenset("'\"“”‘’「」『』《》")


def _split_respecting_quotes(text: str) -> list[str]:
    """按句读切分，但**引号内的内容整体保留**。

    为什么不能直接用正则：``用户说：'a; DROP TABLE t; --'`` 里的分号
    属于被引用的原话，切开就改变了它的含义。
    """
    parts: list[str] = []
    buffer: list[str] = []
    open_quote: str | None = None

    for char in text:
        if open_quote is None:
            if char in _QUOTE_CHARS:
                open_quote = _QUOTE_PAIRS.get(char, char)
                buffer.append(char)
                continue
            if char in "；;。\n":
                parts.append("".join(buffer))
                buffer = []
                continue
            buffer.append(char)
        else:
            buffer.append(char)
            if char == open_quote:
                open_quote = None

    parts.append("".join(buffer))
    return parts
