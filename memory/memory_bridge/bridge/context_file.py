"""L3 桥：上下文文件写入。

把常驻档写进 ``CLAUDE.md`` / ``AGENTS.md`` 的记忆区。**这是零适配的注入方案。**

⚠️ 铁律：
  > **不要碰标记之外的任何字节。** 用户可能在里面写了自己的项目说明，
  > **覆盖它会毁掉用户的文件**。
  > 写入前先 ``read_current()`` 比对，**内容相同就跳过写入**
  > （避免 mtime 变化引发无意义的重载）。

⚠️ 第二条铁律（**实测踩过的坑**）：**标记必须是「独占一行」才算分隔符。**
   用户完全可能在正文里*提到*这两个标记 —— 例如写一句
   「记忆桥只能改 `<!-- memory-bridge:begin -->` 与 … 之间的东西」。
   如果按子串查找，写入器会把记忆区**插进那句话中间**，
   直接毁掉用户的文件（本次实测就发生了这种事）。
   判据：``line.strip() == MARK``。行内的提及一律不算。
"""

from __future__ import annotations

from pathlib import Path

BEGIN_MARK = "<!-- memory-bridge:begin -->"
END_MARK = "<!-- memory-bridge:end -->"

# 记忆区的标题（与 BEGIN_MARK 一起构成固定头部）
_SECTION_TITLE = "## 关于该用户的历史记录（自动维护，请勿手改本区）"


def find_marker_lines(text: str) -> tuple[int, int]:
    """返回 ``(begin 行号, end 行号)``，只认**独占一行**的标记。

    找不到返回 ``(-1, -1)``。行内提及（写在反引号里、句子中间）一律不算 ——
    见模块文档里的第二条铁律。
    """
    begin = end = -1
    for index, line in enumerate(text.splitlines()):
        stripped = line.strip()
        if stripped == BEGIN_MARK and begin == -1:
            begin = index
        elif stripped == END_MARK and begin != -1 and end == -1:
            end = index
            break
    if begin == -1 or end == -1 or end < begin:
        return -1, -1
    return begin, end


def _split_lines_keepends(text: str) -> list[str]:
    return text.splitlines(keepends=True)


class ContextFileWriter:
    """只替换 ``BEGIN_MARK`` 与 ``END_MARK`` 之间的内容。

    ⚠️ 编码一律显式 ``utf-8``：Windows 侧 ``Path.write_text()`` 默认用 locale
       （cp936），写出来的 CLAUDE.md 是 GBK，而 agent 按 UTF-8 读 → 整个记忆区乱码。
       「逐字节相同」的验收项**看不出**这个 bug。
    """

    def __init__(self, targets: list[Path]) -> None:
        self._targets = list(targets)

    @property
    def targets(self) -> list[Path]:
        return list(self._targets)

    def read_current(self, *, owner: str) -> str:
        """读回当前记忆区（用于 diff：没变就不写，避免无谓的文件改动）。

        ``owner`` 不参与内容 —— 记忆区的内容完全由 ``render_resident`` 决定；
        参数保留是为了与端口的其它方法签名一致，以及将来支持多 owner 分文件。
        """
        del owner
        return self.read_block(self._targets[0]) if self._targets else ""

    def read_block(self, path: Path) -> str:
        """读某个文件里记忆区的内容（不含标记本身）。文件/标记不存在返回 ""。"""
        if not path.exists():
            return ""
        text = path.read_text(encoding="utf-8", errors="replace")
        begin, end = find_marker_lines(text)
        if begin == -1:
            return ""
        lines = _split_lines_keepends(text)
        return "".join(lines[begin + 1 : end]).strip("\n")

    def write(self, *, owner: str, resident: str, state: str) -> list[Path]:
        """写记忆区。返回**实际发生写入**的文件列表（内容没变的不算）。

        - 文件不存在 → 创建
        - 标记不存在 → 【追加】到末尾（不覆盖已有内容）
        - 标记存在   → 只替换标记之间的内容
        """
        block = self._render_block(resident=resident, state=state)
        written: list[Path] = []

        for path in self._targets:
            if self._write_one(path, block):
                written.append(path)
        return written

    def _render_block(self, *, resident: str, state: str) -> str:
        inner = resident.strip()
        if state.strip():
            inner = f"{inner}\n\n{state.strip()}" if inner else state.strip()
        if not inner:
            inner = "（暂无）"
        return f"\n{_SECTION_TITLE}\n\n{inner}\n"

    def _write_one(self, path: Path, block: str) -> bool:
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = (
            path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
        )

        begin, end = find_marker_lines(existing)
        lines = _split_lines_keepends(existing)

        if begin == -1:
            # 普通情形：没有完整的标记对 → 追加到末尾，**不覆盖已有内容**。
            # ⚠️ 只有 BEGIN 没有 END（用户误删了 END）时，直接追加会留下两个 BEGIN。
            #    先把那个孤立的 BEGIN 行删掉，但保留它**后面**的用户内容。
            orphan = _find_lone_begin_line(lines)
            if orphan is not None:
                lines = lines[:orphan] + lines[orphan + 1 :]
                existing = "".join(lines)
            separator = "" if not existing else ("" if existing.endswith("\n") else "\n")
            updated = f"{existing}{separator}\n{BEGIN_MARK}{block}{END_MARK}\n"
        else:
            updated = (
                "".join(lines[:begin])
                + f"{BEGIN_MARK}{block}{END_MARK}\n"
                + "".join(lines[end + 1 :])
            )

        # ★ 内容相同就跳过写入（避免 mtime 变化引发无意义的重载）
        if updated == existing:
            return False

        path.write_text(updated, encoding="utf-8")
        return True


def _find_lone_begin_line(lines: list[str]) -> int | None:
    """找「只有 BEGIN、没有配对 END」的那个 BEGIN 行号。"""
    for index, line in enumerate(lines):
        if line.strip() == BEGIN_MARK:
            return index
    return None


def extract_unmanaged(path: Path) -> str:
    """返回「标记之外」的全部内容 —— 验收 D.1 用它证明一字未动。"""
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    begin, end = find_marker_lines(text)
    if begin == -1:
        return text
    lines = _split_lines_keepends(text)
    return "".join(lines[:begin] + lines[end + 1 :])


__all__ = [
    "BEGIN_MARK",
    "END_MARK",
    "ContextFileWriter",
    "extract_unmanaged",
    "find_marker_lines",
]
