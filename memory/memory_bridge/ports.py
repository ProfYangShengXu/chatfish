"""L2 抽象接口：定义「服务需要什么」，由 ``impl/`` 去实现。

**接口的所有权在调用方。** 放在 L2（不是 domain）是因为
domain 必须保持零内部依赖、且不 import 任何外部库。

⚠️ 这个文件的 import：只 import ``domain.models`` 和标准库。
   **不 import psycopg / sentence_transformers** —— 那是 ``impl/`` 的事。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

from .domain.models import Document, Memory, Patch, SourceKind


class MemoryStore(ABC):
    """记忆与文档的持久化。定义在服务侧，实现侧去满足。"""

    @abstractmethod
    def save_memory(self, m: Memory) -> None:
        """落库一条记忆。若调用方已决定 supersede，实现必须在**同一事务**里
        完成「插入新行 + 更新旧行 ``superseded_by``」。"""

    @abstractmethod
    def supersede(self, old_id: str, new_id: str) -> None: ...

    @abstractmethod
    def active_memories(self, owner: str) -> list[Memory]:
        """``WHERE owner = %s AND superseded_by IS NULL``（走部分索引）。"""

    @abstractmethod
    def touch(self, ids: list[str], now: datetime) -> None:
        """反馈闭环：``hit_count += 1``、``last_access = now``。"""

    @abstractmethod
    def save_chunks(self, docs: list[Document]) -> None: ...

    @abstractmethod
    def append_patch(self, p: Patch) -> None: ...

    @abstractmethod
    def pending_patches(self, owner: str) -> list[Patch]: ...

    @abstractmethod
    def clear_patches(self, owner: str) -> None: ...

    @abstractmethod
    def remove_memory(self, memory_id: str) -> None:
        """用户行使删除权。"""

    # —— 以下三个是「演进」所需：档位自适应与归档 ——
    # ⚠️ 规格的 MemoryStore 只有上面 9 个方法，但 ``service/forget.py`` 要求
    #    「归档而非删除」且 policy 要求 tier 可自适应升降。
    #    没有这三个方法，「永不自动删」与「升降级」两层机制在 ports 上无法表达。
    #    已列入汇报的待确认项（属于「规格含糊 → 补最小接口」，不是重新设计）。

    @abstractmethod
    def update_tier(self, memory_id: str, tier: str) -> None:
        """写回自适应后的档位（``resident`` / ``state`` / ``retrieved``）。"""

    @abstractmethod
    def archive_memories(self, memory_ids: list[str]) -> int:
        """归档而非删除（可回溯 + 合规）。返回被归档的条数。"""

    @abstractmethod
    def archived_count(self, owner: str) -> int: ...


class Embedder(ABC):
    @abstractmethod
    def encode(self, texts: list[str]) -> list[list[float]]: ...

    @property
    @abstractmethod
    def dim(self) -> int: ...


class Retriever(ABC):
    """从候选池里挑最相关的 k 条。"""

    @abstractmethod
    def retrieve(
        self,
        query: str,
        owner: str,
        k: int,
        *,
        scope: str | None = None,
        pool: list[Memory] | None = None,
    ) -> list[Memory]:
        """从候选池里挑最相关的 k 条。

        ⚠️ ``pool`` 是**调用方已经过滤好的候选集**（可见性 + 分档 + 常驻预算退回）。
           传了它就必须只从里面挑 —— 否则「常驻预算超限 → 退回检索档」
           这条降级路径会静默失效（退回的条目既不在常驻档、也检索不到）。
           规格的签名里没有 ``pool``；补它是因为没有它这条数据流表达不出来，
           已列入汇报的待确认项。
        """

    @abstractmethod
    def retrieve_docs(
        self, query: str, collection: str, k: int, *, acl: list[str]
    ) -> list[Document]: ...


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, candidates: list[Memory], k: int) -> list[Memory]: ...

    @abstractmethod
    def score(self, query: str, candidates: list[Memory]) -> list[float]:
        """给候选打**相关性分**（与 ``candidates`` 等长、同序）。

        ★ 这个方法存在的唯一理由是「拒答」：
          ``rerank`` 只回答「谁更相关」，不回答「有没有一个够相关的」。
          没有分数就无法设相关性下限，ANN 的 top-k 会永远返回 k 条 ——
          「库里没有」和「库里有一条勉强相关」在结果上完全一样。

        ⚠️ 评分量纲因实现而异（cross-encoder 是 logit；``NoopReranker`` 恒返回 1.0
           表示「不设下限」）；调用方只应当拿它跟**同一个实现**的阈值比。
        """


class NoopReranker(Reranker):
    """没有精排时的合法实现（小库场景直接用它）。

    —— 它让「不用精排」成为一个配置，而不是「缺了个模块」。
    ⚠️ 它**没有能力**做相关性判定，所以 ``score`` 一律返回 ``1.0``：
       即「不设相关性下限」。这是诚实的行为 ——
       假装能判相关性会让拒答指标变成噪声。
    """

    def rerank(self, query: str, candidates: list[Memory], k: int) -> list[Memory]:
        del query
        return candidates[:k]

    def score(self, query: str, candidates: list[Memory]) -> list[float]:
        del query
        return [1.0] * len(candidates)


class Chunker(ABC):
    """两级切块器（父块 + 子块）。

    ⚠️ 规格在 ``KnowledgeService.ingest`` 的签名里引用了 ``Chunker`` 但全项目没有定义；
       这里补上定义（不改语义，只是把签名里出现过的东西落到实处）。
       具体实现在 ``scripts/kb_ingest.py`` 的 ``build_parent_child``。
    """

    @abstractmethod
    def split(self, text: str, *, source_doc: str) -> list[Document]: ...


class TranscriptReader(ABC):
    """把各家 agent 的会话记录读成纯文本（实现归 ``bridge/transcript.py``）。

    ⚠️ 引出这个端口是为了让 ``bridge/session.py`` 不直接依赖
       ``bridge/transcript.py`` 的模块函数，从而可以在单测里替换掉文件 IO。
    """

    @abstractmethod
    def read(self, path: Path, *, max_chars: int) -> str: ...

    @abstractmethod
    def latest(self, agent: str, cwd: Path) -> Path | None: ...


class UsageAttributor(ABC):
    """判定「哪些被召回的记忆在本轮真的被用上了」→ ``used_count``。

    ★ 这是 ``mark_used`` 的难点所在（规格 ``bridge/session.py`` 那节）。
      规格给的主路径是「把本轮召回的记忆 + 模型回答一起喂给模型做归因」，
      但本项目的依赖清单里**没有任何 LLM 客户端**（规格第〇节 15 个包）。
      所以把它抽成端口：装配时注入具体实现，没有 LLM 时用退化实现。

    ⚠️ 规格明确写「不要用 ① 之外的启发式当主路径 —— 会引入噪声，污染升档判据」。
       因此默认实现（``impl/reranker.py`` 之外的 ``LexicalAttributor``）**只在显式配置
       时启用**，并且它在 ``should_promote`` 语义上被标记为弱信号。
    """

    @abstractmethod
    def attribute(
        self, *, memories: list[Memory], answer: str
    ) -> list[str]:
        """返回「确实被用上」的 memory id 列表。"""


class Clock(ABC):
    """可注入的时间源（让 service 与 bridge 的测试不依赖真实时间）。"""

    @abstractmethod
    def now(self) -> datetime: ...


__all__ = [
    "MemoryStore",
    "Embedder",
    "Retriever",
    "Reranker",
    "NoopReranker",
    "Chunker",
    "TranscriptReader",
    "UsageAttributor",
    "Clock",
    "Document",
    "Memory",
    "Patch",
    "SourceKind",
]
