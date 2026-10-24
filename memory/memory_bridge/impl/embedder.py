"""L2 实现：``Embedder`` 的实现。

**唯一持有 sentence-transformers 模型的地方（记忆域）。**

模型：``<models_dir>/bge-small-zh-v1.5``
维度：512

不做什么：
  ❌ 不用 GPU
  ❌ 不换更大的模型（设计文档第十三节写了理由）
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from ..ports import Embedder

# bge 系列在检索任务上的官方查询前缀（中文版）。
# ⚠️ 只加在 **query** 上，不加在 passage 上 —— 两边都加会掉点。
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


class BgeSmallZhEmbedder(Embedder):
    """``BAAI/bge-small-zh-v1.5`` 的封装。

    ⚠️ 加载慢（几秒）→ 必须常驻（这就是 ``server.py`` 存在的理由）。
    ⚠️ ``local_files_only=True``：模型目录缺失时**立刻报错**，
       而不是静默联网下载（国内会挂很久，看起来像进程死了）。
    """

    def __init__(
        self,
        model_dir: Path,
        device: str = "cpu",
        *,
        batch_size: int = 32,
        max_seq_length: int = 512,
    ) -> None:
        if device != "cpu":
            raise ValueError(
                f"BgeSmallZhEmbedder: 只允许 cpu（本机无 GPU，device={device!r} 会直接崩）"
            )
        if not model_dir.exists():
            raise FileNotFoundError(
                f"找不到 embedding 模型目录：{model_dir}\n"
                "请先跑 `python scripts/setup_models.py` 下载模型。"
            )
        self._model_dir = model_dir
        self._batch_size = batch_size
        self._model: Any = SentenceTransformer(
            str(model_dir), device="cpu", local_files_only=True
        )
        self._model.max_seq_length = max_seq_length
        # sentence-transformers 6.x 把 get_sentence_embedding_dimension 改名了，
        # 新名字在 5.x 上不存在 —— 两个都试，避免被版本升级打断。
        getter = getattr(self._model, "get_embedding_dimension", None) or getattr(
            self._model, "get_sentence_embedding_dimension"
        )
        self._dim = int(getter())

    def encode(self, texts: list[str]) -> list[list[float]]:
        """批量编码，**归一化**（让点积 == 余弦）。

        ⚠️ 空文本会被替换成一个占位符再编码：
           空串得到的向量接近零向量，而 pgvector 官方明确说
           **零向量不会被余弦索引命中** —— 那条记忆会永远检索不到却留在 active 集里。
           写入管道本来会拦掉空输入，这里是第二道防线。
        """
        if not texts:
            return []
        cleaned = [t.strip() if t and t.strip() else "（空）" for t in texts]
        vectors = self._model.encode(
            cleaned,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        array = np.asarray(vectors, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        return [[float(x) for x in row] for row in array]

    def encode_query(self, query: str) -> list[float]:
        """查询侧编码（带 bge 的检索指令前缀）。"""
        return self.encode([QUERY_INSTRUCTION + query])[0]

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model_dir.name


class HashEmbedder(Embedder):
    """**仅用于测试**的确定性 embedder —— 不依赖 torch，不依赖模型文件。

    ⚠️ 它不是「第二套存储实现」也不是降级方案：
       它的唯一用途是让 ``tests/`` 能在没有模型文件的环境里跑通
       「管道接线是否正确」，而 **不假装** 自己能做语义检索。
       生产装配永远用 ``BgeSmallZhEmbedder``（见 ``cli.py``）。
    """

    def __init__(self, dim: int = 512) -> None:
        self._dim = dim

    def encode(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vector = np.zeros(self._dim, dtype=np.float32)
            for index, char in enumerate(text):
                vector[(ord(char) + index * 31) % self._dim] += 1.0
            norm = float(np.linalg.norm(vector))
            if norm > 0:
                vector = vector / norm
            out.append([float(x) for x in vector])
        return out

    @property
    def dim(self) -> int:
        return self._dim
