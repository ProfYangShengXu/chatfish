# Memory Bridge 运维须知

## 部署形态

Memory Bridge 的记忆存储只使用 PostgreSQL 16 加 pgvector 0.6.0，不引入 ChromaDB、
Qdrant 或任何向量数据库。数据库跑在 WSL2 的 Ubuntu 里，监听 127.0.0.1:5432；
代码从 Windows 侧通过 WSL2 的 localhost 转发连接。因为走的是 host 认证
（scram-sha-256），DSN 必须带密码；local 连接才是 peer 免密。

pgvector 的扩展必须在目标库里显式创建：`CREATE EXTENSION IF NOT EXISTS vector`。
不创建的话第一次建索引就会失败，而且报错信息不会提示你去建扩展。

## 容量与模型

embedding 模型固定为 BAAI/bge-small-zh-v1.5，输出 512 维向量，归一化之后点积等价于余弦相似度。
精排模型固定为 BAAI/bge-reranker-base，这是一个 cross-encoder，必须成对输入，
无法预计算，所以只能在召回之后使用。

两个模型加起来大约 1.2GB，模型文件放在 D 盘的 datasets 目录下，不要放进 C 盘。
机器没有独立显卡，torch 是 CPU 版本，任何 device='cuda' 的代码都会直接抛异常。

## 召回策略

召回走混合检索：BM25 与向量检索各跑一路，再用 RRF 融合。融合只使用排名，
不使用原始分数，因为 BM25 分与余弦距离的量纲不可比。RRF 的 k 取 60。

常驻档的预算是 800 token，不是条数。一条 5 个字的约束和一条 80 个字的偏好，
占用的上下文差 16 倍，所以只能用 token 计量。超出预算的条目退回检索档，不是删除。

## 缓存友好性

prompt 缓存依赖前缀稳定。所以常驻内容必须逐字节稳定，任何按轮次变化的字段
（时间戳、轮次计数、命中次数）都不允许出现在常驻档。可变内容只允许出现在
prompt 的最后，这样它之后没有别的内容，不产生额外的缓存代价。

## 常见故障

如果召回结果突然变差，先检查 BM25 的 avgdl 统计量。批量导入短文本会让
所有旧条目的 BM25 分数集体下降，因为 avgdl 是全局统计量。解决办法是在大批量
写入之后重建索引，并跑一次回归评估确认没有退化。

如果 supersede 没有生效，检查两条记忆的 embedding 是否真的写进了数据库。
写入失败时行仍然存在，embedding 却是 NULL，这时余弦相似度永远是 NULL，
supersede 判定就永远不会命中。

## 安全边界

owner 过滤必须写在 SQL 的 WHERE 子句里，不能靠排序之后再过滤。
跨 owner 是安全边界，不是排序偏好。知识域的 chunk 不允许写进 memories 表，
两个域的分离本身就是一道安全边界。

外部来源（网页、第三方文档）的内容不进记忆域，只进知识库。
写入内存的来源信任级里，EXTERNAL 一级永远是 False。
