# ADR-0001: 使用 Direct Milvus 作为长期记忆语义投影

## Status

Accepted

## Context

项目需要同时支持 LangGraph 会话恢复、可审计的长期业务记忆和语义相似检索。MySQL 已承担 checkpoint、长期记忆版本、确认状态、审计和 transactional outbox；现有 Milvus collection `adaptive_rag_c1` 承担知识库证据检索。

Mem0 会引入第二套历史语义和独立 HTTP bridge，并与主环境锁定的 `pymilvus==2.6.9` 存在依赖隔离成本。长期记忆仍必须以 MySQL 为权威，语义索引必须可删除和重建。

## Decision

- 移除 Mem0、独立 worker 环境和 HTTP bridge。
- 主环境直接使用 `pymilvus==2.6.9` 操作专用 collection `rag_agent_memory_v1`。
- Milvus 只保存向量、权威 `memory_id`、作用域、状态和版本；正文始终从 MySQL 回填。
- MySQL/outbox 写入与用户回答同步完成；Milvus 投影由后台 worker 异步执行。
- collection 初始化必须显式触发，只允许缺失时创建；现有 schema 不兼容时安全失败，禁止自动删除或重建。

## Consequences

### Positive

- 移除重复的记忆抽取/history 语义和额外 HTTP 服务。
- 消除 Mem0 与主环境 PyMilvus 的版本冲突。
- Milvus 泄漏、过期命中或暂时不可用都不会绕过 MySQL 权威校验。
- `memory_id` 同时作为 MySQL 和 Milvus 标识，投影 upsert/delete 天然幂等。

### Negative

- 需要自行维护 Milvus schema、过滤表达式和 hit 解析。
- 语义检索仍需要同步生成 query embedding；Milvus 故障时只能退化为 MySQL 精确记忆。

### Neutral

- `adaptive_rag_c1`、Neo4j、Web 和技术选型评分逻辑保持不变。
- SQLite checkpoint 文件继续作为冷归档保留。

## Alternatives Considered

- **Mem0 + Milvus**：拒绝，因为引入重复 history、额外服务和 PyMilvus 版本隔离。
- **只使用 MySQL**：拒绝，因为不能提供向量语义召回。
- **只使用 Milvus 保存完整记忆**：拒绝，因为无法满足事务、审批、版本和审计的权威存储要求。

## References

- https://milvus.io/api-reference/pymilvus/v2.6.x/MilvusClient/Vector/search.md
- https://milvus.io/api-reference/pymilvus/v2.6.x/MilvusClient/Collections/create_collection.md
