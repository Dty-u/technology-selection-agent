# 使用 LangGraph Checkpointer 实现短期记忆

## 目标

将短期记忆权威来源从 `rag_memory.memory_messages` 迁移为 `QueryRouterState.messages + add_messages + MySQL checkpointer`，保留最近六轮原始对话，不调用总结模型。长期记忆继续使用 `rag_memory.memory_items + Milvus`。

## 接口与状态

- 保持 `open_memory_application()`、`application.invoke({"query": ...}, config=...)` 和 `Command(resume=...)` 不变。
- 生产调用继续要求显式 `user_id`、`thread_id`、`request_id`；技术选型继续要求 `project_id`。
- `QueryRouterState` 增加 `messages` 和 `session_user_id`。
- `short_term_context` 仅保留为图内派生字段，从 `QueryRouterInput` 删除。
- `QueryRouterOutput` 不暴露 checkpoint 消息。
- 使用 request-scoped 确定性 UUID 保证重试幂等。

## 实施步骤

1. 在顶层图增加 `prepare_short_term_memory` 和 `finalize_short_term_memory`。
2. `prepare` 校验 thread 用户归属、从旧消息生成最近六轮完整原文、追加当前用户消息。
3. `finalize` 追加非空助手回答，并用 `RemoveMessage` 裁剪为最近 12 条。
4. 将 graph 顺序调整为 `START → prepare → classify → 原有路由/子图 → build → finalize → 长期记忆写入 → END`。
5. runner 不再读写长期记忆仓储中的原始消息；恢复时把用户回答合并到原 `Command.update`。
6. interrupt 返回后把 `clarification_questions` 写成助手消息，并保持暂停位置可恢复。
7. 删除 `MemoryMessage`、消息 CRUD、`memory_messages` schema 和旧表专用测试。
8. 新增 checkpoint 行为测试、ADR，并运行分进程回归。

## 数据库切换门禁

代码验证后先只读执行：

```sql
SELECT DATABASE();
SELECT COUNT(*) AS rows_to_delete
FROM rag_memory.memory_messages;
```

只有确认当前数据库为 `rag_memory` 且用户在独立审批点明确同意后，才执行：

```sql
DROP TABLE rag_memory.memory_messages;
```

随后验证 `memory_messages` 不存在，而 `rag_memory.memory_items` 与 `rag_checkpoint.checkpoints` 仍存在。本次不备份、不迁移旧消息，删除后不可恢复。

## 验收

- 工作流前不再查询历史消息表。
- checkpoint 是唯一短期记忆权威，最新 state 仅保留六轮。
- 同 thread 连续问答、重试、用户隔离与 interrupt/resume 均由 checkpoint state 覆盖。
- 长期记忆召回、候选写入、outbox、Milvus 投影及原有 RAG/技术选型行为保持兼容。
- `/close` 只删除指定 checkpoint thread。
