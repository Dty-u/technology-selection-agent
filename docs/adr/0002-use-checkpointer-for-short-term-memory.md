# ADR 0002：使用 LangGraph Checkpointer 作为短期记忆权威来源

- 状态：Accepted
- 日期：2026-08-16

## 背景

此前，`MemoryAwareRunner` 在进入顶层工作流前读取 `rag_memory.memory_messages`，拼接最近六轮原始对话，并在工作流结束后写回用户和助手消息。与此同时，LangGraph checkpointer 已按 `thread_id` 保存工作流 state、interrupt 和恢复位置。两套会话状态来源造成了重复写入、生命周期不一致和恢复语义分裂。

## 决策

短期记忆改为顶层 `QueryRouterState.messages`，使用 `add_messages` reducer，并由生产 MySQL checkpointer 按 `thread_id` 持久化。

- `prepare_short_term_memory` 从 checkpoint 恢复的消息派生最近六轮原文上下文，再追加当前 `HumanMessage`。
- `finalize_short_term_memory` 追加最终 `AIMessage`，并通过 `RemoveMessage` 将最新 state 裁剪为 12 条消息。
- 技术选型 interrupt 的 `clarification_questions` 和 `Command(resume=...)` 回答都计入消息窗口。
- 消息 ID 基于 `user_id + thread_id + request_id + kind` 生成确定性 UUID，重试时替换同 ID 消息。
- `session_user_id` 绑定首次使用该 thread 的用户；不匹配时在模型调用前受控失败。
- `short_term_context` 仅是图内派生字段，不再是公共输入，也不调用总结模型。
- `/close` 只删除指定 `thread_id` 的 checkpoint。旧 checkpoint 快照在此之前由 checkpointer 保留。

`rag_memory.memory_items` 继续作为长期记忆权威来源，Milvus 继续作为可重建语义投影；二者不承担短期会话恢复。

## 影响

- `memory_messages`、`MemoryMessage` 和消息 CRUD 从代码与基础 schema 中退役。
- 新部署不再创建 `memory_messages`。
- 旧表不迁移、不备份；经独立审批执行 `DROP TABLE rag_memory.memory_messages` 后，历史原始消息不可恢复。
- 无 checkpointer 的模块级兼容图仍可单轮运行，但不会跨调用保留短期记忆。
- 现有公共生产入口、普通请求、JSON 输出和 `Command(resume=...)` 调用方式保持不变。

## 替代方案

保留 `memory_messages` 作为短期记忆权威来源被否决，因为它不能天然保存 LangGraph 的 interrupt 位置与完整 state，且要求工作流外额外查询和双写。
