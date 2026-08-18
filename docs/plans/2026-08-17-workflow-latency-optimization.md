# 全工作流生成速度优化实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` and TDD to implement this plan task-by-task.

**Goal:** 在不改变 LangGraph 拓扑、公共调用接口和证据质量门槛的前提下，使普通 RAG、Web RAG 和技术选型冷请求的中位总耗时至少降低 30%。

**Architecture:** 保留现有节点与边，通过复用上游结构化结果、合并重复 LLM 审核、为工具型调用关闭 thinking、启动预热和失败熔断减少串行等待。最终回答模型、检索数据、Embedding、Milvus/Neo4j 索引和记忆职责不变。

**Tech Stack:** Python、LangGraph 1.2.9、LangChain 1.3.14、DeepSeek、Milvus、Neo4j、MySQL Checkpointer。

---

## 1. 基线与性能观测

- 为顶层调用和普通 RAG 内部节点增加轻量耗时日志；内部节点关闭 heartbeat。
- 日志只记录请求作用域、路径、节点、耗时、缓存命中、模型调用次数和检索结果数，不记录问题正文、凭据或完整记忆。
- 增加只读 benchmark 脚本，使用无 repository/checkpointer 的兼容图，结果写入 `runtime/reports/latency-*.json`。
- 覆盖直接回答、本地 RAG、Web RAG 和技术选型各3个固定样本；主指标为中位总耗时，最大值只作为尾延迟观察。

## 2. 减少串行 LLM 调用

- 新增 `utility_llm`（温度0、关闭 thinking、20秒超时、最多1次重试）与 JSON mode 版本；最终回答使用现有 `llm`，设置60秒超时和最多1次重试。
- 入口分类同时返回可选检索意图；`analyze_question` 优先复用，校验失败时回退到原模型链。
- 技术选型证据任务携带内部检索意图，保留节点但避免重复分析。
- 直接回答使用现有安全模板，不调用 LLM。
- 将普通 RAG 的幻觉检查和答案质量检查合并为一次结构化调用；异常时回退到原两段审核。
- 技术选型候选评分同时预计算风险；`analyze_risks` 继续执行原证据校验，结果非法时回退到独立风险链。

## 3. 预热、召回快路径与失败熔断

- 将只读预热放入 `open_memory_application()` 生命周期，覆盖 BGE、主 Milvus retriever、Neo4j 和长期记忆 Milvus client；CLI 不再重复预热。
- 预热失败不终止应用、不创建集合或修改索引，失败资源30秒内快速失败，之后允许恢复探测。
- 本地向量和图检索同时不可用时直接进入 Web fallback，单个失败时保留另一路证据。
- Web 单次 HTTP 超时改为15秒，保留现有最多2次尝试、查询改写和相关性要求。
- 长期记忆当前作用域没有对应权威记忆时跳过 embedding/Milvus；有记忆时保留语义召回、回表和版本校验。
- 持久化候选后清理最新 state 的临时大字段，保留 `messages`、`session_user_id` 和历史 checkpoint 快照。

## 4. 配置、测试与验收

- 增加 LLM 超时/重试、Web 15秒超时、启动预热和30秒资源冷却配置。
- 保持公共输入输出、节点名称、数量、边、重试/证据/审核上限和全部数据库结构不变。
- 按 TDD 覆盖调用次数、异常回退、预热/熔断、记忆快路径、checkpoint 与图拓扑回归。
- 使用项目解释器逐测试文件独立运行；真实验收使用相同环境比较前后中位耗时。
- 未达30%时只对未达标路径间隔5分钟重测一次；仍未达标时保留耗时和调用次数证据，不降低证据或审核门槛。

## Assumptions

- 单机少量用户，优先单请求总耗时，不新增流式公共接口。
- 接受启动变慢换取首请求变快，应用实例在进程生命周期内复用。
- 当前未提交的 `feature-memory` 工作树是优化基线，实施时不建新 worktree，不自动暂存或提交用户改动。
