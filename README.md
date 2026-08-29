# 晶圆智选 · Decision Lab

> 面向半导体领域的技术问答与技术选型智能体。

基于 **LangGraph** 构建多级工作流：主路由图自动识别请求类型，路由到**普通问答**（向量 + 图谱混合检索）或**技术选型**（证据驱动的多轮评估）子工作流，并内置**短期记忆（会话）与长期记忆（持久化）**机制。

## ✨ 核心特性

- 🧠 **智能路由**：识别请求类型（普通问答 / 技术选型 / 直接响应），自动分发到对应子工作流
- 🔍 **混合检索**：Milvus 向量检索 + Neo4j 图谱检索双路并行，融合后经证据相关性筛选
- 🛡️ **质量门控**：生成结果经过幻觉评估与问题解决评估，失败时按「改写查询 → 重新生成 → 联网搜索」有界修复
- 📊 **证据驱动选型**：候选生成 → 硬约束筛选 → 多轮证据补检（上限 3 轮）→ 评分 → 风险分析 → 审查（上限 2 轮）→ Markdown/Word 报告导出
- 💾 **双记忆机制**：短期记忆（会话上下文摘要）+ 长期记忆（MySQL 记忆仓库 + Milvus 语义索引，记忆候选暂存/持久化）
- ⏸️ **断点续跑**：LangGraph 检查点（MySQL，兼容 SQLite），中断后可从上次进度恢复
- 🌐 **Web 交互**：FastAPI REST 接口 + Vue 3 前端，支持会话管理、报告下载、工作流进度跟踪

## 🗺️ 工作流总览

系统由三个工作流组成：**主路由**（入口 + 记忆编排）嵌套 **普通问答** 与 **技术选型** 两个子工作流。

```mermaid
graph TD
    START([开始]) --> prepare[prepare_short_term_memory<br/>准备短期记忆]
    prepare -->|"正常"| classify[classify_query<br/>识别请求类型]
    prepare -->|"status=error"| build[build_query_response<br/>整理统一响应]

    classify -->|"direct_response"| direct[respond_directly<br/>生成直接响应]
    classify -->|"rag_qa / technology_selection"| recall[recall_memory<br/>召回长期记忆]
    classify -->|"status=error"| build

    recall -->|"technology_selection"| invoke_sel[invoke_technology_selection<br/>执行技术选型子工作流]
    recall -->|"rag_qa 默认"| invoke_rag[invoke_rag_qa<br/>执行普通问答子工作流]
    recall -->|"status=error"| build

    direct --> build
    invoke_rag --> build
    invoke_sel --> build

    build --> finalize[finalize_short_term_memory<br/>完成短期记忆]
    finalize --> stage[stage_memory_candidates<br/>暂存记忆候选]
    stage --> persist[persist_memory_candidates<br/>持久化记忆候选]
    persist --> END([结束])

    style recall fill:#f0e1ff
    style invoke_sel fill:#e1f5ff
    style invoke_rag fill:#fff4e1
    style direct fill:#e8f5e9
```

> 📌 图为主路由工作流；两个子工作流的详细节点走向、条件路由与轮次上限见 [workflow_diagram.md](workflow_diagram.md)。

**子工作流速览**

| 工作流 | 入口节点 | 主要流程 | 修复机制 |
|---|---|---|---|
| 普通问答 | `invoke_rag_qa` | 分析问题 → 向量+图谱并行检索 → 证据筛选 → 生成回答 → 质量评估 | 改写查询 / 重新生成 / 联网搜索 |
| 技术选型 | `invoke_technology_selection` | 需求解析 → 候选生成 → 硬约束筛选 → 证据检索 → 评分与风险分析 → 推荐审查 → 报告导出 | 需求澄清 / 证据补检 / 审查回退 |

## 🏗️ 系统架构

```
┌─────────────────────────── 前端 (Vue 3 + Vite) ───────────────────────────┐
│                             会话界面 / 进度展示 / 报告下载                    │
└────────────────────────────────────┬──────────────────────────────────────┘
                                     │ REST /api/v1/projects/{id}/sessions
┌────────────────────────────────────▼──────────────────────────────────────┐
│                            FastAPI 后端 (src/api)                           │
│  会话创建 · 消息发送 · 会话查询 · 取消 · 关闭 · 报告下载 · 进度跟踪             │
└────────────────────────────────────┬──────────────────────────────────────┘
                                     │ MemoryAwareRunner（检查点 + 记忆注入）
┌────────────────────────────────────▼──────────────────────────────────────┐
│                      主路由工作流 (src/workflows)                           │
│  短期记忆 → 请求分类 → 长期记忆召回 → 子工作流 → 统一响应 → 记忆持久化          │
│                                    │                                        │
│            ┌───────────────────────┴───────────────────────┐                │
│            ▼                                               ▼                │
│   普通问答工作流 (rag_qa_graph)                  技术选型工作流 (tech_selection)│
│   分析→混合检索→证据筛选→生成→评估           需求解析→候选→证据→评分→审查→报告  │
└────────────────────────────────────┬──────────────────────────────────────┘
                                     │
┌────────────────────────────────────▼──────────────────────────────────────┐
│                         基础设施 (src/infrastructure)                       │
│  DeepSeek/智谱 LLM · BGE/智谱 Embedding · Milvus · Neo4j · MySQL · Tavily  │
│  文档处理（Markdown 解析 · 向量/图谱入库）· 报告导出（Markdown/DOCX）          │
└───────────────────────────────────────────────────────────────────────────┘
```

## 🛠️ 技术栈

| 类别 | 技术 |
|---|---|
| 工作流编排 | LangGraph 1.2.9（StateGraph + 条件边 + 检查点） |
| LLM / Embedding | DeepSeek（`deepseek-v4-flash`）、智谱（`embedding-3`）、BGE（`bge-small-en-v1.5`） |
| 向量库 | Milvus 2.6.9（文档集合 + 长期记忆语义索引） |
| 图数据库 | Neo4j 6.x（知识图谱检索） |
| 检查点 / 记忆 | MySQL（`langgraph-checkpoint-mysql` + 记忆仓库）、SQLite 兜底 |
| 联网搜索 | Tavily 等（`search_pro` 引擎） |
| 后端 API | FastAPI + aiomysql（异步会话执行） |
| 前端 | Vue 3 + TypeScript + Pinia + Vite + Playwright |
| 报告导出 | Markdown / DOCX（python-docx） |
| 日志与观测 | loguru + 工作流进度跟踪（`src/observability`） |

## ⚙️ 关键参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| 查询改写上限 | 2 次 | 普通问答·证据不足时改写查询重试 |
| 生成重试上限 | 2 次 | 普通问答·回答无依据时重新生成 |
| 证据补检轮次 | 3 轮 | 技术选型·证据不足时定向补检 |
| 审查轮次 | 2 轮 | 技术选型·推荐结论审查 |
| 候选方案数 | 3 个 | 技术选型·候选生成上限 |
| 证据任务并发 | 2 | 技术选型·证据检索并发上限 |
| 检索召回数 k | 4 | 向量检索召回文档数 |
| 图谱检索跳数 | 2 | Neo4j 图谱检索最大跳数 |

> 全部参数在 `src/config/config.yaml` 中可调。

## 📁 项目结构

```
├── src/
│   ├── workflows/            # LangGraph 工作流定义（核心）
│   │   ├── query_router_graph.py          # 主路由图（入口 + 记忆编排）
│   │   ├── rag_qa_graph.py                # 普通问答子图
│   │   ├── technology_selection_graph.py  # 技术选型子图
│   │   ├── application.py                 # 生产组合根（MySQL 检查点 + 记忆）
│   │   ├── nodes/                         # 各工作流节点实现
│   │   ├── routes/                        # 条件路由决策函数
│   │   └── graders/                       # 幻觉/相关性/答案质量评估器
│   ├── infrastructure/      # 外部集成
│   │   ├── llm/                          # 大模型与 Embedding 客户端
│   │   ├── vectorstores/                 # Milvus 向量检索
│   │   ├── graphdatabase/                # Neo4j 图谱检索
│   │   ├── memory/                       # 长期记忆仓库（MySQL + Milvus 索引）
│   │   ├── checkpointing/                # MySQL / SQLite 检查点
│   │   ├── document_processing/          # Markdown 解析与向量/图谱入库
│   │   ├── search/                       # 联网搜索
│   │   └── reporting/                    # Markdown / DOCX 报告导出
│   ├── api/                 # FastAPI 接口（会话管理、进度、报告下载）
│   ├── config/              # 配置加载（settings.py + config.yaml）
│   └── observability/       # 日志与工作流进度观测
├── frontend/                # Vue 3 前端（晶圆智选 · Decision Lab）
├── tests/                   # unittest 测试
└── runtime/                 # 运行时产物（日志、检查点、报告）
```

## 🚀 快速开始

### 环境依赖

- Python 3.12+
- Node.js + pnpm（前端）
- 外部服务：Milvus、Neo4j、MySQL，以及 DeepSeek / 智谱 API Key

### 1️⃣ 安装后端依赖

```bash
pip install -r requirements.txt
```

### 2️⃣ 配置环境变量

复制 `.env.example` 为 `.env` 并填写密钥：

```bash
DEEPSEEK_API_KEY=xxx          # 主 LLM（DeepSeek）
ZHIPU_API_KEY=xxx             # Embedding（智谱）
NEO4J_PASSWORD=xxx            # 图数据库密码
MYSQL_CHECKPOINT_URI=xxx      # 检查点（如 mysql+pymysql://user:pass@host/checkpoint_db）
MYSQL_MEMORY_URI=xxx          # 长期记忆仓库
```

非敏感配置（模型、检索参数、轮次上限等）位于 `src/config/config.yaml`。

### 3️⃣ 数据入库

将半导体知识库 Markdown 文档解析并写入向量库与图谱库（见 `src/infrastructure/document_processing/`）。

### 4️⃣ 启动后端

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

启动后访问 `http://localhost:8000/docs` 可查看交互式接口文档。

### 5️⃣ 启动前端

```bash
cd frontend
pnpm install
pnpm dev
```

启动后访问 `http://localhost:5173` 打开「晶圆智选 · Decision Lab」界面。

### ⌨️ 命令行交互（可选）

```bash
python -m src.workflows.query_router_graph --user-id user-001 --project-id project-001
```

> 会话命令：`/exit` 保留会话状态退出；`/close` 删除会话检查点后退出。

## 📚 相关文档

- [workflow_diagram.md](workflow_diagram.md) — 工作流节点 / 边 / 路由决策详细图
- [AGENTS.md](AGENTS.md) — 仓库开发规范
