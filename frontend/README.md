# 晶圆智选前端工作台

面向个人使用的半导体 RAG 与技术选型工作台。前端只调用现有 `src/api`，不会复制 LangGraph、checkpoint、记忆或证据评分逻辑。

## 功能边界

- 普通 RAG 问答与来源检查
- 后端真实节点执行轨迹；所有会话处理期间通过 GET 短轮询更新
- 同一 `thread_id` 内的技术选型澄清和恢复
- `report_data` V1 结构化报告、证据缺口、风险和审计信息
- 带 `X-User-ID` 的 DOCX 下载
- 最多 20 个按 `user_id + project_id` 隔离的本机会话索引
- API 存活状态仅表示 `/health/live` 可达，不代表数据库或外部依赖全部正常

不包含登录、知识库管理、索引重建、长期记忆管理、人工批准和虚构的节点进度。

## 环境要求

- Node.js `24.12+` 或 `22.18+`
- pnpm
- Rag_Agent Python 环境与所需后端服务

## 安装与启动

在仓库根目录启动后端：

```powershell
D:\py_venv\rag_venv\Scripts\python.exe -m uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8100
```

另开 PowerShell 启动前端：

```powershell
cd frontend
Copy-Item .env.example .env.local
pnpm install
pnpm dev
```

打开 `http://localhost:5173`。首次进入时填写 API Base URL、`user_id` 和 `project_id`。

## 配置与本地数据

`VITE_API_BASE_URL` 默认是 `http://localhost:8100`，可在首次设置页覆盖。设置使用键 `rag-agent.settings.v1`；会话使用按用户和项目编码的 `rag-agent.sessions.v1:*` 键。

浏览器只保存工作区设置、`thread_id`、会话元数据和展示用对话，不保存 DOCX Blob、API Key、数据库连接或服务端长期记忆。删除浏览器数据不会关闭服务端会话；“关闭会话”和“仅移除本机记录”是两个不同操作。

## 验证

```powershell
cd frontend
pnpm lint
pnpm type-check
pnpm test:unit --run
pnpm exec playwright install chromium  # 仅首次运行需要
pnpm test:e2e
pnpm build
```

Playwright 会拦截并模拟 FastAPI 响应，不连接真实 LLM、Milvus、Neo4j、MySQL 或 Web Search。

后端回归测试：

```powershell
cd ..
D:\py_venv\rag_venv\Scripts\python.exe -m unittest discover -s tests -p "test_api.py" -v
```

## 请求行为

- 所有会话请求带 `X-User-ID`；POST 还会生成新的 `X-Request-ID`。创建会话时浏览器预分配 UUID，并通过 `X-Thread-ID` 交给后端校验和使用。
- POST 不自动重试。已有会话的网络结果不明确时，可使用“同步最新状态”发起 GET。
- 首次和后续会话都会在 POST 等待期间短轮询同一 GET 路由，展示后端 `logged_node` 实际产生的阶段；来源在工作流产生并持久化后同步更新。
- “退出本次请求”会请求服务端在当前工作流节点结束后停止后续步骤；会话、既有消息和 checkpoint 会被保留。
- 响应不满足 `report_data` V1 时，界面回退为 Markdown 报告并显示契约警告。
