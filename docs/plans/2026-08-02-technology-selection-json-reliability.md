# 技术选型证据归属 JSON 可靠性 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在不改变技术选型图拓扑和质量门的前提下，避免一次证据归属 JSON 格式错误使整轮证据失效，并减少由此触发的无效审核循环。

**Architecture:** 证据归属调用单独启用模型原生 JSON 模式；解析失败时只重试一次，成功结果才进入严格缓存。多批证据分别隔离失败，成功批次继续形成直接支持关系，失败批次保留原文但不参与评分；审核上限仍作为真实质量问题的保护门，不通过增加轮次或放宽证据标准来隐藏问题。

**Tech Stack:** Python、LangChain `ChatOpenAI`、LangGraph、Pydantic、`unittest`

---

## 边界

- 不修改节点名称、节点数量、边、条件路由、入口/出口和执行顺序。
- 不修改 `query_router_graph.invoke({"query": ...})` 接口和公共返回结构。
- 不修改 Milvus、Neo4j、Embedding、知识库数据、普通 RAG 模型和检索预算。
- 不提高证据轮数或审核轮数，不降低“直接证据才能评分”的规则。
- 仅修改技术选型 JSON 模型配置、证据归属节点内部逻辑及对应测试。

### Task 1: 用回归测试复现 JSON 格式失败

**Files:**
- Modify: `tests/test_technology_selection_graph.py`

**Step 1: Write the failing test**

新增测试：证据归属第一次返回非法 JSON、第二次返回合法 JSON 时，本批证据应恢复正常归属并参与覆盖率计算；模型只调用两次。

**Step 2: Run test to verify it fails**

Run: `D:\py_venv\rag_venv\Scripts\python.exe -m unittest tests.test_technology_selection_graph.EvidenceNodeTest.test_evidence_assessment_retries_once_after_invalid_json -v`

Expected: FAIL，当前实现只调用一次模型并直接抛出 `JSONDecodeError`。

### Task 2: 用回归测试限定批次失败影响范围

**Files:**
- Modify: `tests/test_technology_selection_graph.py`

**Step 1: Write the failing test**

新增测试：两个证据评估批次中一个持续返回非法 JSON 时，成功批次仍保留经过校验的直接支持；失败批次保留证据但清空评分归属，并产生一条受控 warning。

**Step 2: Run test to verify it fails**

Run: `D:\py_venv\rag_venv\Scripts\python.exe -m unittest tests.test_technology_selection_graph.EvidenceNodeTest.test_one_failed_assessment_batch_does_not_discard_successful_batch -v`

Expected: FAIL，当前任一批次异常会进入整轮 `except`，清空全部新增证据的归属。

### Task 3: 实施最小生产修改

**Files:**
- Modify: `src/infrastructure/llm/chat_model.py`
- Modify: `src/workflows/nodes/selection_evidence.py`

**Step 1: Bind JSON mode only for structured selection calls**

在保留 `llm` 和 `selection_llm` 现有行为的基础上增加 `selection_json_llm`，只追加 `response_format={"type": "json_object"}`。证据归属链使用该模型，普通 RAG 不变。

**Step 2: Retry only malformed model JSON once**

在 `_assess_evidence_batch` 内只捕获 `JSONDecodeError`/缺少 JSON 对象的解析失败，记录不含提示词和证据正文的安全诊断信息，然后原输入重试一次。第二次仍失败则向上交给批次隔离逻辑；网络错误和其他程序错误不盲目重试。

**Step 3: Isolate failures per assessment batch**

逐批捕获评估异常并返回“批次、成功映射、异常”三元结果。成功批次继续执行原有 `_validated_supports`；失败批次仅清空自身新增证据的归属。缺口仍从最终直接支持关系确定性派生。

**Step 4: Preserve the quality gate**

不删除或弱化“推荐审查达到上限”warning。上游恢复后它不应再因整轮误清空而被放大；若推荐仍有真实证据缺口或一致性冲突，继续进入人工复核。

### Task 4: 验证回归与结构不变

**Files:**
- Test: `tests/test_technology_selection_graph.py`
- Test: `tests/test_query_router_graph.py`
- Test: `tests/test_workflow_graph_structure.py`
- Test: `tests/test_imports.py`

**Step 1: Run new tests**

Run: `D:\py_venv\rag_venv\Scripts\python.exe -m unittest tests.test_technology_selection_graph.EvidenceNodeTest.test_evidence_assessment_retries_once_after_invalid_json tests.test_technology_selection_graph.EvidenceNodeTest.test_one_failed_assessment_batch_does_not_discard_successful_batch -v`

Expected: PASS。

**Step 2: Run focused technology-selection tests**

Run: `D:\py_venv\rag_venv\Scripts\python.exe -m unittest tests.test_technology_selection_graph -v`

Expected: PASS。

**Step 3: Run each regression file in a separate process**

Run separately:

- `D:\py_venv\rag_venv\Scripts\python.exe -m unittest tests.test_query_router_graph -v`
- `D:\py_venv\rag_venv\Scripts\python.exe -m unittest tests.test_workflow_graph_structure -v`
- `D:\py_venv\rag_venv\Scripts\python.exe -m unittest tests.test_imports -v`

Expected: PASS。

**Step 4: Compile and import the public graph**

Run:

- `D:\py_venv\rag_venv\Scripts\python.exe -m compileall -q src tests`
- `D:\py_venv\rag_venv\Scripts\python.exe -c "from src.workflows.query_router_graph import query_router_graph; print(type(query_router_graph).__name__)"`

Expected: 编译无输出，主图输出 `CompiledStateGraph`。

## 验收结果记录

- [x] 第一次非法 JSON、第二次合法 JSON 时自动恢复，失败响应不进入缓存。
- [x] 单个批次持续失败时，其他成功批次的证据归属不丢失。
- [x] 失败批次证据仍保留，但没有候选、指标或直接支持，不参与评分。
- [x] warning 不包含用户问题、证据正文、模型响应或密钥。
- [x] 审核上限与直接证据质量门保持不变。
- [x] 图结构、公共接口和普通 RAG 行为保持不变。

## 实施验证（2026-08-02）

- 新增 3 个回归测试，均完成 RED → GREEN：JSON 模式请求、一次解析重试、失败批次隔离。
- `tests.test_technology_selection_graph`：64 项通过。
- `tests.test_query_router_graph`：34 项通过。
- `tests.test_workflow_graph_structure`：4 项通过。
- `tests.test_imports`：6 项通过。
- 未执行真实外部模型、Milvus、Neo4j 和 Web 的付费/联网端到端调用；本轮验证使用 Mock 与只读导入检查。
