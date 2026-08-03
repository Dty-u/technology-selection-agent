# 技术选型三阶段证据检索 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将技术选型证据获取改为“本地初检 → 本地定向补检 → Web定向补检”，在只增加必要兜底检索的前提下提高证据覆盖率。

**Architecture:** 保留现有 LangGraph 节点和边，通过证据轮次状态决定现有任务的 `source_preference`。初始任务和第一次补检使用 `auto` 并关闭检索子图的提前Web兜底，只执行本地混合检索及一次本地改写；第二次补检将剩余真实缺口强制标记为 `web`。第三轮结束后进入既有评分和人工复核质量门。

**Tech Stack:** Python、LangGraph、LangChain、Pydantic、`unittest`

---

## 修改边界

- 最大证据检索轮数由2改为3：1轮初检、2轮补检。
- 初始两个证据包均本地优先；第一次补检的全部缺口仍本地优先。
- 第二次补检只包含仍缺失的“候选方案 × 指标”，并强制走Web。
- 不修改节点名称、数量、边、条件路由目标、入口/出口和公共接口。
- 不修改直接证据、有效URL、原文片段、评分和审核质量规则。
- 不修改普通RAG预算、数据库、Embedding和知识库数据。

### Task 1: 固化三阶段路由行为

**Files:**
- Modify: `tests/test_technology_selection_graph.py`

**Step 1: Write failing route tests**

验证证据不足时：`evidence_round=1`进入第一次补检，`evidence_round=2`进入第二次补检，`evidence_round=3`才进入评分。

**Step 2: Run tests and verify RED**

Run: `D:\py_venv\rag_venv\Scripts\python.exe -m unittest tests.test_technology_selection_graph.TechnologySelectionRouteTest.test_routes_read_state_without_calling_models -v`

Expected: 当前 `evidence_round=2`提前进入评分，测试失败。

### Task 2: 固化每一轮的检索来源

**Files:**
- Modify: `tests/test_technology_selection_graph.py`

**Step 1: Write failing source tests**

验证：初始技术包和制造包都是 `auto`；`evidence_round=1`生成的第一次补检任务都是 `auto`；`evidence_round=2`生成的第二次补检任务都是 `web`，且只包含传入的真实缺口。

**Step 2: Run tests and verify RED**

Run: `D:\py_venv\rag_venv\Scripts\python.exe -m unittest tests.test_technology_selection_graph.EvidenceNodeTest.test_plan_evidence_tasks_uses_deterministic_pack_sources tests.test_technology_selection_graph.EvidenceNodeTest.test_optimize_queries_merges_same_source_candidate_gaps tests.test_technology_selection_graph.EvidenceNodeTest.test_second_retry_forces_remaining_gaps_to_web -v`

Expected: 当前制造包和成本缺口提前走Web，且不存在第二次补检强制Web规则，测试失败。

### Task 3: 实施最小生产修改

**Files:**
- Modify: `src/config/config.yaml`
- Modify: `src/workflows/nodes/selection_evidence.py`
- Modify: `src/workflows/routes/technology_selection.py`
- Modify: `src/workflows/technology_selection_state.py`

**Step 1: Increase the bounded evidence rounds**

将 `selection.max_evidence_rounds` 从2改为3。现有路由读取该配置，无需改变图结构。

**Step 2: Keep initial and first retry local-first**

初始证据包统一赋值 `auto`；第一次补检不再按指标类型提前分流Web。技术选型本地任务向检索子图传入 `allow_web_fallback=False`，本地完全无结果时结束本轮，不在轮内提前联网；普通RAG默认保持原有Web兜底。

**Step 3: Force only the second retry to Web**

`optimize_evidence_queries` 根据当前 `evidence_round` 决定来源：小于2为 `auto`，等于或大于2为 `web`。任务仍只从 `missing_evidence_queries` 生成。

**Step 4: Preserve the final quality gate**

第三轮后仍有缺口时继续输出空分、条件性结论或 `needs_human_review`，不新增第四轮。

### Task 4: 回归验证

**Files:**
- Test: `tests/test_technology_selection_graph.py`
- Test: `tests/test_query_router_graph.py`
- Test: `tests/test_workflow_graph_structure.py`
- Test: `tests/test_imports.py`

**Step 1: Run focused tests**

Run: `D:\py_venv\rag_venv\Scripts\python.exe -m unittest tests.test_technology_selection_graph -v`

Expected: 全部通过。

**Step 2: Run separate regression processes**

- `D:\py_venv\rag_venv\Scripts\python.exe -m unittest tests.test_query_router_graph -v`
- `D:\py_venv\rag_venv\Scripts\python.exe -m unittest tests.test_workflow_graph_structure -v`
- `D:\py_venv\rag_venv\Scripts\python.exe -m unittest tests.test_imports -v`

Expected: 全部通过。

**Step 3: Compile and import**

- `D:\py_venv\rag_venv\Scripts\python.exe -m compileall -q src tests`
- `D:\py_venv\rag_venv\Scripts\python.exe -c "from src.workflows.query_router_graph import query_router_graph; print(type(query_router_graph).__name__)"`

Expected: 编译成功，输出 `CompiledStateGraph`。

## 验收清单

- [x] 最多3轮：本地初检、本地补检、Web补检。
- [x] 第一轮证据充足时直接评分，不执行后续补检。
- [x] 第二轮证据充足时直接评分，不执行Web补检。
- [x] Web补检只查询第二轮后仍存在的缺口。
- [x] 第三轮后停止检索，真实缺口继续进入人工复核。
- [x] 图结构、公共接口、普通RAG和质量门保持不变。

## 实施验证（2026-08-02）

- 三阶段来源与停止逻辑相关测试完成 RED → GREEN。
- `tests.test_technology_selection_graph`：65项通过。
- `tests.test_query_router_graph`：34项通过。
- `tests.test_workflow_graph_structure`：4项通过。
- `tests.test_retrieval_graph`：5项通过，普通RAG默认Web兜底保持不变。
- `tests.test_imports`：6项通过。
- 未执行真实外部模型、Milvus、Neo4j与Web付费端到端调用；自动化验证使用Mock和只读导入检查。
