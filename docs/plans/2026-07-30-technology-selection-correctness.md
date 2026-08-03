# Technology Selection Correctness Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 修复技术选型工作流中会直接导致错误推荐、跨候选证据串用和无效补检的三个正确性缺口。

**Architecture:** 保留现有 LangGraph 节点和状态结构，只在决策校验与已有证据重试路径上增加确定性约束。LLM 继续负责抽取与解释，Python 负责候选归属、评分排序、结论等级和路由输入的硬校验。

**Tech Stack:** Python 3、Pydantic 2、LangChain、LangGraph、`unittest`

---

## 审查结论与执行边界

当前工作流主链已经完整：

`需求解析 → 候选生成 → 硬约束筛选 → 分方案检索 → 证据映射 → 评分 → 风险 → 推荐 → 复核 → 报告`

本轮只修改直接影响结论正确性的共享根因：

1. 风险引用必须属于对应候选方案。
2. 无条件推荐不得与完整加权评分排序冲突，备选 ID 必须有效。
3. 复核器提出的补检问题必须进入已有的定向补检节点，复核时必须能看到证据摘要。

本轮不修改以下内容：

- 不重建 Milvus 或 Neo4j，不切换 Embedding 模型。
- 不增加新的检索器、数据库、Agent 或抽象层。
- 不调整普通 RAG 问答行为。
- 不改变公开入口的会话协议；可续跑补问需要 `thread_id`/checkpointer，属于单独的 API 契约变更。
- 不重做证据数据模型；来源时间、引用片段、冲突证据与质量分级放入后续阶段。
- 不修改当前 Conda 环境。当前环境与 `requirements.txt` 不一致，完整图测试需在隔离环境按依赖文件重建后运行。

## Task 1: 限制风险证据的候选归属

**Files:**

- Modify: `tests/test_technology_selection_graph.py`
- Modify: `src/workflows/nodes/selection_decision.py:173-207`

**Step 1: Write the failing test**

增加测试：GAA 风险引用只属于 FinFET 的证据时，证据 ID 被移除，`basis` 降级为 `to_verify`。

**Step 2: Run test to verify it fails**

Run:

```powershell
$env:PYTHONPATH='tests'
python -m unittest test_technology_selection_graph.DecisionNodeTest.test_risk_evidence_must_belong_to_the_same_candidate -v
```

Expected: FAIL，当前实现仍保留跨候选证据并标记为 `evidence`。

**Step 3: Write minimal implementation**

构造 `evidence_id -> candidate_ids` 映射；风险证据只保留包含 `risk.candidate_id` 的记录。原有“无有效引用则降级为 `to_verify`”规则保持不变。

**Step 4: Run test to verify it passes**

重复 Step 2 命令，Expected: PASS。

## Task 2: 约束推荐与评分排序

**Files:**

- Modify: `tests/test_technology_selection_graph.py`
- Modify: `src/workflows/nodes/selection_decision.py:210-263`

**Step 1: Write the failing tests**

增加两个测试：

1. 所有候选都有完整评分时，如果模型无条件推荐较低分方案，结论降级为 `conditional`，并加入人工复核待验证项。
2. `alternative_candidate_ids` 只保留可行候选且不得重复首选方案。

**Step 2: Run tests to verify they fail**

Run:

```powershell
$env:PYTHONPATH='tests'
python -m unittest test_technology_selection_graph.DecisionNodeTest.test_unconditional_recommendation_cannot_conflict_with_score_ranking test_technology_selection_graph.DecisionNodeTest.test_recommendation_filters_invalid_alternative_ids -v
```

Expected: FAIL，当前实现接受较低分首选和无效备选 ID。

**Step 3: Write minimal implementation**

在现有完整性检查之后：

- 过滤无效或等于首选的备选 ID。
- 仅当存在唯一最高分且模型给出 `recommend` 时检查排序。
- 若首选不是唯一最高分方案，将结论降级为 `conditional`，保留模型首选供人工判断，并加入固定待验证项。
- 并列最高分不自动替模型选方案，避免无依据覆盖专业判断。

**Step 4: Run tests to verify they pass**

重复 Step 2 命令，Expected: PASS。

## Task 3: 让复核补检请求进入定向检索

**Files:**

- Modify: `tests/test_technology_selection_graph.py`
- Modify: `src/workflows/nodes/selection_decision.py:71-85,266-316`
- Modify: `src/workflows/routes/technology_selection.py:49-57`
- Modify: `src/workflows/technology_selection_graph.py:80-87`

**Step 1: Write the failing tests**

增加测试：

1. 复核调用参数包含截断后的证据内容。
2. `evidence_gap` 的 `requested_queries` 被写入 `missing_evidence_queries`。
3. 复核路由返回现有 `optimize_evidence_queries` 节点，而不是重新生成通用任务。

**Step 2: Run tests to verify they fail**

Run:

```powershell
$env:PYTHONPATH='tests'
python -m unittest test_technology_selection_graph.DecisionNodeTest.test_review_receives_evidence_and_persists_requested_queries test_technology_selection_graph.TechnologySelectionRouteTest.test_routes_read_state_without_calling_models -v
```

Expected: FAIL，当前复核看不到证据，补检请求未保存，路由返回 `plan_evidence_tasks`。

**Step 3: Write minimal implementation**

- 给复核提示增加 `evidence` 输入，只传每条证据最多 2000 字。
- 当 verdict 为 `evidence_gap` 时，将去重后的 `requested_queries` 写入 `missing_evidence_queries` 和 `evidence_gaps`。
- 将该路由目标改为已有的 `optimize_evidence_queries`，复用现有定向任务生成逻辑。

**Step 4: Run tests to verify they pass**

重复 Step 2 命令，Expected: PASS。

## Task 4: Verification

**Files:**

- Verify only

**Step 1: Run syntax validation**

```powershell
python -m compileall -q src tests
```

Expected: exit code 0。

**Step 2: Run focused regression tests**

```powershell
$env:PYTHONPATH='tests'
python -m unittest test_technology_selection_graph.DecisionNodeTest test_technology_selection_graph.TechnologySelectionRouteTest -v
```

Expected: 所有目标测试通过。

**Step 3: Run dependency-independent repository checks**

```powershell
python -m unittest discover -s tests -p "test_imports.py" -v
python -m unittest discover -s tests -p "test_retriever.py" -v
python -m unittest discover -s tests -p "test_milvus.py" -v
```

Expected: 全部通过；允许现有 protobuf 版本警告，但不得出现测试失败。

**Step 4: Record the known environment blocker**

```powershell
python -m pip check
```

若仍显示 `langgraph-checkpoint` 与 `ormsgpack` 不兼容，则不得宣称完整 LangGraph 端到端测试通过。后续应在隔离虚拟环境按 `requirements.txt` 重建后运行：

```powershell
python -m unittest discover -s tests -p "test_technology_selection_graph.py" -v
python -m unittest discover -s tests -p "test_query_router_graph.py" -v
```

## 后续阶段（本轮不执行）

1. 使用 LangGraph checkpointer/interrupt 和 `thread_id` 实现补问续跑。
2. 将证据升级为“主张、引用片段、位置、来源、发布时间、访问时间、支持/反对关系”的可审计结构。
3. 为五个评分维度定义 1–5 分锚点，并为决策类调用配置确定性模型参数。
4. 建立 FinFET/GAA、EUV/DUV、ALD/CVD 的选型金标准案例，评估约束满足率、引用支持率、推荐一致率和证据不足拒绝率。
5. 正确性稳定后，再评估证据任务并行化与上下文裁剪。

## 执行记录（2026-07-30）

- [x] Task 1：跨候选风险证据回归测试完成，最小修复完成。
- [x] Task 2：推荐与评分排序冲突、无效备选 ID 回归测试完成，最小修复完成。
- [x] Task 3：复核证据输入、补检请求持久化和定向补检路由完成。
- [x] `python -m compileall -q src tests` 通过。
- [x] `DecisionNodeTest` 与 `TechnologySelectionRouteTest` 共 9 项通过。
- [x] `test_imports.py` 6 项、`test_retriever.py` 5 项、`test_milvus.py` 1 项通过。
- [x] 临时隔离依赖验证：使用系统临时目录中的 `ormsgpack 1.12.2`，完整 `test_technology_selection_graph.py` 23 项通过。
- [x] 同一隔离验证下，完整 `test_query_router_graph.py` 17 项通过。
- [ ] 默认 Conda 环境仍无法直接运行完整 LangGraph 测试；未修改该共享环境。

已确认的环境冲突：

```text
langgraph-checkpoint 4.1.1 requires ormsgpack>=1.12.0,
but the current environment has ormsgpack 1.9.1.
```
