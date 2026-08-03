# Workflow Reliability Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make evidence failure, incomplete evidence, candidate limits, and response status truthful without changing LangGraph node names/counts, edges, entry/exit, or execution order.

**Architecture:** Keep the existing three compiled graphs unchanged. Add only state fields and node-internal guards: failed Web retrieval produces no generatable evidence, evidence assessment failures retain retrieved evidence and create gaps, incomplete selection evidence cannot name a preferred candidate, and over-limit requested candidates are reported rather than silently discarded.

**Tech Stack:** Python 3.13, LangGraph, Pydantic, LangChain, unittest.

---

### Task 1: Make ordinary-RAG failure status truthful

**Files:**
- Modify: `src/workflows/nodes/web_search.py`
- Modify: `src/workflows/nodes/query_router.py`
- Test: `tests/test_web_search.py`
- Test: `tests/test_query_router_graph.py`

**Step 1: Write failing tests**

Add tests that a failed/empty Web search returns no evidence plus a `retrieval_error`, and that the unified response preserves `error` rather than overwriting it with `completed`.

**Step 2: Run the focused tests and verify failure**

Run: `D:\py_venv\rag_venv\Scripts\python.exe -m unittest tests.test_web_search tests.test_query_router_graph -v`

**Step 3: Implement the minimal guards**

Return `documents=[]`, `retrieval_error`, and an error status from `web_search` for provider failure or empty usable results. In `build_query_response`, map a child retrieval error to the public error result before generating normal RAG output.

**Step 4: Re-run focused tests**

Expected: all focused tests pass.

### Task 2: Keep selection evidence failures auditable

**Files:**
- Modify: `src/workflows/nodes/selection_evidence.py`
- Test: `tests/test_technology_selection_graph.py`

**Step 1: Write failing test**

Mock `evidence_assessment_chain` to fail after retrieval succeeds; assert that retrieved records are retained, all candidate/criterion coverage is reported as gaps, and a sanitized warning is returned.

**Step 2: Run the focused test and verify failure**

Run: `D:\py_venv\rag_venv\Scripts\python.exe -m unittest tests.test_technology_selection_graph.EvidenceNodeTest -v`

**Step 3: Implement the minimal fallback**

Catch assessment/parser failures inside `retrieve_evidence`; clear unverified candidate/criterion attribution, derive gaps deterministically for every candidate and criterion, and add only an exception type to warnings.

**Step 4: Re-run focused tests**

Expected: all evidence-node tests pass.

### Task 3: Prevent unsupported preferred selections and silent candidate loss

**Files:**
- Modify: `src/workflows/nodes/selection_requirements.py`
- Modify: `src/workflows/nodes/selection_decision.py`
- Test: `tests/test_technology_selection_graph.py`

**Step 1: Write failing tests**

Add tests that requested candidates exceeding the configured limit leave an explicit warning, and that incomplete evaluation produces `conditional` or `insufficient_evidence` with no `recommended_candidate_id`.

**Step 2: Run the focused tests and verify failure**

Run: `D:\py_venv\rag_venv\Scripts\python.exe -m unittest tests.test_technology_selection_graph.RequirementAndCandidateNodeTest tests.test_technology_selection_graph.DecisionNodeTest -v`

**Step 3: Implement the minimal guards**

Record candidate-limit truncation in state warnings. When evaluation is incomplete, clear the recommended candidate; retain the report's unresolved evidence items. Do not alter routes or graph topology.

**Step 4: Re-run focused tests**

Expected: all requirement and decision-node tests pass.

### Task 4: Verify topology and regression behavior

**Files:**
- Test only

**Step 1: Compile sources**

Run: `D:\py_venv\rag_venv\Scripts\python.exe -m compileall -q src tests`

**Step 2: Run isolated test processes**

Run separately:

```powershell
D:\py_venv\rag_venv\Scripts\python.exe -m unittest discover -s tests -p "test_query_router_graph.py" -v
D:\py_venv\rag_venv\Scripts\python.exe -m unittest discover -s tests -p "test_technology_selection_graph.py" -v
D:\py_venv\rag_venv\Scripts\python.exe -m unittest discover -s tests -p "test_workflow_graph_structure.py" -v
```

**Step 3: Confirm topology**

Verify the existing structure tests pass, proving that nodes, edges, entry/exit, and execution order were not changed.
