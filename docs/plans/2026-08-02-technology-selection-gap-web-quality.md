# Technology Selection Gap and Web Quality Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 减少与用户决策无关的“证据不足”，并提高最后一轮 Web 补检的命中率和失败可诊断性，同时保持证据质量门不变。

**Architecture:** 保留现有三轮检索、全部 LangGraph 节点、边、条件路由、入口/出口和公共调用接口。仅修改 `selection_evidence` 节点内部的必需指标判定、第三轮任务拆分与安全进度统计；评分、推荐、直接证据校验及数据库均不变。

**Tech Stack:** Python、LangGraph、Pydantic、`unittest`

---

## Task 1: 将补检缺口限定为决策必需指标

**Files:**

- Modify: `src/workflows/nodes/selection_evidence.py`
- Test: `tests/test_technology_selection_graph.py`

1. 增加失败测试：正权重只有 `performance` 时，其余零权重固定指标不形成补检缺口。
2. 增加失败测试：硬约束明确提到 `power` 时，即使该指标权重为零也必须形成缺口，其他无关零权重指标不形成缺口。
3. 实现确定性必需指标集合：正权重固定指标、全部显式场景指标、硬约束文本明确命中的指标。
4. 保留全部固定指标与场景指标作为证据归属校验目录，仅缩小必须补检的缺口集合。

## Task 2: 第三轮 Web 按候选方案拆分

**Files:**

- Modify: `src/workflows/nodes/selection_evidence.py`
- Test: `tests/test_technology_selection_graph.py`

1. 修改现有失败测试：同一轮 ALD 与 CVD 的 Web 缺口应生成两个候选专属任务。
2. 保持第二轮本地补检合并执行，避免增加本地检索耗时。
3. 第三轮仍然只执行一次 Web 补检，候选专属任务使用现有并发上限执行。

## Task 3: 暴露 Web 失败与有效证据统计

**Files:**

- Modify: `src/workflows/nodes/selection_evidence.py`
- Test: `tests/test_technology_selection_graph.py`

1. 增加失败测试：Web 服务不可用或没有可用结果时，最终 `warnings` 显式说明该 Web 任务未取得结果以及受控错误码。
2. 在证据评估阶段记录本轮 Web 证据数、有效 URL 数和直接支持关系数；不记录查询、证据正文或异常详情。
3. 不缓存失败或空的 Web 结果，不放宽“有效 URL + 可核验原文片段 + 候选/指标直接归属”的评分条件。

## Task 4: 回归验证

1. 单独运行技术选型测试文件。
2. 运行查询路由、图结构、检索图和导入测试。
3. 运行 `compileall` 与主图导入。
4. 验收：三轮上限、节点/边/路由、公共接口、直接证据质量门全部不变。
