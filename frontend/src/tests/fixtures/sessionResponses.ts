import type { ReportData, SessionResponse } from '@/features/sessions/api/sessionSchemas'

export const completedRagResponse: SessionResponse = {
  request_id: '11111111-1111-4111-8111-111111111111',
  project_id: 'project-a',
  thread_id: '22222222-2222-4222-8222-222222222222',
  status: 'completed',
  query_type: 'rag_qa',
  response_type: 'answer',
  answer: 'FinFET 使用鳍式沟道增强栅控能力。',
  report: '',
  report_data: {},
  clarification_questions: [],
  clarification_answers: [],
  clarification_history: [],
  sources: [{ title: 'FinFET Overview', url: 'https://example.com/finfet' }],
  warnings: [],
  report_url: null,
  reports: [],
  progress: {
    request_id: '11111111-1111-4111-8111-111111111111',
    active: false,
    cancel_requested: false,
    cancelled: false,
    current_stage: null,
    events: [
      {
        scope: '入口',
        stage_id: 'classify_query',
        label: '识别请求类型',
        status: 'completed',
        started_at: '2026-08-23T00:00:00+00:00',
        finished_at: '2026-08-23T00:00:00.120000+00:00',
        duration_ms: 120,
      },
      {
        scope: '普通RAG',
        stage_id: 'vector_retrieve',
        label: '向量检索',
        status: 'completed',
        started_at: '2026-08-23T00:00:00.120000+00:00',
        finished_at: '2026-08-23T00:00:00.380000+00:00',
        duration_ms: 260,
      },
    ],
  },
}

export const awaitingSelectionResponse: SessionResponse = {
  ...completedRagResponse,
  request_id: '33333333-3333-4333-8333-333333333333',
  status: 'awaiting_user',
  query_type: 'technology_selection',
  response_type: 'report',
  answer: '',
  clarification_questions: ['目标制程节点是什么？', '功耗和性能哪个优先？'],
}

export function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

export function sseFrame(event: string, data: unknown): string {
  return `event: ${event}
data: ${JSON.stringify(data)}

`
}

/** An SSE response whose chunks the test pushes and closes explicitly. */
export function controlledEventStream() {
  const encoder = new TextEncoder()
  let controller!: ReadableStreamDefaultController<Uint8Array>
  const body = new ReadableStream<Uint8Array>({
    start(value) {
      controller = value
    },
  })
  return {
    response: new Response(body, {
      status: 200,
      headers: { 'Content-Type': 'text/event-stream' },
    }),
    push(chunk: string) {
      controller.enqueue(encoder.encode(chunk))
    },
    close() {
      controller.close()
    },
  }
}

export const validReportData: ReportData = {
  schema_version: '1.0',
  report_version: '2.0',
  conversation: {
    user_request: '比较 FinFET 和 GAA',
    clarification_answers: ['3nm，功耗优先'],
    clarification_history: [],
  },
  decision: {
    status: 'conditional',
    status_label: '有条件推荐',
    primary_candidate_id: 'gaa',
    primary_candidate_name: 'GAA',
    summary: '在满足制造成熟度前提下优先 GAA。',
    decision_basis: 'weighted_score',
    confidence: 'medium',
    confidence_label: '中',
    evidence_coverage: 0.75,
    requires_human_review: true,
    rationale: ['静电控制能力更强'],
    prerequisites: ['验证量产良率'],
    unresolved_items: ['供应链成熟度'],
  },
  requirements: {
    business_goal: '降低移动 SoC 功耗',
    application_scenario: '3nm 移动 SoC',
    hard_constraints: ['量产窗口小于 18 个月'],
    active_criteria: [{ criterion_id: 'power', label: '功耗', weight: 0.6 }],
    inactive_criteria: [{ criterion_id: 'cost', label: '成本' }],
    scenario_criteria: [],
  },
  candidates: [
    { candidate_id: 'gaa', name: 'GAA', summary: '环绕栅结构' },
    { candidate_id: 'finfet', name: 'FinFET', summary: '鳍式沟道结构' },
  ],
  comparison: {
    candidates: [
      { candidate_id: 'gaa', name: 'GAA' },
      { candidate_id: 'finfet', name: 'FinFET' },
    ],
    criteria: [
      {
        criterion_id: 'power',
        label: '功耗',
        weight: '60%',
        cells: [
          {
            candidate_id: 'gaa',
            candidate_name: 'GAA',
            score: 5,
            rationale: '漏电控制较好',
            evidence_ids: ['ev-1'],
            confidence: 'high',
            confidence_label: '高',
          },
          {
            candidate_id: 'finfet',
            candidate_name: 'FinFET',
            score: 3,
            rationale: '成熟但节点扩展受限',
            evidence_ids: ['ev-1'],
            confidence: 'medium',
            confidence_label: '中',
          },
        ],
        judgment: 'GAA领先',
        evidence_status: '证据充分',
      },
    ],
  },
  risks: [
    {
      candidate_id: 'gaa',
      candidate_name: 'GAA',
      category: '制造',
      level: 'medium',
      impact: '工艺窗口收窄',
      mitigation: '增加试产验证',
      evidence_ids: ['ev-1'],
      basis: 'evidence',
      basis_label: '相关证据',
    },
  ],
  evidence_gaps: [{ priority: '高', description: '缺少量产良率数据' }],
  evidence: [
    {
      evidence_id: 'ev-1',
      title: 'GAA Process Review',
      source_type: 'web_identified',
      source_label: '有来源网页',
      display_source: 'https://example.com/evidence',
      url: 'https://example.com/evidence',
      excerpt: 'GAA 在先进节点具有更强栅控能力。',
      supports: [
        {
          candidate_id: 'gaa',
          candidate_name: 'GAA',
          criterion_id: 'power',
          criterion_label: '功耗',
          level: 'direct',
          level_label: '直接证据',
          quote: 'Measured leakage was reduced.',
        },
      ],
    },
  ],
  audit: {
    status: 'needs_human_review',
    review_verdict: 'pass',
    review_label: '通过',
    review_reasons: [],
    evidence_round: 2,
    review_round: 1,
    warnings: [],
  },
}
