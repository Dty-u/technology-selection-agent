import { describe, expect, it } from 'vitest'

import { parseReportData, parseSessionResponse } from '../sessionSchemas'

const completedResponse = {
  request_id: '11111111-1111-4111-8111-111111111111',
  project_id: 'project-a',
  thread_id: '22222222-2222-4222-8222-222222222222',
  status: 'completed',
  query_type: 'technology_selection',
  response_type: 'report',
  answer: '',
  report: '# 半导体技术选型报告',
  report_data: {
    schema_version: '1.0',
    report_version: '2.0',
    conversation: {
      user_request: '为移动 SoC 选择晶体管结构',
      clarification_answers: ['3nm，低功耗优先'],
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
      requires_human_review: false,
      rationale: ['静电控制能力更强'],
      prerequisites: ['验证良率'],
      unresolved_items: ['供应链成熟度'],
    },
    requirements: {
      business_goal: '降低移动 SoC 功耗',
      application_scenario: '3nm 移动 SoC',
      hard_constraints: ['量产窗口小于 18 个月'],
      active_criteria: [{ criterion_id: 'power', label: '功耗', weight: 0.6 }],
      inactive_criteria: [],
      scenario_criteria: [],
    },
    candidates: [{ candidate_id: 'gaa', name: 'GAA', summary: '环绕栅结构' }],
    comparison: {
      candidates: [{ candidate_id: 'gaa', name: 'GAA' }],
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
      status: 'completed',
      review_verdict: 'pass',
      review_label: '通过',
      review_reasons: [],
      evidence_round: 2,
      review_round: 1,
      warnings: [],
    },
    future_field: 'must remain accepted',
  },
  clarification_questions: [],
  clarification_answers: ['3nm，低功耗优先'],
  clarification_history: [],
  sources: [],
  warnings: [],
  report_url: '/api/v1/projects/project-a/sessions/222/report',
  future_top_level_field: true,
}

describe('sessionSchemas', () => {
  it('accepts the current FastAPI response and keeps future fields compatible', () => {
    const parsed = parseSessionResponse(completedResponse)

    expect(parsed.status).toBe('completed')
    expect(parsed.query_type).toBe('technology_selection')
    expect(parsed.report_data.future_field).toBe('must remain accepted')
  })

  it('rejects unsupported workflow statuses at the HTTP boundary', () => {
    expect(() =>
      parseSessionResponse({ ...completedResponse, status: 'secret_internal_state' }),
    ).toThrow(/status/i)
  })

  it('parses a complete report_data v1 contract', () => {
    const report = parseReportData(completedResponse.report_data)

    expect(report?.decision.primary_candidate_name).toBe('GAA')
    expect(report?.comparison.criteria[0]?.cells[0]?.score).toBe(5)
    expect(report?.evidence[0]?.supports[0]?.level).toBe('direct')
  })

  it('returns null for incomplete report_data so the UI can fall back to markdown', () => {
    expect(parseReportData({ schema_version: '1.0', decision: {} })).toBeNull()
  })
})
