import { z } from 'zod'

const workflowStatusSchema = z.enum([
  'running',
  'completed',
  'awaiting_user',
  'needs_human_review',
  'cancelled',
  'error',
])

const queryTypeSchema = z.enum(['direct_response', 'rag_qa', 'technology_selection'])
const responseTypeSchema = z.enum(['answer', 'report'])
const confidenceSchema = z.enum(['low', 'medium', 'high'])
const dictionarySchema = z.record(z.string(), z.unknown())

const sourceSchema = z
  .object({
    title: z.string().optional(),
    url: z.string().optional(),
    content: z.string().optional(),
    source: z.string().optional(),
  })
  .passthrough()

const workflowStageSchema = z.object({
  scope: z.string(),
  stage_id: z.string(),
  label: z.string(),
})

const workflowProgressEventSchema = workflowStageSchema.extend({
  status: z.enum(['running', 'completed', 'paused', 'error']),
  started_at: z.string(),
  finished_at: z.string().nullable(),
  duration_ms: z.number().int().nonnegative().nullable(),
})

const workflowProgressSchema = z.object({
  request_id: z.string().nullable(),
  active: z.boolean(),
  cancel_requested: z.boolean().optional().default(false),
  cancelled: z.boolean().optional().default(false),
  current_stage: workflowStageSchema.nullable(),
  events: z.array(workflowProgressEventSchema),
})

const cancelSessionResponseSchema = z.object({
  thread_id: z.string(),
  request_id: z.string().nullable(),
  status: z.enum(['cancelling', 'cancelled', 'not_running']),
})

const reportSummarySchema = z.object({
  report_id: z.string(),
  title: z.string(),
  created_at: z.string(),
  format: z.enum(['markdown', 'docx']),
  download_url: z.string(),
})

export const sessionResponseSchema = z
  .object({
    request_id: z.string().nullable(),
    project_id: z.string(),
    thread_id: z.string(),
    status: workflowStatusSchema,
    query_type: queryTypeSchema.nullable(),
    response_type: responseTypeSchema.nullable(),
    answer: z.string(),
    report: z.string(),
    report_data: dictionarySchema,
    clarification_questions: z.array(z.string()),
    clarification_answers: z.array(z.string()),
    clarification_history: z.array(dictionarySchema),
    sources: z.array(sourceSchema),
    warnings: z.array(z.string()),
    report_url: z.string().nullable(),
    reports: z.array(reportSummarySchema).default([]),
    progress: workflowProgressSchema.optional(),
  })
  .passthrough()

const criterionSchema = z
  .object({
    criterion_id: z.string(),
    label: z.string(),
  })
  .passthrough()

const comparisonCellSchema = z
  .object({
    candidate_id: z.string(),
    candidate_name: z.string(),
    score: z.number().nullable(),
    rationale: z.string(),
    evidence_ids: z.array(z.string()),
    confidence: confidenceSchema,
    confidence_label: z.string(),
  })
  .passthrough()

const comparisonCriterionSchema = z
  .object({
    criterion_id: z.string(),
    label: z.string(),
    weight: z.string(),
    cells: z.array(comparisonCellSchema),
    judgment: z.string(),
    evidence_status: z.string(),
  })
  .passthrough()

const evidenceSupportSchema = z
  .object({
    candidate_id: z.string(),
    candidate_name: z.string(),
    criterion_id: z.string(),
    criterion_label: z.string(),
    level: z.enum(['direct', 'indirect', 'none']),
    level_label: z.string(),
    quote: z.string(),
  })
  .passthrough()

const evidenceSchema = z
  .object({
    evidence_id: z.string(),
    title: z.string(),
    source_type: z.enum(['internal', 'web_identified', 'web_unattributed']),
    source_label: z.string(),
    display_source: z.string(),
    url: z.string(),
    excerpt: z.string(),
    supports: z.array(evidenceSupportSchema),
  })
  .passthrough()

const riskSchema = z
  .object({
    candidate_id: z.string(),
    candidate_name: z.string(),
    category: z.string(),
    level: z.enum(['low', 'medium', 'high', 'unknown']),
    impact: z.string(),
    mitigation: z.string(),
    evidence_ids: z.array(z.string()),
    basis: z.enum(['evidence', 'inference', 'to_verify']),
    basis_label: z.string(),
  })
  .passthrough()

export const reportDataSchema = z
  .object({
    schema_version: z.literal('1.0'),
    report_version: z.literal('2.0'),
    conversation: z
      .object({
        user_request: z.string(),
        clarification_answers: z.array(z.string()),
        clarification_history: z.array(dictionarySchema),
      })
      .passthrough(),
    decision: z
      .object({
        status: z.enum(['recommend', 'conditional', 'insufficient_evidence']),
        status_label: z.string(),
        primary_candidate_id: z.string().nullable(),
        primary_candidate_name: z.string(),
        summary: z.string(),
        decision_basis: z.enum(['weighted_score', 'scenario_criteria', 'insufficient_evidence']),
        confidence: confidenceSchema,
        confidence_label: z.string(),
        evidence_coverage: z.number().min(0).max(1),
        requires_human_review: z.boolean(),
        rationale: z.array(z.string()),
        prerequisites: z.array(z.string()),
        unresolved_items: z.array(z.string()),
      })
      .passthrough(),
    requirements: z
      .object({
        business_goal: z.string(),
        application_scenario: z.string(),
        hard_constraints: z.array(z.string()),
        active_criteria: z.array(criterionSchema.extend({ weight: z.number().min(0).max(1) })),
        inactive_criteria: z.array(criterionSchema),
        scenario_criteria: z.array(
          criterionSchema.extend({
            priority: z.string(),
            priority_label: z.string(),
          }),
        ),
      })
      .passthrough(),
    candidates: z.array(
      z
        .object({
          candidate_id: z.string(),
          name: z.string(),
          summary: z.string(),
        })
        .passthrough(),
    ),
    comparison: z
      .object({
        candidates: z.array(z.object({ candidate_id: z.string(), name: z.string() }).passthrough()),
        criteria: z.array(comparisonCriterionSchema),
      })
      .passthrough(),
    risks: z.array(riskSchema),
    evidence_gaps: z.array(
      z.object({ priority: z.string(), description: z.string() }).passthrough(),
    ),
    evidence: z.array(evidenceSchema),
    audit: z
      .object({
        status: workflowStatusSchema,
        review_verdict: z.enum(['pass', 'evidence_gap', 'reevaluate']).nullable(),
        review_label: z.string(),
        review_reasons: z.array(z.string()),
        evidence_round: z.number().int().nonnegative(),
        review_round: z.number().int().nonnegative(),
        warnings: z.array(z.string()),
      })
      .passthrough(),
  })
  .passthrough()

export type WorkflowStatus = z.infer<typeof workflowStatusSchema>
export type QueryType = z.infer<typeof queryTypeSchema>
export type SessionResponse = z.infer<typeof sessionResponseSchema>
export type WorkflowProgress = z.infer<typeof workflowProgressSchema>
export type CancelSessionResponse = z.infer<typeof cancelSessionResponseSchema>
export type ReportSummary = z.infer<typeof reportSummarySchema>
export type ReportData = z.infer<typeof reportDataSchema>
export type SessionSource = z.infer<typeof sourceSchema>

export function parseSessionResponse(value: unknown): SessionResponse {
  return sessionResponseSchema.parse(value)
}

export function parseCancelSessionResponse(value: unknown): CancelSessionResponse {
  return cancelSessionResponseSchema.parse(value)
}

export function parseReportData(value: unknown): ReportData | null {
  const result = reportDataSchema.safeParse(value)
  return result.success ? result.data : null
}
