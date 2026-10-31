import type { QueryType, ReportData, SessionResponse, WorkflowStatus } from '../api/sessionSchemas'

export type MessageRole = 'user' | 'assistant' | 'system'
// 'streaming' marks the live, never-persisted assistant draft.
export type MessageDelivery = 'pending' | 'sent' | 'cancelled' | 'failed' | 'streaming'

export type ChatMessage = {
  id: string
  role: MessageRole
  content: string
  createdAt: string
  delivery: MessageDelivery
  requestId?: string
}

export type LocalSession = {
  threadId: string
  userId: string
  projectId: string
  title: string
  createdAt: string
  updatedAt: string
  status: WorkflowStatus
  messages: ChatMessage[]
}

export type SessionUiStatus = 'idle' | 'submitting' | 'cancelling' | 'unknown' | WorkflowStatus

export type ActiveSessionView = {
  record: LocalSession
  response: SessionResponse | null
  reportData: ReportData | null
  queryType: QueryType | null
}
