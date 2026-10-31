import { z } from 'zod'

import { ApiError } from '@/shared/api/errors'
import {
  requestBlob,
  requestEventStream,
  requestJson,
  requestVoid,
  type ServerSentEvent,
} from '@/shared/api/http'
import {
  parseCancelSessionResponse,
  parseImportSessionResponse,
  parseMessageList,
  parseProject,
  parseProjectList,
  parseReportContent,
  parseReportList,
  parseSessionList,
  parseSessionResponse,
  type CancelSessionResponse,
  type MessageList,
  type ProjectSummary,
  type ReportContent,
  type ReportLibraryItem,
  type SessionList,
  type SessionResponse,
} from './sessionSchemas'

export type ApiSettings = {
  apiBaseUrl: string
  userId: string
  projectId: string
}

export type MessageRequest = {
  message: string
  report_format: 'markdown' | 'docx'
}

export type ImportSessionRequest = {
  thread_id: string
  title: string
  status: string
  created_at: string
  updated_at: string
  messages: {
    id: string
    role: string
    content: string
    delivery: string
    created_at: string
    request_id?: string
  }[]
}

export type AnswerStreamHandlers = {
  onReset?: () => void
  onDelta?: (text: string) => void
}

const healthSchema = z.object({ status: z.literal('ok') }).passthrough()
const deltaSchema = z.object({ text: z.string() })

function dispatchAnswerEvent(handlers: AnswerStreamHandlers) {
  return (event: ServerSentEvent) => {
    if (event.event === 'reset') {
      handlers.onReset?.()
      return
    }
    if (event.event !== 'delta') return
    try {
      const delta = deltaSchema.parse(JSON.parse(event.data))
      if (delta.text) handlers.onDelta?.(delta.text)
    } catch {
      // A malformed partial chunk is cosmetic; the final event is authoritative.
    }
  }
}

function normalizeBaseUrl(value: string): string {
  return value.replace(/\/+$/, '')
}

export function createSessionApi(settings: ApiSettings, fetcher: typeof fetch = fetch) {
  const baseUrl = normalizeBaseUrl(settings.apiBaseUrl)
  const projectPath = `/api/v1/projects/${encodeURIComponent(settings.projectId)}`
  const sessionHeaders = (requestId?: string): Record<string, string> => ({
    Accept: 'application/json',
    'X-User-ID': settings.userId,
    ...(requestId ? { 'X-Request-ID': requestId } : {}),
  })
  const sessionUrl = (threadId: string) =>
    `${baseUrl}${projectPath}/sessions/${encodeURIComponent(threadId)}`
  // The report library spans projects, so these name their project explicitly.
  const reportUrl = (projectId: string, threadId: string, reportId: string) =>
    `${baseUrl}/api/v1/projects/${encodeURIComponent(projectId)}/sessions/` +
    `${encodeURIComponent(threadId)}/reports/${encodeURIComponent(reportId)}`

  return {
    health(signal?: AbortSignal) {
      return requestJson(
        fetcher,
        `${baseUrl}/health/live`,
        { method: 'GET', headers: { Accept: 'application/json' }, signal },
        (value) => healthSchema.parse(value),
      )
    },

    createSession(
      message: MessageRequest,
      requestId: string,
      threadId: string,
      signal?: AbortSignal,
    ): Promise<SessionResponse> {
      return requestJson(
        fetcher,
        `${baseUrl}${projectPath}/sessions`,
        {
          method: 'POST',
          headers: {
            ...sessionHeaders(requestId),
            'X-Thread-ID': threadId,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(message),
          signal,
        },
        parseSessionResponse,
      )
    },

    sendMessage(
      threadId: string,
      message: MessageRequest,
      requestId: string,
      signal?: AbortSignal,
    ): Promise<SessionResponse> {
      return requestJson(
        fetcher,
        `${sessionUrl(threadId)}/messages`,
        {
          method: 'POST',
          headers: {
            ...sessionHeaders(requestId),
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(message),
          signal,
        },
        parseSessionResponse,
      )
    },

    createSessionStream(
      message: MessageRequest,
      requestId: string,
      threadId: string,
      handlers: AnswerStreamHandlers,
      signal?: AbortSignal,
    ): Promise<SessionResponse> {
      return requestEventStream(
        fetcher,
        `${baseUrl}${projectPath}/sessions/stream`,
        {
          method: 'POST',
          headers: {
            ...sessionHeaders(requestId),
            Accept: 'text/event-stream',
            'X-Thread-ID': threadId,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(message),
          signal,
        },
        parseSessionResponse,
        dispatchAnswerEvent(handlers),
      )
    },

    sendMessageStream(
      threadId: string,
      message: MessageRequest,
      requestId: string,
      handlers: AnswerStreamHandlers,
      signal?: AbortSignal,
    ): Promise<SessionResponse> {
      return requestEventStream(
        fetcher,
        `${sessionUrl(threadId)}/messages/stream`,
        {
          method: 'POST',
          headers: {
            ...sessionHeaders(requestId),
            Accept: 'text/event-stream',
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(message),
          signal,
        },
        parseSessionResponse,
        dispatchAnswerEvent(handlers),
      )
    },

    listProjects(signal?: AbortSignal): Promise<ProjectSummary[]> {
      return requestJson(
        fetcher,
        `${baseUrl}/api/v1/projects`,
        { method: 'GET', headers: sessionHeaders(), signal },
        parseProjectList,
      )
    },

    ensureProject(signal?: AbortSignal): Promise<ProjectSummary> {
      return requestJson(
        fetcher,
        `${baseUrl}${projectPath}`,
        {
          method: 'PUT',
          headers: { ...sessionHeaders(), 'Content-Type': 'application/json' },
          body: '{}',
          signal,
        },
        parseProject,
      )
    },

    listSessions(signal?: AbortSignal): Promise<SessionList> {
      return requestJson(
        fetcher,
        `${baseUrl}${projectPath}/sessions`,
        { method: 'GET', headers: sessionHeaders(), signal },
        parseSessionList,
      )
    },

    listMessages(threadId: string, signal?: AbortSignal): Promise<MessageList> {
      return requestJson(
        fetcher,
        `${sessionUrl(threadId)}/messages`,
        { method: 'GET', headers: sessionHeaders(), signal },
        parseMessageList,
      )
    },

    importSession(session: ImportSessionRequest, signal?: AbortSignal) {
      return requestJson(
        fetcher,
        `${baseUrl}${projectPath}/sessions/import`,
        {
          method: 'POST',
          headers: { ...sessionHeaders(), 'Content-Type': 'application/json' },
          body: JSON.stringify(session),
          signal,
        },
        parseImportSessionResponse,
      )
    },

    getSession(threadId: string, signal?: AbortSignal): Promise<SessionResponse> {
      return requestJson(
        fetcher,
        sessionUrl(threadId),
        { method: 'GET', headers: sessionHeaders(), signal },
        parseSessionResponse,
      )
    },

    cancelSession(threadId: string, signal?: AbortSignal): Promise<CancelSessionResponse> {
      return requestJson(
        fetcher,
        `${sessionUrl(threadId)}/cancel`,
        { method: 'POST', headers: sessionHeaders(), signal },
        parseCancelSessionResponse,
      )
    },

    closeSession(threadId: string, signal?: AbortSignal): Promise<void> {
      return requestVoid(fetcher, sessionUrl(threadId), {
        method: 'DELETE',
        headers: sessionHeaders(),
        signal,
      })
    },

    listReports(signal?: AbortSignal): Promise<ReportLibraryItem[]> {
      return requestJson(
        fetcher,
        `${baseUrl}/api/v1/reports`,
        { method: 'GET', headers: sessionHeaders(), signal },
        parseReportList,
      )
    },

    getReportContent(
      projectId: string,
      threadId: string,
      reportId: string,
      signal?: AbortSignal,
    ): Promise<ReportContent> {
      return requestJson(
        fetcher,
        `${reportUrl(projectId, threadId, reportId)}/content`,
        { method: 'GET', headers: sessionHeaders(), signal },
        parseReportContent,
      )
    },

    deleteReport(
      projectId: string,
      threadId: string,
      reportId: string,
      signal?: AbortSignal,
    ): Promise<void> {
      return requestVoid(fetcher, reportUrl(projectId, threadId, reportId), {
        method: 'DELETE',
        headers: sessionHeaders(),
        signal,
      })
    },

    async downloadReport(reportUrl: string, fallbackFilename?: string, signal?: AbortSignal) {
      const target = new URL(reportUrl, `${baseUrl}/`)
      const apiOrigin = new URL(baseUrl).origin
      if (target.origin !== apiOrigin) {
        throw new ApiError(0, 'unsafe_report_url', '报告地址不属于当前 Agent API。')
      }
      const downloaded = await requestBlob(fetcher, target.toString(), {
        method: 'GET',
        headers: sessionHeaders(),
        signal,
      })
      return {
        ...downloaded,
        filename:
          downloaded.filename === 'technology-selection-report.docx' && fallbackFilename
            ? fallbackFilename
            : downloaded.filename,
      }
    },
  }
}
