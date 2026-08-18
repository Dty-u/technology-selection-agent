import { z } from 'zod'

import { ApiError } from '@/shared/api/errors'
import { requestBlob, requestJson, requestVoid } from '@/shared/api/http'
import { parseSessionResponse, type SessionResponse } from './sessionSchemas'

export type ApiSettings = {
  apiBaseUrl: string
  userId: string
  projectId: string
}

export type MessageRequest = {
  message: string
  report_format: 'markdown' | 'docx'
}

const healthSchema = z.object({ status: z.literal('ok') }).passthrough()

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
      signal?: AbortSignal,
    ): Promise<SessionResponse> {
      return requestJson(
        fetcher,
        `${baseUrl}${projectPath}/sessions`,
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

    getSession(threadId: string, signal?: AbortSignal): Promise<SessionResponse> {
      return requestJson(
        fetcher,
        sessionUrl(threadId),
        { method: 'GET', headers: sessionHeaders(), signal },
        parseSessionResponse,
      )
    },

    closeSession(threadId: string, signal?: AbortSignal): Promise<void> {
      return requestVoid(fetcher, sessionUrl(threadId), {
        method: 'DELETE',
        headers: sessionHeaders(),
        signal,
      })
    },

    downloadReport(reportUrl: string, signal?: AbortSignal) {
      const target = new URL(reportUrl, `${baseUrl}/`)
      const apiOrigin = new URL(baseUrl).origin
      if (target.origin !== apiOrigin) {
        throw new ApiError(0, 'unsafe_report_url', '报告地址不属于当前 Agent API。')
      }
      return requestBlob(fetcher, target.toString(), {
        method: 'GET',
        headers: sessionHeaders(),
        signal,
      })
    },
  }
}
