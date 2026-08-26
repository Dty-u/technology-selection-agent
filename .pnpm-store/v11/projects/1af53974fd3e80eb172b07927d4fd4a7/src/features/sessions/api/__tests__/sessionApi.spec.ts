import { describe, expect, it, vi } from 'vitest'

import { ApiError } from '@/shared/api/errors'
import { createSessionApi } from '../sessionApi'

const responseBody = {
  request_id: '11111111-1111-4111-8111-111111111111',
  project_id: 'project / a',
  thread_id: '22222222-2222-4222-8222-222222222222',
  status: 'completed',
  query_type: 'rag_qa',
  response_type: 'answer',
  answer: 'FinFET 使用鳍式沟道。',
  report: '',
  report_data: {},
  clarification_questions: [],
  clarification_answers: [],
  clarification_history: [],
  sources: [],
  warnings: [],
  report_url: null,
}

function jsonResponse(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

describe('sessionApi', () => {
  it('checks only API liveness without sending workspace identity', async () => {
    const fetcher = vi.fn<typeof fetch>().mockResolvedValue(jsonResponse({ status: 'ok' }))
    const api = createSessionApi(
      { apiBaseUrl: 'http://localhost:8000', userId: 'user-a', projectId: 'project-a' },
      fetcher,
    )

    await expect(api.health()).resolves.toEqual({ status: 'ok' })
    expect(fetcher).toHaveBeenCalledWith('http://localhost:8000/health/live', {
      method: 'GET',
      headers: { Accept: 'application/json' },
      signal: undefined,
    })
  })

  it('creates a session with encoded scope, identity, request id, and exact message payload', async () => {
    const fetcher = vi.fn<typeof fetch>().mockResolvedValue(jsonResponse(responseBody, 201))
    const api = createSessionApi(
      {
        apiBaseUrl: 'http://localhost:8000/',
        userId: 'user-a',
        projectId: 'project / a',
      },
      fetcher,
    )

    const result = await api.createSession(
      { message: 'What is FinFET?', report_format: 'markdown' },
      '11111111-1111-4111-8111-111111111111',
      '22222222-2222-4222-8222-222222222222',
    )

    expect(result.answer).toBe('FinFET 使用鳍式沟道。')
    const [url, init] = fetcher.mock.calls[0]!
    expect(url).toBe('http://localhost:8000/api/v1/projects/project%20%2F%20a/sessions')
    expect(init?.method).toBe('POST')
    expect(new Headers(init?.headers).get('X-User-ID')).toBe('user-a')
    expect(new Headers(init?.headers).get('X-Request-ID')).toBe(
      '11111111-1111-4111-8111-111111111111',
    )
    expect(new Headers(init?.headers).get('X-Thread-ID')).toBe(
      '22222222-2222-4222-8222-222222222222',
    )
    expect(init?.body).toBe(
      JSON.stringify({ message: 'What is FinFET?', report_format: 'markdown' }),
    )
  })

  it('maps sanitized FastAPI errors without exposing private response text', async () => {
    const fetcher = vi.fn<typeof fetch>().mockResolvedValue(
      jsonResponse(
        {
          detail: {
            code: 'service_unavailable',
            message: 'a required service is unavailable',
          },
        },
        503,
      ),
    )
    const api = createSessionApi(
      { apiBaseUrl: 'http://localhost:8000', userId: 'user-a', projectId: 'project-a' },
      fetcher,
    )

    await expect(api.getSession('22222222-2222-4222-8222-222222222222')).rejects.toMatchObject({
      name: 'ApiError',
      status: 503,
      code: 'service_unavailable',
    } satisfies Partial<ApiError>)
  })

  it('sends a follow-up to the existing thread and can close that same scoped session', async () => {
    const fetcher = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(jsonResponse(responseBody))
      .mockResolvedValueOnce(new Response(null, { status: 204 }))
    const api = createSessionApi(
      { apiBaseUrl: 'http://localhost:8000', userId: 'user-a', projectId: 'project-a' },
      fetcher,
    )
    const threadId = '22222222-2222-4222-8222-222222222222'

    await api.sendMessage(
      threadId,
      { message: '继续比较成本', report_format: 'docx' },
      '33333333-3333-4333-8333-333333333333',
    )
    await api.closeSession(threadId)

    expect(fetcher.mock.calls[0]?.[0]).toBe(
      `http://localhost:8000/api/v1/projects/project-a/sessions/${threadId}/messages`,
    )
    expect(fetcher.mock.calls[0]?.[1]?.body).toBe(
      JSON.stringify({ message: '继续比较成本', report_format: 'docx' }),
    )
    expect(fetcher.mock.calls[1]).toEqual([
      `http://localhost:8000/api/v1/projects/project-a/sessions/${threadId}`,
      {
        method: 'DELETE',
        headers: { Accept: 'application/json', 'X-User-ID': 'user-a' },
        signal: undefined,
      },
    ])
  })

  it('requests scoped cooperative cancellation without deleting the session', async () => {
    const fetcher = vi.fn<typeof fetch>().mockResolvedValue(
      jsonResponse({
        thread_id: '22222222-2222-4222-8222-222222222222',
        request_id: '33333333-3333-4333-8333-333333333333',
        status: 'cancelling',
      }),
    )
    const api = createSessionApi(
      { apiBaseUrl: 'http://localhost:8000', userId: 'user-a', projectId: 'project-a' },
      fetcher,
    )

    await expect(api.cancelSession('22222222-2222-4222-8222-222222222222')).resolves.toMatchObject({
      status: 'cancelling',
    })
    expect(fetcher).toHaveBeenCalledWith(
      'http://localhost:8000/api/v1/projects/project-a/sessions/22222222-2222-4222-8222-222222222222/cancel',
      {
        method: 'POST',
        headers: { Accept: 'application/json', 'X-User-ID': 'user-a' },
        signal: undefined,
      },
    )
  })

  it('downloads a report through authenticated fetch instead of a plain link', async () => {
    const fetcher = vi.fn<typeof fetch>().mockResolvedValue(
      new Response('docx', {
        status: 200,
        headers: {
          'Content-Disposition': 'attachment; filename="selection.docx"',
          'Content-Type': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        },
      }),
    )
    const api = createSessionApi(
      { apiBaseUrl: 'http://localhost:8000', userId: 'user-a', projectId: 'project-a' },
      fetcher,
    )

    const downloaded = await api.downloadReport(
      '/api/v1/projects/project-a/sessions/22222222-2222-4222-8222-222222222222/report',
    )

    expect(downloaded.filename).toBe('selection.docx')
    expect(await downloaded.blob.text()).toBe('docx')
    const [url, init] = fetcher.mock.calls[0]!
    expect(url).toBe(
      'http://localhost:8000/api/v1/projects/project-a/sessions/22222222-2222-4222-8222-222222222222/report',
    )
    expect(new Headers(init?.headers).get('X-User-ID')).toBe('user-a')
  })

  it('uses the visible report title when the browser cannot read Content-Disposition', async () => {
    const fetcher = vi.fn<typeof fetch>().mockResolvedValue(new Response('docx', { status: 200 }))
    const api = createSessionApi(
      { apiBaseUrl: 'http://localhost:8000', userId: 'user-a', projectId: 'project-a' },
      fetcher,
    )

    const downloaded = await api.downloadReport(
      '/api/v1/projects/project-a/sessions/22222222-2222-4222-8222-222222222222/reports/report-1',
      '移动 SoC 技术选型.docx',
    )

    expect(downloaded.filename).toBe('移动 SoC 技术选型.docx')
  })
})
