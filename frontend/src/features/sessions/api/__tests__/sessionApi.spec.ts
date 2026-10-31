import { describe, expect, it, vi } from 'vitest'

import { ApiError } from '@/shared/api/errors'
import { controlledEventStream, sseFrame } from '@/tests/fixtures/sessionResponses'
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

  it('streams deltas split across chunks and resolves with the final event', async () => {
    const stream = controlledEventStream()
    const fetcher = vi.fn<typeof fetch>().mockResolvedValue(stream.response)
    const api = createSessionApi(
      { apiBaseUrl: 'http://localhost:8000', userId: 'user-a', projectId: 'project-a' },
      fetcher,
    )
    const deltas: string[] = []
    const onReset = vi.fn<() => void>()

    const pending = api.sendMessageStream(
      responseBody.thread_id,
      { message: '继续', report_format: 'markdown' },
      '33333333-3333-4333-8333-333333333333',
      { onReset, onDelta: (text) => deltas.push(text) },
    )
    const frames = [
      ': keepalive\n\n',
      sseFrame('reset', {}),
      sseFrame('delta', { text: '鳍式' }).replace(/\n/g, '\r\n'),
      sseFrame('delta', { text: '沟道' }),
      sseFrame('final', responseBody),
    ].join('')
    const middle = Math.floor(frames.length / 2)
    stream.push(frames.slice(0, middle))
    stream.push(frames.slice(middle))
    stream.close()

    await expect(pending).resolves.toMatchObject({ answer: responseBody.answer })
    expect(onReset).toHaveBeenCalledTimes(1)
    expect(deltas.join('')).toBe('鳍式沟道')
    const [url, init] = fetcher.mock.calls[0] ?? []
    expect(url).toBe(
      `http://localhost:8000/api/v1/projects/project-a/sessions/${responseBody.thread_id}/messages/stream`,
    )
    expect(new Headers(init?.headers).get('Accept')).toBe('text/event-stream')
  })

  it('turns an in-stream error event into a sanitized ApiError', async () => {
    const stream = controlledEventStream()
    const fetcher = vi.fn<typeof fetch>().mockResolvedValue(stream.response)
    const api = createSessionApi(
      { apiBaseUrl: 'http://localhost:8000', userId: 'user-a', projectId: 'project-a' },
      fetcher,
    )

    const pending = api.createSessionStream(
      { message: 'What is FinFET?', report_format: 'markdown' },
      '11111111-1111-4111-8111-111111111111',
      responseBody.thread_id,
      {},
    )
    stream.push(sseFrame('delta', { text: '部分' }))
    stream.push(sseFrame('error', { code: 'service_unavailable', message: 'mysql down' }))
    stream.close()

    const error = await pending.catch((reason: unknown) => reason)
    expect(error).toBeInstanceOf(ApiError)
    expect(error).toMatchObject({ status: 503, code: 'service_unavailable' })
    expect((error as ApiError).message).not.toContain('mysql')
  })

  it('accepts a plain JSON reply and rejects a stream that ends without a final event', async () => {
    const api = createSessionApi(
      { apiBaseUrl: 'http://localhost:8000', userId: 'user-a', projectId: 'project-a' },
      vi.fn<typeof fetch>().mockResolvedValue(jsonResponse(responseBody, 201)),
    )
    await expect(
      api.createSessionStream(
        { message: 'What is FinFET?', report_format: 'markdown' },
        '11111111-1111-4111-8111-111111111111',
        responseBody.thread_id,
        {},
      ),
    ).resolves.toMatchObject({ answer: responseBody.answer })

    const stream = controlledEventStream()
    const truncated = createSessionApi(
      { apiBaseUrl: 'http://localhost:8000', userId: 'user-a', projectId: 'project-a' },
      vi.fn<typeof fetch>().mockResolvedValue(stream.response),
    ).createSessionStream(
      { message: 'What is FinFET?', report_format: 'markdown' },
      '11111111-1111-4111-8111-111111111111',
      responseBody.thread_id,
      {},
    )
    stream.push(sseFrame('delta', { text: '半' }))
    stream.close()
    await expect(truncated).rejects.toMatchObject({ code: 'invalid_response' })
  })
})
