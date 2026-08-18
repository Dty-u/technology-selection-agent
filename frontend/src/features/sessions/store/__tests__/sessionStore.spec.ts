import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useSettingsStore } from '@/features/settings/settingsStore'
import {
  awaitingSelectionResponse,
  completedRagResponse,
  jsonResponse,
} from '@/tests/fixtures/sessionResponses'
import { useSessionStore } from '../sessionStore'

function configureWorkspace() {
  useSettingsStore().save({
    apiBaseUrl: 'http://localhost:8000',
    userId: 'user-a',
    projectId: 'project-a',
  })
}

describe('sessionStore', () => {
  beforeEach(() => {
    localStorage.clear()
    setActivePinia(createPinia())
    vi.restoreAllMocks()
    configureWorkspace()
  })

  it('creates and persists a session only after the server returns its thread id', async () => {
    const fetcher = vi.fn<typeof fetch>().mockResolvedValue(jsonResponse(completedRagResponse, 201))
    vi.stubGlobal('fetch', fetcher)
    const store = useSessionStore()
    store.initialize()

    const threadId = await store.submit('What is FinFET?', false)

    expect(threadId).toBe(completedRagResponse.thread_id)
    expect(store.sessions).toHaveLength(1)
    expect(store.activeSession?.messages.map((message) => message.role)).toEqual([
      'user',
      'assistant',
    ])
    expect(store.activeSession?.messages.every((message) => message.delivery === 'sent')).toBe(true)
    expect(localStorage.length).toBeGreaterThan(1)
  })

  it('sends clarification answers to the same thread and exposes awaiting_user questions', async () => {
    const fetcher = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(jsonResponse(completedRagResponse, 201))
      .mockResolvedValueOnce(jsonResponse(awaitingSelectionResponse))
    vi.stubGlobal('fetch', fetcher)
    const store = useSessionStore()
    store.initialize()
    await store.submit('比较 FinFET 和 GAA', false)

    await store.submit('3nm，功耗优先', true)

    expect(store.uiStatus).toBe('awaiting_user')
    expect(store.activeResponse?.clarification_questions).toEqual([
      '目标制程节点是什么？',
      '功耗和性能哪个优先？',
    ])
    expect(fetcher.mock.calls[1]?.[0]).toContain(
      `/sessions/${completedRagResponse.thread_id}/messages`,
    )
  })

  it('marks a follow-up failed and keeps the session when the API returns 409', async () => {
    const fetcher = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(jsonResponse(completedRagResponse, 201))
      .mockResolvedValueOnce(
        jsonResponse(
          { detail: { code: 'session_busy', message: 'session is already processing' } },
          409,
        ),
      )
    vi.stubGlobal('fetch', fetcher)
    const store = useSessionStore()
    store.initialize()
    await store.submit('What is FinFET?', false)

    await store.submit('继续说明', false)

    expect(store.uiStatus).toBe('error')
    const messages = store.activeSession?.messages ?? []
    expect(messages[messages.length - 1]?.delivery).toBe('failed')
    expect(store.notice?.code).toBe('session_busy')
    expect(store.sessions).toHaveLength(1)
  })

  it('stops only local waiting for an existing session and enters detached state', async () => {
    const fetcher = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(jsonResponse(completedRagResponse, 201))
      .mockImplementationOnce(
        (_input, init) =>
          new Promise<Response>((_resolve, reject) => {
            init?.signal?.addEventListener('abort', () =>
              reject(new DOMException('aborted', 'AbortError')),
            )
          }),
      )
    vi.stubGlobal('fetch', fetcher)
    const store = useSessionStore()
    store.initialize()
    await store.submit('What is FinFET?', false)

    const pending = store.submit('继续说明', false)
    store.stopWaiting()
    await pending

    expect(store.uiStatus).toBe('detached')
    expect(store.notice?.code).toBe('request_aborted')
  })

  it('keeps the local record when server-side close fails', async () => {
    const fetcher = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(jsonResponse(completedRagResponse, 201))
      .mockResolvedValueOnce(jsonResponse({ detail: { code: 'internal_error' } }, 500))
    vi.stubGlobal('fetch', fetcher)
    const store = useSessionStore()
    store.initialize()
    await store.submit('What is FinFET?', false)

    await store.closeActiveSession()

    expect(store.sessions).toHaveLength(1)
    expect(store.notice?.code).toBe('internal_error')
  })

  it('synchronizes the latest server state for a known thread', async () => {
    const fetcher = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(jsonResponse(completedRagResponse, 201))
      .mockResolvedValueOnce(jsonResponse(awaitingSelectionResponse))
    vi.stubGlobal('fetch', fetcher)
    const store = useSessionStore()
    store.initialize()
    await store.submit('比较 FinFET 和 GAA', false)

    await store.syncActiveSession()

    expect(store.uiStatus).toBe('awaiting_user')
    expect(store.activeResponse?.clarification_questions).toHaveLength(2)
    expect(fetcher.mock.calls[1]?.[1]?.method).toBe('GET')
  })

  it('reports API reachability without claiming dependency health', async () => {
    const fetcher = vi.fn<typeof fetch>().mockResolvedValue(jsonResponse({ status: 'ok' }))
    vi.stubGlobal('fetch', fetcher)
    const store = useSessionStore()
    store.initialize()

    await store.checkHealth()

    expect(store.apiReachable).toBe(true)
  })
})
