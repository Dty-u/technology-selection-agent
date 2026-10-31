import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { useSettingsStore } from '@/features/settings/settingsStore'
import { jsonResponse } from '@/tests/fixtures/sessionResponses'
import { sessionStorageKey } from '../../storage/sessionRepository'
import { useSessionStore } from '../sessionStore'

const THREAD_A = '11111111-1111-4111-8111-111111111111'
const THREAD_B = '22222222-2222-4222-8222-222222222222'
const REQUEST_ID = '33333333-3333-4333-8333-333333333333'

function summary(threadId: string, title: string) {
  return {
    thread_id: threadId,
    project_id: 'project-a',
    title,
    status: 'completed',
    created_at: '2026-01-01T00:00:00.000Z',
    updated_at: '2026-01-02T00:00:00.000Z',
  }
}

function storedMessage(id: string, role: string, content: string, kind = 'message') {
  return {
    id,
    role,
    content,
    delivery: 'sent',
    kind,
    request_id: REQUEST_ID,
    created_at: '2026-01-01T00:00:00.000Z',
  }
}

function localRecord(threadId: string, delivery = 'sent') {
  return {
    threadId,
    userId: 'user-a',
    projectId: 'project-a',
    title: '本机会话',
    createdAt: '2025-01-01T00:00:00.000Z',
    updatedAt: '2025-01-01T00:00:00.000Z',
    status: 'completed',
    messages: [
      {
        id: 'local-1',
        role: 'user',
        content: '本机问题',
        createdAt: '2025-01-01T00:00:00.000Z',
        delivery,
      },
    ],
  }
}

function seedLocal(...records: ReturnType<typeof localRecord>[]) {
  localStorage.setItem(
    sessionStorageKey('user-a', 'project-a'),
    JSON.stringify({ version: 1, sessions: records }),
  )
}

type Route = { status?: number; body: unknown }

function stubBackend(routes: Record<string, Route | (() => Route)>) {
  const fetcher = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const path = new URL(String(input)).pathname.replace('/api/v1/projects/project-a', '')
    const key = `${init?.method ?? 'GET'} ${path}`
    const route = routes[key]
    if (!route) throw new TypeError(`network down: ${key}`)
    const { status = 200, body } = typeof route === 'function' ? route() : route
    return jsonResponse(body, status)
  })
  vi.stubGlobal('fetch', fetcher)
  return fetcher
}

const PROJECT: Route = {
  body: {
    project_id: 'project-a',
    name: 'project-a',
    created_at: '2026-01-01T00:00:00.000Z',
    updated_at: '2026-01-01T00:00:00.000Z',
  },
}
const IMPORTED: Route = { body: { thread_id: THREAD_A, imported: true } }

function setup() {
  useSettingsStore().save({
    apiBaseUrl: 'http://localhost:8000',
    userId: 'user-a',
    projectId: 'project-a',
  })
  const store = useSessionStore()
  store.initialize()
  return store
}

function requests(fetcher: ReturnType<typeof stubBackend>, method: string, suffix: string) {
  return fetcher.mock.calls.filter(
    ([url, init]) => init?.method === method && String(url).endsWith(suffix),
  )
}

describe('session history from the backend', () => {
  beforeEach(() => {
    localStorage.clear()
    setActivePinia(createPinia())
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('lists stored sessions for the current user and project', async () => {
    const fetcher = stubBackend({
      'PUT ': PROJECT,
      'GET /sessions': { body: { sessions: [summary(THREAD_A, '后端会话')] } },
    })
    const store = setup()

    expect(await store.syncFromBackend()).toBe(true)

    expect(store.sessions.map((session) => session.title)).toEqual(['后端会话'])
    const [, init] = fetcher.mock.calls[0] as [unknown, RequestInit]
    expect((init.headers as Record<string, string>)['X-User-ID']).toBe('user-a')
  })

  it('uploads browser sessions only once', async () => {
    seedLocal(localRecord(THREAD_A))
    const fetcher = stubBackend({
      'PUT ': PROJECT,
      'POST /sessions/import': IMPORTED,
      'GET /sessions': { body: { sessions: [summary(THREAD_A, '本机会话')] } },
    })
    const store = setup()

    await store.syncFromBackend()
    await store.syncFromBackend()

    expect(requests(fetcher, 'POST', '/sessions/import')).toHaveLength(1)
  })

  it('does not let a rejected session block the upload forever', async () => {
    seedLocal(localRecord(THREAD_A))
    const fetcher = stubBackend({
      'PUT ': PROJECT,
      'POST /sessions/import': {
        status: 404,
        body: { detail: { code: 'session_not_found', message: 'session not found' } },
      },
      'GET /sessions': { body: { sessions: [summary(THREAD_B, '后端会话')] } },
    })
    const store = setup()

    expect(await store.syncFromBackend()).toBe(true)
    await store.syncFromBackend()

    expect(requests(fetcher, 'POST', '/sessions/import')).toHaveLength(1)
    expect(store.sessions.map((session) => session.threadId)).toEqual([THREAD_B])
  })

  it('retries the upload after an outage and keeps the local cache', async () => {
    seedLocal(localRecord(THREAD_A))
    stubBackend({ 'PUT ': PROJECT })
    const store = setup()

    expect(await store.syncFromBackend()).toBe(false)

    expect(store.sessions.map((session) => session.threadId)).toEqual([THREAD_A])
    expect(store.storageWarning).toContain('本机缓存')

    const fetcher = stubBackend({
      'PUT ': PROJECT,
      'POST /sessions/import': IMPORTED,
      'GET /sessions': { body: { sessions: [summary(THREAD_A, '本机会话')] } },
    })
    expect(await store.syncFromBackend()).toBe(true)
    expect(requests(fetcher, 'POST', '/sessions/import')).toHaveLength(1)
    expect(store.storageWarning).toBeNull()
  })

  it('removes sessions deleted elsewhere but keeps undelivered ones', async () => {
    seedLocal(localRecord(THREAD_A), localRecord(THREAD_B, 'failed'))
    stubBackend({
      'PUT ': PROJECT,
      'POST /sessions/import': IMPORTED,
      'GET /sessions': { body: { sessions: [] } },
    })
    const store = setup()

    await store.syncFromBackend()

    expect(store.sessions.map((session) => session.threadId)).toEqual([THREAD_B])
  })

  it('loads messages on demand without clarification bubbles', async () => {
    stubBackend({
      'PUT ': PROJECT,
      'GET /sessions': { body: { sessions: [summary(THREAD_A, '后端会话')] } },
      [`GET /sessions/${THREAD_A}/messages`]: {
        body: {
          messages: [
            storedMessage('1', 'user', '问题'),
            storedMessage('2', 'assistant', '预算是多少？', 'clarification'),
            storedMessage('3', 'assistant', '回答'),
          ],
        },
      },
    })
    const store = setup()
    await store.syncFromBackend()

    store.selectSession(THREAD_A)
    await store.loadActiveMessages()

    expect(store.visibleMessages.map((message) => message.content)).toEqual(['问题', '回答'])
  })

  it('keeps a failed local message when loading stored messages', async () => {
    seedLocal(localRecord(THREAD_A, 'failed'))
    stubBackend({
      'PUT ': PROJECT,
      'POST /sessions/import': IMPORTED,
      'GET /sessions': { body: { sessions: [summary(THREAD_A, '本机会话')] } },
      [`GET /sessions/${THREAD_A}/messages`]: {
        body: { messages: [storedMessage('1', 'user', '已送达')] },
      },
    })
    const store = setup()
    await store.syncFromBackend()

    store.selectSession(THREAD_A)
    await store.loadActiveMessages()

    expect(store.visibleMessages.map((message) => message.content)).toEqual(['已送达', '本机问题'])
  })
})
