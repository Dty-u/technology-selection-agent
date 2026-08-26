import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { defineComponent, h, nextTick } from 'vue'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { useSettingsStore } from '@/features/settings/settingsStore'
import MessageBubble from '@/features/sessions/components/MessageBubble.vue'
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

function replyToCreate(response = completedRagResponse) {
  return (_input: RequestInfo | URL, init?: RequestInit) => {
    const threadId = new Headers(init?.headers).get('X-Thread-ID')
    return Promise.resolve(
      jsonResponse(
        {
          ...response,
          thread_id: threadId || response.thread_id,
        },
        201,
      ),
    )
  }
}

describe('sessionStore', () => {
  beforeEach(() => {
    localStorage.clear()
    setActivePinia(createPinia())
    vi.restoreAllMocks()
    configureWorkspace()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('creates and persists a session after the first response completes', async () => {
    const fetcher = vi.fn<typeof fetch>().mockImplementation(replyToCreate())
    vi.stubGlobal('fetch', fetcher)
    const store = useSessionStore()
    store.initialize()

    const threadId = await store.submit('What is FinFET?', false)

    expect(threadId).toBe(store.activeSession?.threadId)
    expect(threadId).toMatch(/^[0-9a-f-]{36}$/i)
    expect(store.sessions).toHaveLength(1)
    expect(store.activeSession?.messages.map((message) => message.role)).toEqual([
      'user',
      'assistant',
    ])
    expect(store.activeSession?.messages.every((message) => message.delivery === 'sent')).toBe(true)
    expect(localStorage.length).toBeGreaterThan(1)
  })

  it('polls real workflow stages while the first session POST waits', async () => {
    vi.useFakeTimers()
    let requestId = ''
    let threadId = ''
    let completePost: ((response: Response) => void) | undefined
    const fetcher = vi.fn<typeof fetch>().mockImplementation((_input, init) => {
      if (init?.method === 'POST') {
        const headers = new Headers(init.headers)
        requestId = headers.get('X-Request-ID') ?? ''
        threadId = headers.get('X-Thread-ID') ?? ''
        return new Promise<Response>((resolve) => {
          completePost = resolve
        })
      }
      return Promise.resolve(
        jsonResponse({
          ...completedRagResponse,
          thread_id: threadId,
          request_id: null,
          status: 'running',
          answer: '',
          sources: [],
          progress: {
            request_id: requestId,
            active: true,
            current_stage: {
              scope: '普通RAG',
              stage_id: 'vector_retrieve',
              label: '向量检索',
            },
            events: [
              {
                scope: '普通RAG',
                stage_id: 'vector_retrieve',
                label: '向量检索',
                status: 'running',
                started_at: '2026-08-23T00:00:00+00:00',
                finished_at: null,
                duration_ms: null,
              },
            ],
          },
        }),
      )
    })
    vi.stubGlobal('fetch', fetcher)
    const store = useSessionStore()
    store.initialize()

    const pending = store.submit('光刻机是什么', false)
    await vi.advanceTimersByTimeAsync(650)
    const observedThreadId = store.activeThreadId
    const observedStage = store.activeResponse?.progress?.current_stage?.label
    const polled = fetcher.mock.calls.some((call) => call[1]?.method === 'GET')

    completePost?.(
      jsonResponse(
        {
          ...completedRagResponse,
          thread_id: threadId || completedRagResponse.thread_id,
          request_id: requestId || completedRagResponse.request_id,
        },
        201,
      ),
    )
    await pending

    expect(threadId).toMatch(/^[0-9a-f-]{36}$/i)
    expect(observedThreadId).toBe(threadId)
    expect(observedStage).toBe('向量检索')
    expect(polled).toBe(true)
  })

  it('removes the pending label from the rendered message after a follow-up POST succeeds', async () => {
    const followUpResponse = {
      ...completedRagResponse,
      request_id: '33333333-3333-4333-8333-333333333333',
      answer: '韩信是历史人物，与半导体领域无关。',
      progress: {
        request_id: '33333333-3333-4333-8333-333333333333',
        active: false,
        current_stage: null,
        events: [],
      },
    }
    let resolveFollowUp: ((response: Response) => void) | undefined
    const fetcher = vi
      .fn<typeof fetch>()
      .mockImplementationOnce(replyToCreate())
      .mockImplementationOnce(
        () =>
          new Promise<Response>((resolve) => {
            resolveFollowUp = resolve
          }),
      )
    vi.stubGlobal('fetch', fetcher)
    const store = useSessionStore()
    store.initialize()
    await store.submit('What is FinFET?', false)

    const Host = defineComponent({
      setup() {
        return () =>
          h(
            'div',
            store.visibleMessages.map((message) => h(MessageBubble, { key: message.id, message })),
          )
      },
    })
    const wrapper = mount(Host)

    const pending = store.submit('韩信是谁', false)
    await nextTick()
    expect(wrapper.text()).toContain('正在发送…')

    resolveFollowUp?.(jsonResponse(followUpResponse))
    await pending
    await nextTick()

    expect(wrapper.text()).toContain('韩信是历史人物')
    expect(wrapper.text()).not.toContain('正在发送…')
  })

  it('sends clarification answers to the same thread and exposes awaiting_user questions', async () => {
    const fetcher = vi
      .fn<typeof fetch>()
      .mockImplementationOnce(replyToCreate())
      .mockResolvedValueOnce(jsonResponse(awaitingSelectionResponse))
    vi.stubGlobal('fetch', fetcher)
    const store = useSessionStore()
    store.initialize()
    await store.submit('比较 FinFET 和 GAA', false)
    const threadId = store.activeThreadId

    await store.submit('3nm，功耗优先', true)

    expect(store.uiStatus).toBe('awaiting_user')
    expect(store.activeResponse?.clarification_questions).toEqual([
      '目标制程节点是什么？',
      '功耗和性能哪个优先？',
    ])
    expect(fetcher.mock.calls[1]?.[0]).toContain(`/sessions/${threadId}/messages`)
  })

  it('marks a follow-up failed and keeps the session when the API returns 409', async () => {
    const fetcher = vi
      .fn<typeof fetch>()
      .mockImplementationOnce(replyToCreate())
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

  it('exits the active request, ignores its aborted POST, and preserves the session', async () => {
    let requestId = ''
    const fetcher = vi.fn<typeof fetch>().mockImplementation((input, init) => {
      const url = String(input)
      if (url.endsWith('/cancel')) {
        return Promise.resolve(
          jsonResponse({
            thread_id: completedRagResponse.thread_id,
            request_id: requestId,
            status: 'cancelling',
          }),
        )
      }
      if (init?.method === 'GET') {
        return Promise.resolve(
          jsonResponse({
            ...completedRagResponse,
            request_id: requestId,
            status: 'cancelled',
            answer: '',
            progress: {
              request_id: requestId,
              active: false,
              cancel_requested: true,
              cancelled: true,
              current_stage: null,
              events: [],
            },
          }),
        )
      }
      if (new Headers(init?.headers).has('X-Thread-ID')) return replyToCreate()(input, init)
      requestId = new Headers(init?.headers).get('X-Request-ID') ?? ''
      return new Promise<Response>((_resolve, reject) => {
        init?.signal?.addEventListener('abort', () =>
          reject(new DOMException('aborted', 'AbortError')),
        )
      })
    })
    vi.stubGlobal('fetch', fetcher)
    const store = useSessionStore()
    store.initialize()
    await store.submit('What is FinFET?', false)

    const pending = store.submit('继续说明', false)
    await expect(store.exitCurrentRequest()).resolves.toBe(true)
    await pending
    await vi.waitFor(() => expect(store.inFlight).toBe(false))

    expect(fetcher.mock.calls.some(([url]) => String(url).endsWith('/cancel'))).toBe(true)
    expect(store.uiStatus).toBe('cancelled')
    const messages = store.activeSession?.messages ?? []
    expect(messages[messages.length - 1]?.delivery).toBe('cancelled')
    expect(store.sessions).toHaveLength(1)
  })

  it('polls the GET session route for real workflow stages while a follow-up POST waits', async () => {
    const createFetcher = vi.fn<typeof fetch>().mockImplementation(replyToCreate())
    vi.stubGlobal('fetch', createFetcher)
    const store = useSessionStore()
    store.initialize()
    await store.submit('What is FinFET?', false)

    vi.useFakeTimers()
    let requestId = ''
    let completePost: ((response: Response) => void) | undefined
    const fetcher = vi.fn<typeof fetch>().mockImplementation((input, init) => {
      if (init?.method === 'POST') {
        requestId = (init.headers as Record<string, string>)['X-Request-ID'] ?? ''
        return new Promise<Response>((resolve) => {
          completePost = resolve
        })
      }
      return Promise.resolve(
        jsonResponse({
          ...completedRagResponse,
          status: 'running',
          progress: {
            request_id: requestId,
            active: true,
            current_stage: {
              scope: '普通RAG',
              stage_id: 'vector_retrieve',
              label: '向量检索',
            },
            events: [
              {
                scope: '普通RAG',
                stage_id: 'vector_retrieve',
                label: '向量检索',
                status: 'running',
                started_at: '2026-08-23T00:00:00+00:00',
                finished_at: null,
                duration_ms: null,
              },
            ],
          },
        }),
      )
    })
    vi.stubGlobal('fetch', fetcher)

    const pending = store.submit('继续说明', false)
    await vi.advanceTimersByTimeAsync(650)

    expect(store.uiStatus).toBe('running')
    expect(store.activeResponse?.progress?.current_stage?.label).toBe('向量检索')
    expect(fetcher.mock.calls.some((call) => call[1]?.method === 'GET')).toBe(true)

    completePost?.(
      jsonResponse({
        ...completedRagResponse,
        request_id: requestId,
        progress: {
          request_id: requestId,
          active: false,
          current_stage: null,
          events: [],
        },
      }),
    )
    await pending
    expect(store.uiStatus).toBe('completed')
  })

  it('keeps an executing response scoped to its session while the user visits another session', async () => {
    const runningThreadId = 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa'
    const otherThreadId = 'bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb'
    const runningCompleted = { ...completedRagResponse, thread_id: runningThreadId }
    const otherCompleted = {
      ...completedRagResponse,
      thread_id: otherThreadId,
      answer: '另一个会话的回答',
    }
    let requestId = ''
    let progressPoll = 0
    let completePost: ((response: Response) => void) | undefined
    const fetcher = vi.fn<typeof fetch>().mockImplementation((input, init) => {
      if (init?.method === 'POST') {
        requestId = (init.headers as Record<string, string>)['X-Request-ID'] ?? ''
        return new Promise<Response>((resolve) => {
          completePost = resolve
        })
      }
      if (String(input).includes(otherThreadId))
        return Promise.resolve(jsonResponse(otherCompleted))
      progressPoll += 1
      if (progressPoll === 1) return Promise.resolve(jsonResponse(runningCompleted))
      const stage = progressPoll === 2 ? '检索证据' : '分析风险'
      return Promise.resolve(
        jsonResponse({
          ...runningCompleted,
          status: 'running',
          progress: {
            request_id: requestId,
            active: true,
            current_stage: {
              scope: '技术选型',
              stage_id: progressPoll === 2 ? 'retrieve_evidence' : 'analyze_risks',
              label: stage,
            },
            events: [],
          },
        }),
      )
    })
    vi.stubGlobal('fetch', fetcher)
    const store = useSessionStore()
    store.initialize()
    store.sessions.push(
      {
        threadId: runningThreadId,
        userId: 'user-a',
        projectId: 'project-a',
        title: '运行中的会话',
        createdAt: '2026-08-23T00:00:00.000Z',
        updatedAt: '2026-08-23T00:00:00.000Z',
        status: 'completed',
        messages: [],
      },
      {
        threadId: otherThreadId,
        userId: 'user-a',
        projectId: 'project-a',
        title: '另一个会话',
        createdAt: '2026-08-23T00:00:00.000Z',
        updatedAt: '2026-08-23T00:00:00.000Z',
        status: 'completed',
        messages: [],
      },
    )
    store.selectSession(runningThreadId)
    await store.syncActiveSession()

    vi.useFakeTimers()
    const pending = store.submit('继续执行技术选型', false)
    await vi.advanceTimersByTimeAsync(650)
    expect(store.activeResponse?.progress?.current_stage?.label).toBe('检索证据')

    store.selectSession(otherThreadId)
    await store.syncActiveSession()
    expect(store.activeResponse?.thread_id).toBe(otherThreadId)
    await vi.advanceTimersByTimeAsync(650)

    store.selectSession(runningThreadId)
    expect(store.activeResponse?.thread_id).toBe(runningThreadId)
    expect(store.activeResponse?.progress?.current_stage?.label).toBe('分析风险')

    store.selectSession(otherThreadId)
    completePost?.(
      jsonResponse({
        ...runningCompleted,
        request_id: requestId,
        progress: {
          request_id: requestId,
          active: false,
          current_stage: null,
          events: [],
        },
      }),
    )
    await pending

    expect(store.activeResponse?.thread_id).toBe(otherThreadId)
    store.selectSession(runningThreadId)
    expect(store.uiStatus).toBe('completed')
  })

  it('keeps the local record when server-side close fails', async () => {
    const fetcher = vi
      .fn<typeof fetch>()
      .mockImplementationOnce(replyToCreate())
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
      .mockImplementationOnce(replyToCreate())
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
