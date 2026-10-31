import { computed, ref } from 'vue'
import { defineStore } from 'pinia'

import { useSettingsStore } from '@/features/settings/settingsStore'
import { ApiError, apiErrorFromUnknown } from '@/shared/api/errors'
import { createSessionApi } from '../api/sessionApi'
import { parseReportData, type ReportData, type SessionResponse } from '../api/sessionSchemas'
import type { ChatMessage, LocalSession, SessionUiStatus } from '../model/sessionTypes'
import {
  isSessionImportDone,
  loadSessionRecords,
  markSessionImportDone,
  saveSessionRecords,
} from '../storage/sessionRepository'

function newId(): string {
  if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID()
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (character) => {
    const random = Math.floor(Math.random() * 16)
    const value = character === 'x' ? random : (random & 0x3) | 0x8
    return value.toString(16)
  })
}

function now(): string {
  return new Date().toISOString()
}

function titleFor(message: string): string {
  const normalized = message.replace(/\s+/g, ' ').trim()
  return normalized.length > 32 ? `${normalized.slice(0, 32)}…` : normalized
}

function userMessage(content: string, requestId: string): ChatMessage {
  return {
    id: newId(),
    role: 'user',
    content,
    createdAt: now(),
    delivery: 'pending',
    requestId,
  }
}

function assistantContent(response: SessionResponse): string {
  return response.answer.trim() || response.report.trim()
}

function initialRunningResponse(
  projectId: string,
  threadId: string,
  requestId: string,
): SessionResponse {
  return {
    request_id: requestId,
    project_id: projectId,
    thread_id: threadId,
    status: 'running',
    query_type: null,
    response_type: null,
    answer: '',
    report: '',
    report_data: {},
    clarification_questions: [],
    clarification_answers: [],
    clarification_history: [],
    sources: [],
    warnings: [],
    report_url: null,
    reports: [],
    progress: {
      request_id: requestId,
      active: true,
      cancel_requested: false,
      cancelled: false,
      current_stage: null,
      events: [],
    },
  }
}

function scheduleFrame(callback: () => void): () => void {
  if (typeof globalThis.requestAnimationFrame === 'function') {
    const handle = globalThis.requestAnimationFrame(callback)
    return () => globalThis.cancelAnimationFrame(handle)
  }
  const handle = globalThis.setTimeout(callback, 16)
  return () => globalThis.clearTimeout(handle)
}

function waitForPoll(signal: AbortSignal, delayMs = 600): Promise<void> {
  return new Promise((resolve) => {
    if (signal.aborted) {
      resolve()
      return
    }
    const timer = window.setTimeout(finish, delayMs)
    signal.addEventListener('abort', finish, { once: true })

    function finish() {
      window.clearTimeout(timer)
      signal.removeEventListener('abort', finish)
      resolve()
    }
  })
}

export const useSessionStore = defineStore('sessions', () => {
  const settingsStore = useSettingsStore()
  const sessions = ref<LocalSession[]>([])
  const activeThreadId = ref<string | null>(null)
  const activeResponse = ref<SessionResponse | null>(null)
  const reportData = ref<ReportData | null>(null)
  const draftMessages = ref<ChatMessage[]>([])
  const uiStatus = ref<SessionUiStatus>('idle')
  const notice = ref<ApiError | null>(null)
  const storageWarning = ref<string | null>(null)
  const apiReachable = ref<boolean | null>(null)
  const inFlight = ref(false)
  const unknownStatusIds = ref(new Set<string>())
  const responsesByThreadId = new Map<string, SessionResponse>()
  const streamingAnswer = ref<{ threadId: string; message: ChatMessage } | null>(null)
  let pendingDelta = ''
  let deltaFrameScheduled = false
  let cancelDeltaFrame: (() => void) | null = null
  let currentController: AbortController | null = null
  let cancellationController: AbortController | null = null
  let activeRun: { threadId: string; requestId: string; cancellationRequested: boolean } | null =
    null

  const activeSession = computed(
    () => sessions.value.find((session) => session.threadId === activeThreadId.value) ?? null,
  )
  const visibleMessages = computed(() => {
    const messages = activeSession.value?.messages ?? draftMessages.value
    const streaming = streamingAnswer.value
    if (!streaming?.message.content || streaming.threadId !== activeThreadId.value) return messages
    return [...messages, streaming.message]
  })
  const canExitRequest = computed(
    () => inFlight.value && activeRun !== null && !activeRun.cancellationRequested,
  )

  function requireSettings() {
    if (!settingsStore.settings) {
      throw new ApiError(0, 'validation_error', '请先完成 Agent 工作区设置。')
    }
    return settingsStore.settings
  }

  function persist() {
    const settings = settingsStore.settings
    if (!settings) return
    const result = saveSessionRecords(
      localStorage,
      settings.userId,
      settings.projectId,
      sessions.value,
    )
    storageWarning.value = result.warning ?? null
  }

  function initialize() {
    if (!settingsStore.settings) settingsStore.load()
    const settings = settingsStore.settings
    if (!settings) return
    const result = loadSessionRecords(localStorage, settings.userId, settings.projectId)
    sessions.value = result.value
    unknownStatusIds.value = new Set()
    responsesByThreadId.clear()
    storageWarning.value = result.warning ?? null
  }

  // Browser records from before history moved to the backend are uploaded once.
  async function importLocalSessions(
    api: ReturnType<typeof createSessionApi>,
    userId: string,
    projectId: string,
  ) {
    if (isSessionImportDone(localStorage, userId, projectId)) return
    const results = await Promise.allSettled(
      sessions.value.map((record) =>
        api.importSession({
          thread_id: record.threadId,
          title: record.title,
          status: record.status,
          created_at: record.createdAt,
          updated_at: record.updatedAt,
          messages: record.messages
            .filter((message) => message.delivery === 'sent' || message.delivery === 'cancelled')
            .map((message) => ({
              id: message.id,
              role: message.role,
              content: message.content,
              delivery: message.delivery,
              created_at: message.createdAt,
              ...(message.requestId ? { request_id: message.requestId } : {}),
            })),
        }),
      ),
    )
    // Rejected records can never import, so only an outage is worth retrying.
    const outage = results.some(
      (result) =>
        result.status === 'rejected' &&
        !(
          result.reason instanceof ApiError &&
          result.reason.status >= 400 &&
          result.reason.status < 500
        ),
    )
    if (outage) throw new ApiError(0, 'import_failed', '本机会话暂时无法上传。')
    markSessionImportDone(localStorage, userId, projectId)
  }

  async function syncFromBackend() {
    const settings = settingsStore.settings
    if (!settings) return false
    const api = createSessionApi(settings)
    try {
      await api.ensureProject()
      await importLocalSessions(api, settings.userId, settings.projectId)
      const { sessions: stored } = await api.listSessions()
      const local = new Map(sessions.value.map((record) => [record.threadId, record]))
      const merged: LocalSession[] = stored.map((summary) => {
        const record = local.get(summary.thread_id)
        if (summary.thread_id === activeRun?.threadId && record) return record
        return {
          threadId: summary.thread_id,
          userId: settings.userId,
          projectId: settings.projectId,
          title: summary.title,
          createdAt: summary.created_at,
          updatedAt: summary.updated_at,
          status: summary.status,
          messages: record?.messages ?? [],
        }
      })
      // A session absent from the backend was deleted elsewhere, unless it is
      // still being sent or failed before the backend ever stored it.
      const stillLocal = sessions.value.filter(
        (record) =>
          !stored.some((summary) => summary.thread_id === record.threadId) &&
          (record.threadId === activeRun?.threadId ||
            record.messages.some((message) => message.delivery === 'failed')),
      )
      sessions.value = [...stillLocal, ...merged]
      storageWarning.value = null
      persist()
      return true
    } catch {
      storageWarning.value = '暂时无法读取后端会话，当前显示本机缓存。'
      return false
    }
  }

  async function loadActiveMessages() {
    const record = activeSession.value
    const settings = settingsStore.settings
    if (!record || !settings || record.threadId === activeRun?.threadId) return
    let stored
    try {
      stored = await createSessionApi(settings).listMessages(record.threadId)
    } catch {
      // Sessions not stored yet, or an outage, keep whatever is cached locally.
      return
    }
    if (record.threadId === activeRun?.threadId) return
    const messages: ChatMessage[] = stored.messages
      // Clarifications render as a card from the session response, not as bubbles.
      .filter((message) => message.kind === 'message')
      .map((message) => ({
        id: message.id,
        role: message.role,
        content: message.content,
        createdAt: message.created_at,
        delivery: message.delivery,
        requestId: message.request_id,
      }))
    const storedRequestIds = new Set(messages.map((message) => message.requestId))
    const undelivered = record.messages.filter(
      (message) =>
        message.delivery === 'failed' &&
        !(message.requestId && storedRequestIds.has(message.requestId)),
    )
    record.messages = [...messages, ...undelivered]
    persist()
  }

  function startNewSession() {
    activeThreadId.value = null
    activeResponse.value = null
    reportData.value = null
    draftMessages.value = []
    uiStatus.value = 'idle'
    notice.value = null
  }

  function selectSession(threadId: string) {
    const record = sessions.value.find((session) => session.threadId === threadId)
    if (!record) return false
    const cachedResponse = responsesByThreadId.get(threadId) ?? null
    activeThreadId.value = threadId
    activeResponse.value = cachedResponse
    reportData.value = cachedResponse ? parseReportData(cachedResponse.report_data) : null
    uiStatus.value = cachedResponse?.progress?.active
      ? 'running'
      : (cachedResponse?.status ?? record.status)
    notice.value = null
    return true
  }

  function cacheResponse(record: LocalSession, response: SessionResponse) {
    responsesByThreadId.set(record.threadId, response)
    if (activeThreadId.value !== record.threadId) return
    activeResponse.value = response
    reportData.value = parseReportData(response.report_data)
    uiStatus.value = response.progress?.active ? 'running' : response.status
  }

  function appendAssistant(record: LocalSession, response: SessionResponse) {
    const content = assistantContent(response)
    if (!content) return
    const isDuplicate = record.messages.some(
      (message) => message.role === 'assistant' && message.content === content,
    )
    if (isDuplicate) return
    record.messages.push({
      id: newId(),
      role: 'assistant',
      content,
      createdAt: now(),
      delivery: 'sent',
      ...(response.request_id ? { requestId: response.request_id } : {}),
    })
  }

  function reconcileRecoveredUserMessage(record: LocalSession, response: SessionResponse) {
    if (response.status === 'running' || response.progress?.active) return
    const requestId = response.request_id ?? response.progress?.request_id
    if (requestId) {
      for (let index = record.messages.length - 1; index >= 0; index -= 1) {
        const message = record.messages[index]
        if (
          message?.role === 'user' &&
          message.delivery === 'pending' &&
          message.requestId === requestId
        ) {
          message.delivery = response.status === 'cancelled' ? 'cancelled' : 'sent'
          return
        }
      }
    }

    const content = assistantContent(response)
    let assistantIndex = -1
    for (let index = record.messages.length - 1; index > 0; index -= 1) {
      const message = record.messages[index]
      if (message?.role === 'assistant' && message.content === content) {
        assistantIndex = index
        break
      }
    }
    if (assistantIndex < 1) return
    const previousMessage = record.messages[assistantIndex - 1]
    if (previousMessage?.role === 'user' && previousMessage.delivery === 'pending') {
      previousMessage.delivery = 'sent'
    }
  }

  function beginAnswerStream(threadId: string, requestId: string) {
    endAnswerStream()
    streamingAnswer.value = {
      threadId,
      message: {
        id: newId(),
        role: 'assistant',
        content: '',
        createdAt: now(),
        delivery: 'streaming',
        requestId,
      },
    }
  }

  function flushAnswerDelta() {
    deltaFrameScheduled = false
    cancelDeltaFrame = null
    if (streamingAnswer.value) streamingAnswer.value.message.content += pendingDelta
    pendingDelta = ''
  }

  function appendAnswerDelta(requestId: string, text: string) {
    if (streamingAnswer.value?.message.requestId !== requestId) return
    // Coalesce tokens per frame so Markdown re-renders at most once a frame.
    pendingDelta += text
    if (deltaFrameScheduled) return
    deltaFrameScheduled = true
    cancelDeltaFrame = scheduleFrame(flushAnswerDelta)
  }

  function discardPendingDelta() {
    cancelDeltaFrame?.()
    cancelDeltaFrame = null
    deltaFrameScheduled = false
    pendingDelta = ''
  }

  function resetAnswerStream(requestId: string) {
    if (streamingAnswer.value?.message.requestId !== requestId) return
    discardPendingDelta()
    streamingAnswer.value.message.content = ''
  }

  function endAnswerStream(requestId?: string) {
    if (requestId && streamingAnswer.value?.message.requestId !== requestId) return
    discardPendingDelta()
    streamingAnswer.value = null
  }

  function applyResponse(record: LocalSession, response: SessionResponse) {
    const nextStatus = response.progress?.active ? 'running' : response.status
    const statusChanged = record.status !== nextStatus
    const messageCount = record.messages.length
    record.status = nextStatus
    appendAssistant(record, response)
    reconcileRecoveredUserMessage(record, response)
    // Merely syncing an unchanged session must not make it look recently active.
    if (statusChanged || record.messages.length !== messageCount) record.updatedAt = now()
    cacheResponse(record, response)
  }

  async function pollWorkflowProgress(
    api: ReturnType<typeof createSessionApi>,
    threadId: string,
    requestId: string,
    signal: AbortSignal,
  ) {
    while (!signal.aborted) {
      await waitForPoll(signal)
      if (signal.aborted) return
      try {
        const response = await api.getSession(threadId, signal)
        if (signal.aborted) return
        if (response.progress?.request_id !== requestId) continue
        const record = sessions.value.find((session) => session.threadId === threadId)
        if (!record) return
        const nextStatus = response.progress.active ? 'running' : response.status
        if (record.status !== nextStatus) record.updatedAt = now()
        record.status = nextStatus
        cacheResponse(record, response)
      } catch {
        if (signal.aborted) return
      }
    }
  }

  async function submit(message: string, wantsDocx: boolean): Promise<string | null> {
    const normalized = message.trim()
    if (!normalized || normalized.length > 10_000) {
      notice.value = new ApiError(422, 'validation_error', '消息长度必须在 1 到 10,000 字符之间。')
      uiStatus.value = 'error'
      return activeThreadId.value
    }
    if (inFlight.value) {
      notice.value = new ApiError(409, 'session_busy', '该会话已有请求正在处理。')
      return activeThreadId.value
    }

    let settings
    try {
      settings = requireSettings()
    } catch (error) {
      notice.value = apiErrorFromUnknown(error)
      uiStatus.value = 'error'
      return null
    }

    const api = createSessionApi(settings)
    const requestId = newId()
    const outgoing = userMessage(normalized, requestId)
    const existing = activeSession.value
    let record: LocalSession
    if (existing) {
      record = existing
      record.messages.push(outgoing)
      record.updatedAt = now()
    } else {
      const createdAt = outgoing.createdAt
      sessions.value.unshift({
        threadId: newId(),
        userId: settings.userId,
        projectId: settings.projectId,
        title: titleFor(normalized),
        createdAt,
        updatedAt: createdAt,
        status: 'running',
        messages: [outgoing],
      })
      // Keep the reactive proxy: writes to the raw object would never reach the sidebar.
      record = sessions.value[0]!
      activeThreadId.value = record.threadId
      draftMessages.value = []
    }
    const setOutgoingDelivery = (delivery: ChatMessage['delivery']) => {
      const messageToUpdate = record.messages.find((item) => item.id === outgoing.id) ?? outgoing
      messageToUpdate.delivery = delivery
    }

    notice.value = null
    uiStatus.value = 'submitting'
    inFlight.value = true
    currentController = new AbortController()
    const cachedResponse = responsesByThreadId.get(record.threadId)
    const previousResponse =
      activeResponse.value?.thread_id === record.threadId ? activeResponse.value : cachedResponse
    const runningResponse = previousResponse
      ? {
          ...previousResponse,
          request_id: requestId,
          status: 'running' as const,
          progress: {
            request_id: requestId,
            active: true,
            cancel_requested: false,
            cancelled: false,
            current_stage: null,
            events: [],
          },
        }
      : initialRunningResponse(settings.projectId, record.threadId, requestId)
    record.status = 'running'
    cacheResponse(record, runningResponse)
    persist()

    const progressController = new AbortController()
    activeRun = {
      threadId: record.threadId,
      requestId,
      cancellationRequested: false,
    }
    currentController.signal.addEventListener('abort', () => progressController.abort(), {
      once: true,
    })
    const progressTask = pollWorkflowProgress(
      api,
      record.threadId,
      requestId,
      progressController.signal,
    )
    beginAnswerStream(record.threadId, requestId)
    const answerHandlers = {
      onReset: () => resetAnswerStream(requestId),
      onDelta: (text: string) => appendAnswerDelta(requestId, text),
    }

    try {
      const payload = {
        message: normalized,
        report_format: wantsDocx ? 'docx' : 'markdown',
      } as const
      const response = existing
        ? await api.sendMessageStream(
            record.threadId,
            payload,
            requestId,
            answerHandlers,
            currentController.signal,
          )
        : await api.createSessionStream(
            payload,
            requestId,
            record.threadId,
            answerHandlers,
            currentController.signal,
          )

      progressController.abort()
      await progressTask
      endAnswerStream(requestId)

      if (!existing && response.thread_id !== record.threadId) {
        throw new ApiError(502, 'thread_id_mismatch', '服务端返回的会话标识与本次请求不一致。')
      }
      if (activeRun?.requestId !== requestId || activeRun.cancellationRequested) {
        return record.threadId
      }
      setOutgoingDelivery('sent')
      applyResponse(record, response)
      persist()
      return response.thread_id
    } catch (error) {
      progressController.abort()
      await progressTask
      const cancellationRequested =
        activeRun?.requestId === requestId && activeRun.cancellationRequested
      if (cancellationRequested) {
        return record.threadId
      }
      const latestResponse = responsesByThreadId.get(record.threadId)
      if (latestResponse?.progress?.request_id === requestId) {
        cacheResponse(record, {
          ...latestResponse,
          progress: {
            ...latestResponse.progress,
            active: false,
            current_stage: null,
          },
        })
      }
      const apiError = apiErrorFromUnknown(error)
      notice.value = apiError
      setOutgoingDelivery('failed')
      uiStatus.value = 'error'
      // A 409 means the server is still working, so the running status stays accurate.
      if (apiError.status !== 409) record.status = 'error'
      record.updatedAt = now()
      persist()
      return activeThreadId.value
    } finally {
      progressController.abort()
      endAnswerStream(requestId)
      if (activeRun?.requestId === requestId && !activeRun.cancellationRequested) {
        activeRun = null
        inFlight.value = false
        currentController = null
      }
    }
  }

  function completeCancellation(
    record: LocalSession,
    response: SessionResponse,
    requestId: string,
  ) {
    applyResponse(record, response)
    const outgoing = record.messages.find(
      (message) => message.role === 'user' && message.requestId === requestId,
    )
    if (outgoing?.delivery === 'pending') {
      outgoing.delivery = response.status === 'cancelled' ? 'cancelled' : 'sent'
    }
    persist()
    cancellationController = null
    currentController = null
    activeRun = null
    inFlight.value = false
  }

  async function waitForCancellation(
    api: ReturnType<typeof createSessionApi>,
    threadId: string,
    requestId: string,
    signal: AbortSignal,
  ) {
    while (!signal.aborted && activeRun?.requestId === requestId) {
      try {
        const response = await api.getSession(threadId, signal)
        if (signal.aborted || activeRun?.requestId !== requestId) return
        if (!response.progress?.active) {
          const record = sessions.value.find((session) => session.threadId === threadId)
          if (record) completeCancellation(record, response, requestId)
          return
        }
      } catch {
        if (signal.aborted || activeRun?.requestId !== requestId) return
      }
      await waitForPoll(signal)
    }
  }

  async function exitCurrentRequest() {
    const run = activeRun
    if (!run || run.cancellationRequested) return false
    try {
      const api = createSessionApi(requireSettings())
      const cancellation = await api.cancelSession(run.threadId)
      if (activeRun !== run || cancellation.status === 'not_running') return false

      run.cancellationRequested = true
      notice.value = null
      if (activeThreadId.value === run.threadId) uiStatus.value = 'cancelling'
      currentController?.abort()
      cancellationController?.abort()
      cancellationController = new AbortController()
      void waitForCancellation(
        api,
        run.threadId,
        run.requestId,
        cancellationController.signal,
      )
      return true
    } catch (error) {
      notice.value = apiErrorFromUnknown(error)
      return false
    }
  }

  async function syncActiveSession() {
    const record = activeSession.value
    if (!record) return
    const api = createSessionApi(requireSettings())
    notice.value = null
    uiStatus.value = 'running'
    try {
      const response = await api.getSession(record.threadId)
      applyResponse(record, response)
      unknownStatusIds.value.delete(record.threadId)
      persist()
    } catch (error) {
      if (activeThreadId.value !== record.threadId) return
      notice.value = apiErrorFromUnknown(error)
      uiStatus.value = 'error'
    }
  }

  // Records left "running" by a closed tab or an older build would otherwise never update.
  async function refreshRunningSessions() {
    const settings = settingsStore.settings
    if (!settings) return
    const api = createSessionApi(settings)
    const running = sessions.value.filter(
      (session) => session.status === 'running' && session.threadId !== activeRun?.threadId,
    )
    await Promise.all(
      running.map(async (record) => {
        try {
          const response = await api.getSession(record.threadId)
          if (record.threadId === activeRun?.threadId) return
          applyResponse(record, response)
          unknownStatusIds.value.delete(record.threadId)
        } catch {
          unknownStatusIds.value.add(record.threadId)
        }
      }),
    )
    if (running.length) persist()
  }

  function discardFailedMessage(messageId: string) {
    const record = activeSession.value
    if (!record || inFlight.value) return false
    const message = record.messages.find((item) => item.id === messageId)
    if (message?.delivery !== 'failed') return false
    record.messages = record.messages.filter((item) => item.id !== messageId)
    notice.value = null
    if (record.messages.length) {
      uiStatus.value = record.status
      persist()
    } else {
      // Nothing else was delivered, so the local record has nothing left to keep.
      removeLocalSession(record.threadId)
    }
    return true
  }

  async function retryFailedMessage(messageId: string, wantsDocx: boolean) {
    const message = activeSession.value?.messages.find((item) => item.id === messageId)
    if (!message || !discardFailedMessage(messageId)) return activeThreadId.value
    return submit(message.content, wantsDocx)
  }

  async function checkHealth() {
    try {
      await createSessionApi(requireSettings()).health()
      apiReachable.value = true
    } catch {
      apiReachable.value = false
    }
  }

  function removeLocalSession(threadId: string) {
    sessions.value = sessions.value.filter((session) => session.threadId !== threadId)
    responsesByThreadId.delete(threadId)
    if (activeThreadId.value === threadId) startNewSession()
    persist()
  }

  async function closeActiveSession() {
    const record = activeSession.value
    if (!record || inFlight.value) return false
    try {
      await createSessionApi(requireSettings()).closeSession(record.threadId)
      removeLocalSession(record.threadId)
      return true
    } catch (error) {
      notice.value = apiErrorFromUnknown(error)
      uiStatus.value = 'error'
      return false
    }
  }

  async function downloadReport(
    reportUrl = activeResponse.value?.report_url ?? null,
    fallbackFilename?: string,
  ) {
    if (!reportUrl) return null
    try {
      notice.value = null
      return await createSessionApi(requireSettings()).downloadReport(reportUrl, fallbackFilename)
    } catch (error) {
      notice.value = apiErrorFromUnknown(error)
      return null
    }
  }

  return {
    sessions,
    activeThreadId,
    activeResponse,
    reportData,
    activeSession,
    draftMessages,
    visibleMessages,
    uiStatus,
    notice,
    storageWarning,
    apiReachable,
    inFlight,
    canExitRequest,
    unknownStatusIds,
    initialize,
    syncFromBackend,
    loadActiveMessages,
    startNewSession,
    selectSession,
    submit,
    exitCurrentRequest,
    syncActiveSession,
    refreshRunningSessions,
    discardFailedMessage,
    retryFailedMessage,
    checkHealth,
    removeLocalSession,
    closeActiveSession,
    downloadReport,
  }
})
