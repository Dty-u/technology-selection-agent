import { computed, ref } from 'vue'
import { defineStore } from 'pinia'

import { useSettingsStore } from '@/features/settings/settingsStore'
import { ApiError, apiErrorFromUnknown } from '@/shared/api/errors'
import { createSessionApi } from '../api/sessionApi'
import { parseReportData, type ReportData, type SessionResponse } from '../api/sessionSchemas'
import type { ChatMessage, LocalSession, SessionUiStatus } from '../model/sessionTypes'
import { loadSessionRecords, saveSessionRecords } from '../storage/sessionRepository'

function newId(): string {
  return globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random().toString(16).slice(2)}`
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
  let currentController: AbortController | null = null

  const activeSession = computed(
    () => sessions.value.find((session) => session.threadId === activeThreadId.value) ?? null,
  )
  const visibleMessages = computed(() => activeSession.value?.messages ?? draftMessages.value)
  const canStopWaiting = computed(
    () => inFlight.value && activeThreadId.value !== null && currentController !== null,
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
    storageWarning.value = result.warning ?? null
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
    activeThreadId.value = threadId
    activeResponse.value = null
    reportData.value = null
    uiStatus.value = record.status
    notice.value = null
    return true
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

  function applyResponse(record: LocalSession, response: SessionResponse) {
    record.status = response.status
    record.updatedAt = now()
    appendAssistant(record, response)
    activeResponse.value = response
    reportData.value = parseReportData(response.report_data)
    uiStatus.value = response.status
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
    if (existing) {
      existing.messages.push(outgoing)
      existing.updatedAt = now()
      persist()
    } else {
      draftMessages.value = [outgoing]
    }

    notice.value = null
    uiStatus.value = 'submitting'
    inFlight.value = true
    currentController = new AbortController()

    try {
      const payload = {
        message: normalized,
        report_format: wantsDocx ? 'docx' : 'markdown',
      } as const
      const response = existing
        ? await api.sendMessage(existing.threadId, payload, requestId, currentController.signal)
        : await api.createSession(payload, requestId, currentController.signal)

      outgoing.delivery = 'sent'
      if (existing) {
        applyResponse(existing, response)
      } else {
        const createdAt = outgoing.createdAt
        const record: LocalSession = {
          threadId: response.thread_id,
          userId: settings.userId,
          projectId: settings.projectId,
          title: titleFor(normalized),
          createdAt,
          updatedAt: now(),
          status: response.status,
          messages: [outgoing],
        }
        appendAssistant(record, response)
        sessions.value.unshift(record)
        activeThreadId.value = record.threadId
        draftMessages.value = []
        activeResponse.value = response
        reportData.value = parseReportData(response.report_data)
        uiStatus.value = response.status
      }
      persist()
      return response.thread_id
    } catch (error) {
      const apiError = apiErrorFromUnknown(error)
      notice.value = apiError
      if (apiError.code === 'request_aborted' && existing) {
        uiStatus.value = 'detached'
      } else {
        outgoing.delivery = 'failed'
        uiStatus.value = 'error'
      }
      persist()
      return activeThreadId.value
    } finally {
      inFlight.value = false
      currentController = null
    }
  }

  function stopWaiting() {
    if (canStopWaiting.value) currentController?.abort()
  }

  async function syncActiveSession() {
    const record = activeSession.value
    if (!record || inFlight.value) return
    const api = createSessionApi(requireSettings())
    notice.value = null
    uiStatus.value = 'running'
    try {
      const response = await api.getSession(record.threadId)
      applyResponse(record, response)
      persist()
    } catch (error) {
      notice.value = apiErrorFromUnknown(error)
      uiStatus.value = 'error'
    }
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

  async function downloadActiveReport() {
    const reportUrl = activeResponse.value?.report_url
    if (!reportUrl) return null
    try {
      notice.value = null
      return await createSessionApi(requireSettings()).downloadReport(reportUrl)
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
    canStopWaiting,
    initialize,
    startNewSession,
    selectSession,
    submit,
    stopWaiting,
    syncActiveSession,
    checkHealth,
    removeLocalSession,
    closeActiveSession,
    downloadActiveReport,
  }
})
