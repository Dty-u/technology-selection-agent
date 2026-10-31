<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import {
  FolderClock,
  PanelLeft,
  PanelRight,
  PanelRightClose,
  PanelRightOpen,
  PowerOff,
  RefreshCw,
} from '@lucide/vue'
import { useRoute, useRouter } from 'vue-router'

import SettingsDrawer from '@/features/settings/SettingsDrawer.vue'
import { useSettingsStore } from '@/features/settings/settingsStore'
import AppDrawer from '@/shared/components/AppDrawer.vue'
import ConfirmDialog, { type ConfirmRequest } from '@/shared/components/ConfirmDialog.vue'
import RequestNotice from '@/shared/components/RequestNotice.vue'
import ConversationPane from '../components/ConversationPane.vue'
import InspectorPanel from '../components/InspectorPanel.vue'
import ReportLibrary from '../components/ReportLibrary.vue'
import SessionSidebar from '../components/SessionSidebar.vue'
import { useWorkspacePanelResize } from '../composables/useWorkspacePanelResize'
import { useSessionStore } from '../store/sessionStore'
import type { ReportLibraryItem, ReportSummary } from '../api/sessionSchemas'
import { saveDownloadedFile } from '../model/saveDownloadedFile'

const route = useRoute()
const router = useRouter()
const settingsStore = useSettingsStore()
const store = useSessionStore()
const leftDrawerOpen = ref(false)
const rightDrawerOpen = ref(false)
const settingsOpen = ref(false)
const reportLibraryOpen = ref(false)
const sourceFocus = ref<{ index: number; key: number } | null>(null)
const confirmRequest = ref<ConfirmRequest | null>(null)
let settleConfirmRequest: ((confirmed: boolean) => void) | null = null
const {
  inspectorCollapsed,
  setInspectorCollapsed,
  inspectorMaximum,
  inspectorMinimum,
  inspectorWidth,
  layoutStyle,
  sidebarMaximum,
  sidebarMinimum,
  sidebarWidth,
  resetWidth,
  resizeWithKeyboard,
  startResize,
} = useWorkspacePanelResize()

const threadId = computed(() =>
  typeof route.params.threadId === 'string' ? route.params.threadId : null,
)
const workspaceTitle = computed(() => store.activeSession?.title ?? '新会话')

async function restoreFromRoute(value: string | null) {
  if (!value) {
    store.startNewSession()
    return
  }
  if (!store.selectSession(value)) {
    store.startNewSession()
    return
  }
  await store.loadActiveMessages()
  await store.syncActiveSession()
}

onMounted(async () => {
  settingsStore.load()
  store.initialize()
  // Restoring a route needs the backend's session list to know the thread exists.
  await store.syncFromBackend()
  await Promise.all([
    store.checkHealth(),
    restoreFromRoute(threadId.value),
    store.refreshRunningSessions(),
  ])
})

watch(threadId, (value, previous) => {
  sourceFocus.value = null
  if (value !== previous && value !== store.activeThreadId) void restoreFromRoute(value)
})

async function followReturnedThread(returnedThreadId: string | null) {
  if (returnedThreadId && returnedThreadId !== threadId.value) {
    await router.replace(`/workspace/${encodeURIComponent(returnedThreadId)}`)
  }
}

async function submit(message: string, wantsDocx: boolean) {
  await followReturnedThread(await store.submit(message, wantsDocx))
}

async function retryMessage(messageId: string, wantsDocx: boolean) {
  await followReturnedThread(await store.retryFailedMessage(messageId, wantsDocx))
}

function discardFailedMessage(messageId: string) {
  store.discardFailedMessage(messageId)
  if (!store.activeThreadId && threadId.value) void router.replace('/workspace')
}

function focusSource(index: number) {
  sourceFocus.value = { index, key: (sourceFocus.value?.key ?? 0) + 1 }
  setInspectorCollapsed(false)
  if (window.matchMedia('(max-width: 960px)').matches) rightDrawerOpen.value = true
}

function closeRightDrawer() {
  rightDrawerOpen.value = false
  // Otherwise reopening the drawer would highlight the same source again.
  sourceFocus.value = null
}

function askConfirm(request: ConfirmRequest): Promise<boolean> {
  confirmRequest.value = request
  return new Promise((resolve) => {
    settleConfirmRequest = resolve
  })
}

function settleConfirm(confirmed: boolean) {
  confirmRequest.value = null
  settleConfirmRequest?.(confirmed)
  settleConfirmRequest = null
}

async function selectSession(value: string) {
  leftDrawerOpen.value = false
  if (value !== threadId.value) await router.push(`/workspace/${encodeURIComponent(value)}`)
  else await store.syncActiveSession()
}

async function startNew() {
  leftDrawerOpen.value = false
  store.startNewSession()
  await router.push('/workspace')
}

async function removeLocal(value: string) {
  const confirmed = await askConfirm({
    title: '从列表中移除？',
    message: '这个会话只会暂时从当前页面的列表中移除，刷新后会重新出现。',
    details: ['服务端的会话内容和消息不会被删除。要彻底删除，请使用“结束会话”。'],
    confirmLabel: '移除记录',
  })
  if (!confirmed) return
  store.removeLocalSession(value)
  if (value === threadId.value) void router.push('/workspace')
}

async function closeServerSession() {
  if (!store.activeSession) return
  const confirmed = await askConfirm({
    title: '结束这个会话？',
    message: '结束后无法继续这个会话，以下内容也无法恢复。',
    details: [
      '会删除：服务端的会话进度、全部消息记录和报告。',
      '会移除：会话列表里的这条记录。',
      '不会删除：已批准的长期记忆。',
    ],
    confirmLabel: '结束并删除',
  })
  if (!confirmed) return
  if (await store.closeActiveSession()) await router.push('/workspace')
}

async function downloadReport(report?: ReportSummary) {
  const fallbackFilename = report
    ? `${report.title}.${report.format === 'docx' ? 'docx' : 'md'}`
    : undefined
  const file = await store.downloadReport(report?.download_url, fallbackFilename)
  if (file) saveDownloadedFile(file)
}

function openReportLibrary() {
  leftDrawerOpen.value = false
  reportLibraryOpen.value = true
}

async function openReportSession(value: string) {
  reportLibraryOpen.value = false
  await selectSession(value)
}

function reportDeleted(report: ReportLibraryItem) {
  // The inspector may still be showing the report that was just deleted.
  if (report.thread_id === store.activeThreadId) void store.syncActiveSession()
}

function settingsSaved() {
  settingsOpen.value = false
  store.startNewSession()
  store.initialize()
  void router.push('/workspace')
  void store.checkHealth()
  void store.syncFromBackend().then(() => store.refreshRunningSessions())
}
</script>

<template>
  <main
    class="workspace-view"
    :class="{ 'is-inspector-collapsed': inspectorCollapsed }"
    :style="layoutStyle"
  >
    <div class="desktop-sidebar">
      <SessionSidebar
        :sessions="store.sessions"
        :active-thread-id="store.activeThreadId"
        :api-reachable="store.apiReachable"
        :unknown-status-ids="store.unknownStatusIds"
        @create="startNew"
        @select="selectSession"
        @settings="settingsOpen = true"
        @reports="openReportLibrary"
        @remove-local="removeLocal"
      />
      <button
        class="workspace-resizer workspace-resizer--left"
        type="button"
        role="separator"
        aria-label="调整会话列表宽度"
        aria-orientation="vertical"
        :aria-valuemin="sidebarMinimum"
        :aria-valuemax="Math.round(sidebarMaximum)"
        :aria-valuenow="Math.round(sidebarWidth)"
        title="拖动调整会话列表宽度；双击恢复默认"
        @pointerdown="startResize('sidebar', $event)"
        @keydown="resizeWithKeyboard('sidebar', $event)"
        @dblclick="resetWidth('sidebar')"
      >
        <span aria-hidden="true"></span>
      </button>
    </div>
    <section class="workspace-center">
      <header class="workspace-topbar">
        <button
          class="icon-button mobile-only"
          type="button"
          aria-label="打开会话列表"
          title="打开会话列表"
          @click="leftDrawerOpen = true"
        >
          <PanelLeft :size="20" />
        </button>
        <div class="workspace-topbar__identity">
          <span class="eyebrow">当前会话</span><strong>{{ workspaceTitle }}</strong>
        </div>
        <div class="topbar-actions">
          <button
            v-if="store.activeSession"
            class="icon-button"
            type="button"
            aria-label="同步最新状态"
            title="同步最新状态"
            @click="store.syncActiveSession"
          >
            <RefreshCw :size="18" />
          </button>
          <button
            class="icon-button mobile-only"
            type="button"
            aria-label="打开证据检查器"
            title="打开执行与证据面板"
            @click="rightDrawerOpen = true"
          >
            <PanelRight :size="20" />
          </button>
          <button
            class="icon-button desktop-only"
            type="button"
            :aria-label="inspectorCollapsed ? '展开执行与证据面板' : '收起执行与证据面板'"
            :aria-expanded="!inspectorCollapsed"
            :title="inspectorCollapsed ? '展开执行与证据面板' : '收起执行与证据面板'"
            @click="setInspectorCollapsed(!inspectorCollapsed)"
          >
            <PanelRightOpen v-if="inspectorCollapsed" :size="18" />
            <PanelRightClose v-else :size="18" />
          </button>
          <button
            v-if="store.activeSession"
            class="icon-button danger-action topbar-actions__end-session"
            type="button"
            aria-label="结束会话并删除服务端记录"
            title="结束会话并删除服务端记录"
            @click="closeServerSession"
          >
            <PowerOff :size="18" />
          </button>
        </div>
      </header>
      <RequestNotice
        v-if="store.storageWarning"
        tone="warning"
        title="会话同步受限"
        :message="store.storageWarning"
      />
      <ConversationPane
        :messages="store.visibleMessages"
        :response="store.activeResponse"
        :status="store.uiStatus"
        :notice="store.notice"
        :in-flight="store.inFlight"
        :can-exit="store.canExitRequest"
        :session-key="store.activeThreadId"
        @submit="submit"
        @exit="store.exitCurrentRequest"
        @retry="retryMessage"
        @discard-failed="discardFailedMessage"
        @focus-source="focusSource"
      />
    </section>
    <div class="desktop-inspector">
      <button
        class="workspace-resizer workspace-resizer--right"
        type="button"
        role="separator"
        aria-label="调整检查器宽度"
        aria-orientation="vertical"
        :aria-valuemin="inspectorMinimum"
        :aria-valuemax="Math.round(inspectorMaximum)"
        :aria-valuenow="Math.round(inspectorWidth)"
        title="拖动调整检查器宽度；双击恢复默认"
        @pointerdown="startResize('inspector', $event)"
        @keydown="resizeWithKeyboard('inspector', $event)"
        @dblclick="resetWidth('inspector')"
      >
        <span aria-hidden="true"></span>
      </button>
      <InspectorPanel
        :response="store.activeResponse"
        :report-data="store.reportData"
        :status="store.uiStatus"
        :source-focus="sourceFocus"
        @download="downloadReport"
      />
    </div>

    <AppDrawer :open="leftDrawerOpen" title="会话" side="left" @close="leftDrawerOpen = false">
      <SessionSidebar
        :sessions="store.sessions"
        :active-thread-id="store.activeThreadId"
        :api-reachable="store.apiReachable"
        :unknown-status-ids="store.unknownStatusIds"
        @create="startNew"
        @select="selectSession"
        @settings="settingsOpen = true"
        @reports="openReportLibrary"
        @remove-local="removeLocal"
      />
    </AppDrawer>
    <AppDrawer
      :open="rightDrawerOpen"
      title="执行、证据与结论"
      side="right"
      @close="closeRightDrawer"
    >
      <InspectorPanel
        :response="store.activeResponse"
        :report-data="store.reportData"
        :status="store.uiStatus"
        :source-focus="sourceFocus"
        @download="downloadReport"
      />
    </AppDrawer>
    <AppDrawer :open="settingsOpen" title="工作区设置" side="right" @close="settingsOpen = false"
      ><SettingsDrawer @saved="settingsSaved"
    /></AppDrawer>
    <AppDrawer
      :open="reportLibraryOpen"
      title="报告库"
      side="right"
      wide
      @close="reportLibraryOpen = false"
    >
      <ReportLibrary
        :current-project-id="settingsStore.settings?.projectId ?? null"
        :confirm="askConfirm"
        @open-session="openReportSession"
        @deleted="reportDeleted"
      />
    </AppDrawer>
    <ConfirmDialog :request="confirmRequest" @settle="settleConfirm" />
    <div
      v-if="!store.sessions.length && !store.visibleMessages.length"
      class="workspace-watermark"
      aria-hidden="true"
    >
      <FolderClock :size="14" /> 会话在首次成功响应后保存
    </div>
  </main>
</template>
