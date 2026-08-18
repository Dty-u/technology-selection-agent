<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { FolderClock, PanelLeft, PanelRight, RefreshCw, Trash2 } from '@lucide/vue'
import { useRoute, useRouter } from 'vue-router'

import SettingsDrawer from '@/features/settings/SettingsDrawer.vue'
import { useSettingsStore } from '@/features/settings/settingsStore'
import AppDrawer from '@/shared/components/AppDrawer.vue'
import RequestNotice from '@/shared/components/RequestNotice.vue'
import ConversationPane from '../components/ConversationPane.vue'
import InspectorPanel from '../components/InspectorPanel.vue'
import SessionSidebar from '../components/SessionSidebar.vue'
import { useSessionStore } from '../store/sessionStore'

const route = useRoute()
const router = useRouter()
const settingsStore = useSettingsStore()
const store = useSessionStore()
const leftDrawerOpen = ref(false)
const rightDrawerOpen = ref(false)
const settingsOpen = ref(false)

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
  await store.syncActiveSession()
}

onMounted(async () => {
  settingsStore.load()
  store.initialize()
  await Promise.all([store.checkHealth(), restoreFromRoute(threadId.value)])
})

watch(threadId, (value, previous) => {
  if (value !== previous && value !== store.activeThreadId) void restoreFromRoute(value)
})

async function submit(message: string, wantsDocx: boolean) {
  const returnedThreadId = await store.submit(message, wantsDocx)
  if (returnedThreadId && returnedThreadId !== threadId.value) {
    await router.replace(`/workspace/${encodeURIComponent(returnedThreadId)}`)
  }
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

function removeLocal(value: string) {
  if (!window.confirm('仅移除本机记录？服务端 checkpoint 和消息不会被删除。')) return
  store.removeLocalSession(value)
  if (value === threadId.value) void router.push('/workspace')
}

async function closeServerSession() {
  if (!store.activeSession) return
  const confirmed = window.confirm(
    '关闭会话会删除服务端 checkpoint、短期消息和安全报告，但不会删除已批准的长期记忆。确认继续？',
  )
  if (!confirmed) return
  if (await store.closeActiveSession()) await router.push('/workspace')
}

async function downloadReport() {
  const file = await store.downloadActiveReport()
  if (!file) return
  const url = URL.createObjectURL(file.blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = file.filename
  anchor.click()
  URL.revokeObjectURL(url)
}

function settingsSaved() {
  settingsOpen.value = false
  store.startNewSession()
  store.initialize()
  void router.push('/workspace')
  void store.checkHealth()
}
</script>

<template>
  <main class="workspace-view">
    <div class="desktop-sidebar">
      <SessionSidebar
        :sessions="store.sessions"
        :active-thread-id="store.activeThreadId"
        :api-reachable="store.apiReachable"
        @create="startNew"
        @select="selectSession"
        @settings="settingsOpen = true"
        @remove-local="removeLocal"
      />
    </div>
    <section class="workspace-center">
      <header class="workspace-topbar">
        <button
          class="icon-button mobile-only"
          type="button"
          aria-label="打开会话列表"
          @click="leftDrawerOpen = true"
        >
          <PanelLeft :size="20" />
        </button>
        <div>
          <span class="eyebrow">ACTIVE THREAD</span><strong>{{ workspaceTitle }}</strong>
        </div>
        <div class="topbar-actions">
          <button
            v-if="store.activeSession"
            class="icon-button"
            type="button"
            aria-label="同步最新状态"
            @click="store.syncActiveSession"
          >
            <RefreshCw :size="18" />
          </button>
          <button
            v-if="store.activeSession"
            class="icon-button danger-action"
            type="button"
            aria-label="关闭服务端会话"
            @click="closeServerSession"
          >
            <Trash2 :size="18" />
          </button>
          <button
            class="icon-button mobile-only"
            type="button"
            aria-label="打开证据检查器"
            @click="rightDrawerOpen = true"
          >
            <PanelRight :size="20" />
          </button>
        </div>
      </header>
      <RequestNotice
        v-if="store.storageWarning"
        tone="warning"
        title="本机保存受限"
        :message="store.storageWarning"
      />
      <ConversationPane
        :messages="store.visibleMessages"
        :response="store.activeResponse"
        :status="store.uiStatus"
        :notice="store.notice"
        :in-flight="store.inFlight"
        :can-stop="store.canStopWaiting"
        @submit="submit"
        @stop="store.stopWaiting"
        @sync="store.syncActiveSession"
      />
    </section>
    <div class="desktop-inspector">
      <InspectorPanel
        :response="store.activeResponse"
        :report-data="store.reportData"
        :status="store.uiStatus"
        @download="downloadReport"
      />
    </div>

    <AppDrawer :open="leftDrawerOpen" title="本机会话" side="left" @close="leftDrawerOpen = false">
      <SessionSidebar
        :sessions="store.sessions"
        :active-thread-id="store.activeThreadId"
        :api-reachable="store.apiReachable"
        @create="startNew"
        @select="selectSession"
        @settings="settingsOpen = true"
        @remove-local="removeLocal"
      />
    </AppDrawer>
    <AppDrawer
      :open="rightDrawerOpen"
      title="证据与结论"
      side="right"
      @close="rightDrawerOpen = false"
    >
      <InspectorPanel
        :response="store.activeResponse"
        :report-data="store.reportData"
        :status="store.uiStatus"
        @download="downloadReport"
      />
    </AppDrawer>
    <AppDrawer :open="settingsOpen" title="工作区设置" side="right" @close="settingsOpen = false"
      ><SettingsDrawer @saved="settingsSaved"
    /></AppDrawer>
    <div
      v-if="!store.sessions.length && !store.visibleMessages.length"
      class="workspace-watermark"
      aria-hidden="true"
    >
      <FolderClock :size="14" /> 会话在首次成功响应后保存
    </div>
  </main>
</template>
