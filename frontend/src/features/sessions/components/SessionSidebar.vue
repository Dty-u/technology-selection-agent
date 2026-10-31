<script setup lang="ts">
import { computed, ref } from 'vue'
import { CircuitBoard, FileText, Plus, Radio, Search, Settings2, Trash2 } from '@lucide/vue'

import AppButton from '@/shared/components/AppButton.vue'
import StatusBadge from '@/shared/components/StatusBadge.vue'
import { formatSessionTime, sessionGroupLabel } from '../model/sessionTime'
import type { LocalSession } from '../model/sessionTypes'

const SEARCH_THRESHOLD = 8

const props = defineProps<{
  sessions: LocalSession[]
  activeThreadId: string | null
  apiReachable: boolean | null
  unknownStatusIds?: ReadonlySet<string>
}>()
const emit = defineEmits<{
  create: []
  select: [threadId: string]
  settings: []
  reports: []
  removeLocal: [threadId: string]
}>()

const query = ref('')
const showSearch = computed(() => props.sessions.length > SEARCH_THRESHOLD)
const groups = computed(() => {
  const keyword = showSearch.value ? query.value.trim().toLowerCase() : ''
  const matching = props.sessions
    .filter((session) => session.title.toLowerCase().includes(keyword))
    .sort((left, right) => right.updatedAt.localeCompare(left.updatedAt))
  const grouped = new Map<string, LocalSession[]>()
  for (const session of matching) {
    const label = sessionGroupLabel(session.updatedAt)
    grouped.set(label, [...(grouped.get(label) ?? []), session])
  }
  return [...grouped].map(([label, sessions]) => ({ label, sessions }))
})
const apiStatusLabel = computed(() => {
  if (props.apiReachable === true) return '服务已连接'
  return props.apiReachable === false ? '服务未连接' : '正在检测服务'
})
</script>

<template>
  <aside class="session-sidebar">
    <header class="workspace-brand">
      <div class="brand-mark"><CircuitBoard :size="20" aria-hidden="true" /></div>
      <div><strong>晶圆智选</strong><span>DECISION LAB</span></div>
    </header>
    <AppButton @click="emit('create')"><Plus :size="17" aria-hidden="true" /> 新建会话</AppButton>
    <AppButton variant="secondary" @click="emit('reports')">
      <FileText :size="17" aria-hidden="true" /> 报告库
    </AppButton>
    <div class="sidebar-section-heading">
      <span>会话</span><span>{{ sessions.length }}</span>
    </div>
    <label v-if="showSearch" class="session-search">
      <Search :size="14" aria-hidden="true" />
      <span class="sr-only">搜索会话</span>
      <input v-model="query" type="search" placeholder="搜索会话标题" />
    </label>
    <nav class="session-sidebar__sessions" aria-label="会话">
      <ul class="session-list">
        <template v-for="group in groups" :key="group.label">
          <li class="session-group-label">{{ group.label }}</li>
          <li v-for="session in group.sessions" :key="session.threadId" class="session-item">
            <button
              type="button"
              class="session-item__select"
              :class="{ 'is-active': session.threadId === activeThreadId }"
              :aria-current="session.threadId === activeThreadId ? 'page' : undefined"
              @click="emit('select', session.threadId)"
            >
              <strong class="session-item__title">{{ session.title }}</strong>
              <span class="session-item__meta">
                <time :datetime="session.updatedAt">{{ formatSessionTime(session.updatedAt) }}</time>
                <StatusBadge
                  :status="unknownStatusIds?.has(session.threadId) ? 'unknown' : session.status"
                />
              </span>
            </button>
            <button
              type="button"
              class="session-item__remove"
              :aria-label="`从列表中移除：${session.title}`"
              title="从列表中移除，刷新后会重新出现"
              @click="emit('removeLocal', session.threadId)"
            >
              <Trash2 :size="13" aria-hidden="true" />
            </button>
          </li>
        </template>
        <li v-if="!sessions.length" class="empty-copy">尚无会话。发出第一条消息后，会话会显示在这里。</li>
        <li v-else-if="!groups.length" class="empty-copy">没有匹配的会话。</li>
      </ul>
    </nav>
    <footer class="sidebar-footer">
      <div class="api-status" title="根据后端 /health/live 的响应判断，不代表模型或数据库可用">
        <Radio :size="15" aria-hidden="true" />
        <span>
          <strong>{{ apiStatusLabel }}</strong>
          <small>仅检测后端连通性</small>
        </span>
      </div>
      <button
        type="button"
        class="icon-button"
        aria-label="打开设置"
        title="打开设置"
        @click="emit('settings')"
      >
        <Settings2 :size="18" />
      </button>
    </footer>
  </aside>
</template>
