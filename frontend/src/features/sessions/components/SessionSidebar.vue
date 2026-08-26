<script setup lang="ts">
import { CircuitBoard, MoreHorizontal, Plus, Radio, Settings2, Trash2 } from '@lucide/vue'

import AppButton from '@/shared/components/AppButton.vue'
import StatusBadge from '@/shared/components/StatusBadge.vue'
import type { LocalSession } from '../model/sessionTypes'

defineProps<{
  sessions: LocalSession[]
  activeThreadId: string | null
  apiReachable: boolean | null
}>()
const emit = defineEmits<{
  create: []
  select: [threadId: string]
  settings: []
  removeLocal: [threadId: string]
}>()
</script>

<template>
  <aside class="session-sidebar">
    <header class="workspace-brand">
      <div class="brand-mark"><CircuitBoard :size="20" aria-hidden="true" /></div>
      <div><strong>晶圆智选</strong><span>DECISION LAB</span></div>
    </header>
    <AppButton @click="emit('create')"><Plus :size="17" aria-hidden="true" /> 新建会话</AppButton>
    <div class="sidebar-section-heading">
      <span>本机会话</span><span>{{ sessions.length }}/20</span>
    </div>
    <nav class="session-sidebar__sessions" aria-label="本机会话">
      <ul class="session-list">
        <li v-for="session in sessions" :key="session.threadId" class="session-item">
          <button
            type="button"
            class="session-item__select"
            :class="{ 'is-active': session.threadId === activeThreadId }"
            :aria-current="session.threadId === activeThreadId ? 'page' : undefined"
            @click="emit('select', session.threadId)"
          >
            <span class="session-item__line"
              ><strong>{{ session.title }}</strong
              ><MoreHorizontal :size="15" aria-hidden="true"
            /></span>
            <span>{{
              new Date(session.updatedAt).toLocaleDateString('zh-CN', {
                month: 'short',
                day: 'numeric',
              })
            }}</span>
            <StatusBadge :status="session.status" />
          </button>
          <button
            type="button"
            class="session-item__remove"
            :aria-label="`仅移除本机记录：${session.title}`"
            title="仅移除本机记录"
            @click="emit('removeLocal', session.threadId)"
          >
            <Trash2 :size="13" aria-hidden="true" />
          </button>
        </li>
        <li v-if="!sessions.length" class="empty-copy">
          尚无本机会话。发出第一条消息后，thread_id 会保存在这里。
        </li>
      </ul>
    </nav>
    <footer class="sidebar-footer">
      <div class="api-status">
        <Radio :size="15" aria-hidden="true" />
        <span
          ><strong
            >API
            {{
              apiReachable === true ? '可达' : apiReachable === false ? '不可达' : '未检测'
            }}</strong
          ><small>仅代表 /health/live</small></span
        >
      </div>
      <button type="button" class="icon-button" aria-label="打开设置" @click="emit('settings')">
        <Settings2 :size="18" />
      </button>
    </footer>
  </aside>
</template>
