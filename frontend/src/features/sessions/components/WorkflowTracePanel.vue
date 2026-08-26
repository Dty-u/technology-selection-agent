<script setup lang="ts">
import { computed } from 'vue'
import { Check, ChevronDown, CircleAlert, LoaderCircle, Pause, TerminalSquare } from '@lucide/vue'

import type { WorkflowProgress } from '../api/sessionSchemas'

const props = defineProps<{ progress?: WorkflowProgress }>()

const events = computed(() => props.progress?.events.slice(-40) ?? [])

const statusLabels = {
  running: '执行中',
  completed: '完成',
  paused: '等待输入',
  error: '失败',
} as const

function durationLabel(durationMs: number | null): string {
  if (durationMs === null) return ''
  if (durationMs < 1_000) return `${durationMs} ms`
  return `${(durationMs / 1_000).toFixed(1)} s`
}
</script>

<template>
  <details class="inspector-section workflow-trace" aria-labelledby="workflow-trace-title" open>
    <summary class="section-heading">
      <TerminalSquare :size="15" aria-hidden="true" />
      <h3 id="workflow-trace-title">执行轨迹</h3>
      <span>{{ events.length }}</span>
      <ChevronDown class="section-heading__chevron" :size="15" aria-hidden="true" />
    </summary>

    <div class="inspector-section__body">
      <div v-if="progress?.active && progress.current_stage" class="workflow-current" role="status">
        <LoaderCircle class="spin" :size="15" aria-hidden="true" />
        <div>
          <small>当前阶段 · {{ progress.current_stage.scope }}</small>
          <strong>{{ progress.current_stage.label }}</strong>
        </div>
      </div>

      <ol v-if="events.length" class="workflow-event-list">
        <li
          v-for="(event, index) in events"
          :key="`${event.stage_id}-${event.started_at}-${index}`"
          class="workflow-event"
          :data-status="event.status"
        >
          <span class="workflow-event__icon" aria-hidden="true">
            <LoaderCircle v-if="event.status === 'running'" class="spin" :size="12" />
            <Check v-else-if="event.status === 'completed'" :size="12" />
            <Pause v-else-if="event.status === 'paused'" :size="12" />
            <CircleAlert v-else :size="12" />
          </span>
          <div>
            <strong>{{ event.label }}</strong>
            <small>{{ event.scope }} · {{ event.stage_id }}</small>
          </div>
          <span class="workflow-event__meta">
            {{ durationLabel(event.duration_ms) || statusLabels[event.status] }}
          </span>
        </li>
      </ol>
      <p v-else class="empty-copy">提交请求后，这里会显示后端实际执行的工作流阶段。</p>
    </div>
  </details>
</template>
