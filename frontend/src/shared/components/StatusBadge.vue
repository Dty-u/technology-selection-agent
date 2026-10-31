<script setup lang="ts">
import type { SessionUiStatus } from '@/features/sessions/model/sessionTypes'

defineProps<{ status: SessionUiStatus }>()

const labels: Record<SessionUiStatus, string> = {
  idle: '待命',
  submitting: '正在提交',
  running: '处理中',
  cancelling: '正在退出',
  awaiting_user: '等待补充',
  completed: '已完成',
  needs_human_review: '需要复核',
  cancelled: '已退出',
  error: '异常',
  unknown: '状态未知',
}
</script>

<template>
  <span class="status-badge" :class="`status-badge--${status}`">
    <span class="status-badge__dot" aria-hidden="true" />
    {{ labels[status] }}
  </span>
</template>
