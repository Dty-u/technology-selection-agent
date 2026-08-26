<script setup lang="ts">
import { computed, onBeforeUnmount, ref } from 'vue'
import { LoaderCircle } from '@lucide/vue'

const props = defineProps<{ canExit: boolean; cancelling?: boolean; startedAt?: string }>()
const emit = defineEmits<{ exit: [] }>()

const mountedAt = Date.now()
const currentTime = ref(mountedAt)
const timer = window.setInterval(() => (currentTime.value = Date.now()), 1_000)
onBeforeUnmount(() => window.clearInterval(timer))

const startedAt = computed(() => {
  const parsed = props.startedAt ? Date.parse(props.startedAt) : Number.NaN
  return Number.isFinite(parsed) ? parsed : mountedAt
})
const elapsedLabel = computed(() => {
  const seconds = Math.floor(Math.max(0, currentTime.value - startedAt.value) / 1_000)
  const minutes = Math.floor(seconds / 60)
  return minutes ? `${minutes} 分 ${seconds % 60} 秒` : `${seconds} 秒`
})
</script>

<template>
  <div class="waiting-state" role="status" aria-live="polite">
    <LoaderCircle class="spin" :size="18" aria-hidden="true" />
    <div>
      <strong>{{ cancelling ? '正在结束本次请求' : '服务端正在处理' }}</strong>
      <p v-if="!cancelling">已等待 {{ elapsedLabel }}，真实阶段会在右侧执行轨迹中更新。</p>
      <p v-else>当前节点结束后将停止后续工作流步骤，并保留已有会话记录。</p>
    </div>
    <button v-if="canExit" type="button" class="text-button" @click="emit('exit')">退出本次请求</button>
  </div>
</template>
