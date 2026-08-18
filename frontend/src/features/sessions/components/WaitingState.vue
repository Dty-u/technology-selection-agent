<script setup lang="ts">
import { computed, onBeforeUnmount, ref } from 'vue'
import { LoaderCircle, PauseCircle, RefreshCw } from '@lucide/vue'

defineProps<{ canStop: boolean; detached?: boolean }>()
const emit = defineEmits<{ stop: []; sync: [] }>()

const startedAt = Date.now()
const elapsed = ref(0)
const timer = window.setInterval(() => (elapsed.value = Date.now() - startedAt), 1_000)
onBeforeUnmount(() => window.clearInterval(timer))

const elapsedLabel = computed(() => {
  const seconds = Math.floor(elapsed.value / 1_000)
  const minutes = Math.floor(seconds / 60)
  return minutes ? `${minutes} 分 ${seconds % 60} 秒` : `${seconds} 秒`
})
</script>

<template>
  <div class="waiting-state" role="status" aria-live="polite">
    <LoaderCircle v-if="!detached" class="spin" :size="18" aria-hidden="true" />
    <PauseCircle v-else :size="18" aria-hidden="true" />
    <div>
      <strong>{{ detached ? '浏览器已停止等待' : '服务端正在处理' }}</strong>
      <p v-if="!detached">已等待 {{ elapsedLabel }}，这不是工作流节点进度。</p>
      <p v-else>服务端任务可能仍在运行，请稍后同步可验证状态。</p>
    </div>
    <button v-if="canStop" type="button" class="text-button" @click="emit('stop')">停止等待</button>
    <button v-if="detached" type="button" class="text-button" @click="emit('sync')">
      <RefreshCw :size="14" aria-hidden="true" /> 同步最新状态
    </button>
  </div>
</template>
