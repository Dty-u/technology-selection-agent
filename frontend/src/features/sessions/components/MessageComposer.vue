<script setup lang="ts">
import { computed, ref } from 'vue'
import { FileText, SendHorizontal } from '@lucide/vue'

const props = withDefaults(
  defineProps<{
    disabled?: boolean
    awaitingUser?: boolean
  }>(),
  { disabled: false, awaitingUser: false },
)

const emit = defineEmits<{
  submit: [message: string, wantsDocx: boolean]
}>()

const message = ref('')
const wantsDocx = ref(false)
const error = ref('')
const count = computed(() => message.value.length)

function submit() {
  const normalized = message.value.trim()
  if (!normalized) {
    error.value = '请输入问题或补充信息。'
    return
  }
  if (message.value.length > 10_000) {
    error.value = '消息不能超过 10,000 个字符。'
    return
  }
  if (props.disabled) return
  error.value = ''
  emit('submit', normalized, wantsDocx.value)
  message.value = ''
}

function handleKeydown(event: KeyboardEvent) {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault()
    submit()
  }
}
</script>

<template>
  <form class="composer" aria-label="发送消息" @submit.prevent="submit">
    <label class="sr-only" for="agent-message">消息</label>
    <textarea
      id="agent-message"
      v-model="message"
      :disabled="disabled"
      :placeholder="
        awaitingUser ? '集中回答上方问题，按 Enter 继续…' : '描述你的问题或技术选型需求…'
      "
      rows="3"
      @keydown="handleKeydown"
    />
    <div class="composer__footer">
      <label class="composer__format">
        <input v-model="wantsDocx" type="checkbox" :disabled="disabled" />
        <FileText :size="15" aria-hidden="true" />
        需要 DOCX
      </label>
      <div class="composer__actions">
        <span :class="['composer__count', { 'is-over': count > 10_000 }]">
          {{ count.toLocaleString() }} / 10,000
        </span>
        <button class="composer__send" type="submit" :disabled="disabled || !message.trim()">
          <span>发送</span>
          <SendHorizontal :size="17" aria-hidden="true" />
        </button>
      </div>
    </div>
    <p v-if="error" class="composer__error" role="alert">{{ error }}</p>
  </form>
</template>
