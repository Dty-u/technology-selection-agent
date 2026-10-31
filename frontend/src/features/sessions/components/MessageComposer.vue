<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { FileText, SendHorizontal } from '@lucide/vue'

const props = withDefaults(
  defineProps<{
    disabled?: boolean
    awaitingUser?: boolean
    modelValue?: string
  }>(),
  { disabled: false, awaitingUser: false, modelValue: '' },
)

const emit = defineEmits<{
  submit: [message: string, wantsDocx: boolean]
  'update:modelValue': [value: string]
}>()

const textarea = ref<HTMLTextAreaElement | null>(null)
const message = ref(props.modelValue)
const wantsDocx = ref(false)
const error = ref('')
const count = computed(() => message.value.length)

watch(
  () => props.modelValue,
  (value) => {
    message.value = value
    error.value = ''
  },
)

watch(message, (value) => emit('update:modelValue', value))

const MAX_TEXTAREA_HEIGHT = 220

function fitTextarea() {
  const element = textarea.value
  if (!element) return
  element.style.height = 'auto'
  const contentHeight = element.scrollHeight
  element.style.height = `${Math.min(contentHeight, MAX_TEXTAREA_HEIGHT)}px`
  element.style.overflowY = contentHeight > MAX_TEXTAREA_HEIGHT ? 'auto' : 'hidden'
}

watch(message, fitTextarea, { flush: 'post' })

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

function focus() {
  textarea.value?.focus()
}

function handleKeydown(event: KeyboardEvent) {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault()
    submit()
  }
}

defineExpose({ focus, wantsDocx })
</script>

<template>
  <form class="composer" aria-label="发送消息" @submit.prevent="submit">
    <label class="sr-only" for="agent-message">消息</label>
    <textarea
      id="agent-message"
      ref="textarea"
      v-model="message"
      :disabled="disabled"
      :placeholder="
        awaitingUser ? '集中回答上方问题，按 Enter 继续…' : '描述你的问题或技术选型需求…'
      "
      rows="1"
      @keydown="handleKeydown"
    />
    <div class="composer__footer">
      <label class="composer__format">
        <input v-model="wantsDocx" type="checkbox" :disabled="disabled" />
        <FileText :size="15" aria-hidden="true" />
        需要 DOCX
      </label>
      <div class="composer__actions">
        <span class="composer__shortcut" aria-hidden="true">Enter 发送 · Shift+Enter 换行</span>
        <span :class="['composer__count', { 'is-over': count > 10_000 }]">
          {{ count.toLocaleString() }} / 10,000
        </span>
        <button
          class="composer__send"
          type="submit"
          aria-label="发送消息"
          :disabled="disabled || !message.trim()"
        >
          <span>发送</span>
          <SendHorizontal :size="17" aria-hidden="true" />
        </button>
      </div>
    </div>
    <p v-if="error" class="composer__error" role="alert">{{ error }}</p>
  </form>
</template>
