<script setup lang="ts">
import { computed, onBeforeUnmount, ref } from 'vue'
import { Bot, Check, CircleAlert, Copy, Pencil, RotateCcw, UserRound } from '@lucide/vue'

import MarkdownContent from '@/shared/components/MarkdownContent.vue'
import type { SessionSource } from '../api/sessionSchemas'
import { formatMessageTime } from '../model/sessionTime'
import type { ChatMessage } from '../model/sessionTypes'
import { resolveCitationSource } from './citationSource'
import { compactEvidenceIndex } from './compactEvidenceIndex'

const props = withDefaults(
  defineProps<{ message: ChatMessage; sources?: SessionSource[]; actionsDisabled?: boolean }>(),
  { sources: () => [], actionsDisabled: false },
)
const emit = defineEmits<{
  retry: [messageId: string]
  edit: [message: ChatMessage]
  focusSource: [sourceIndex: number]
}>()

const visibleContent = computed(() =>
  props.message.role === 'assistant'
    ? compactEvidenceIndex(props.message.content)
    : props.message.content,
)
const canCopy = computed(
  () =>
    props.message.role === 'assistant' &&
    props.message.delivery !== 'streaming' &&
    props.message.content !== '',
)

function resolveCitation(citation: number): number | null {
  return resolveCitationSource(props.message.content, citation, props.sources)
}

const isCitable = computed(() =>
  props.message.role === 'assistant'
    ? (citation: number) => resolveCitation(citation) !== null
    : undefined,
)

function focusCitation(citation: number) {
  const sourceIndex = resolveCitation(citation)
  if (sourceIndex !== null) emit('focusSource', sourceIndex)
}

const COPY_FEEDBACK_MS = 1_500
const copyState = ref<'idle' | 'copied' | 'failed'>('idle')
let copyTimer: number | undefined

async function copyContent() {
  try {
    await navigator.clipboard.writeText(props.message.content)
    copyState.value = 'copied'
  } catch {
    copyState.value = 'failed'
  }
  window.clearTimeout(copyTimer)
  copyTimer = window.setTimeout(() => (copyState.value = 'idle'), COPY_FEEDBACK_MS)
}

onBeforeUnmount(() => window.clearTimeout(copyTimer))
</script>

<template>
  <article
    class="message"
    :class="`message--${message.role}`"
    :aria-busy="message.delivery === 'streaming' || undefined"
  >
    <div class="message__avatar" aria-hidden="true">
      <UserRound v-if="message.role === 'user'" :size="17" />
      <Bot v-else :size="18" />
    </div>
    <div class="message__body">
      <div class="message__meta">
        <strong>{{ message.role === 'user' ? '你' : '晶圆智选' }}</strong>
        <span>{{ formatMessageTime(message.createdAt) }}</span>
      </div>
      <MarkdownContent
        :content="visibleContent"
        :citable="isCitable"
        @citation="focusCitation"
      />
      <span
        v-if="message.delivery === 'streaming'"
        class="message__cursor"
        data-streaming-cursor
        aria-hidden="true"
      />
      <p v-if="message.delivery === 'pending'" class="message__delivery" role="status">正在发送…</p>
      <p v-else-if="message.delivery === 'cancelled'" class="message__delivery" role="status">
        本次请求已退出
      </p>
      <div
        v-else-if="message.delivery === 'failed'"
        class="message__delivery message__delivery--failed"
        role="alert"
      >
        <CircleAlert :size="14" aria-hidden="true" /> 本条消息发送失败
        <span class="message__failure-actions">
          <button
            type="button"
            class="text-button"
            :disabled="actionsDisabled"
            @click="emit('retry', message.id)"
          >
            <RotateCcw :size="13" aria-hidden="true" /> 重试
          </button>
          <button
            type="button"
            class="text-button"
            :disabled="actionsDisabled"
            @click="emit('edit', message)"
          >
            <Pencil :size="13" aria-hidden="true" /> 编辑后重发
          </button>
        </span>
      </div>
      <div v-if="canCopy" class="message__actions">
        <button type="button" class="text-button" @click="copyContent">
          <Check v-if="copyState === 'copied'" :size="13" aria-hidden="true" />
          <Copy v-else :size="13" aria-hidden="true" />
          {{ copyState === 'copied' ? '已复制' : copyState === 'failed' ? '复制失败' : '复制' }}
        </button>
      </div>
    </div>
  </article>
</template>
