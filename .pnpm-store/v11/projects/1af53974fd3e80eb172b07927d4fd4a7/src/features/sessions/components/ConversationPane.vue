<script setup lang="ts">
import { computed, nextTick, ref, watch } from 'vue'
import { Bot, Sparkles } from '@lucide/vue'

import RequestNotice from '@/shared/components/RequestNotice.vue'
import type { ApiError } from '@/shared/api/errors'
import type { SessionResponse } from '../api/sessionSchemas'
import type { ChatMessage, SessionUiStatus } from '../model/sessionTypes'
import ClarificationCard from './ClarificationCard.vue'
import MessageBubble from './MessageBubble.vue'
import MessageComposer from './MessageComposer.vue'
import WaitingState from './WaitingState.vue'

const props = defineProps<{
  messages: ChatMessage[]
  response: SessionResponse | null
  status: SessionUiStatus
  notice: ApiError | null
  inFlight: boolean
  canExit: boolean
}>()
const emit = defineEmits<{
  submit: [message: string, wantsDocx: boolean]
  exit: []
}>()
const list = ref<HTMLElement | null>(null)
const composer = ref<{ focus: () => void } | null>(null)
const draft = ref('')
const quickStartPrompts = [
  {
    label: '比较 FinFET 与 GAA',
    message:
      '请系统比较 FinFET 与 GAA 的结构与工作原理，并从性能、功耗、制造复杂度、工艺成熟度、成本、设计生态和典型适用场景说明差异；最后总结各自的优势、局限与适用条件。',
    icon: true,
  },
  {
    label: '解释先进封装中的 CoWoS',
    message:
      '请系统解释先进封装中的 CoWoS：包括基本结构、实现原理、主要类型、性能与带宽优势、典型应用场景、与 InFO 和 SoIC 等先进封装方案的区别，以及设计、散热、良率和成本方面的主要挑战。',
    icon: false,
  },
]

watch(
  () => props.messages.length,
  async () => {
    await nextTick()
    list.value?.scrollTo({ top: list.value.scrollHeight, behavior: 'smooth' })
  },
)

const isWaiting = computed(() => ['submitting', 'running', 'cancelling'].includes(props.status))
const waitingStartedAt = computed(() => {
  for (let index = props.messages.length - 1; index >= 0; index -= 1) {
    const message = props.messages[index]
    if (message?.role === 'user' && message.delivery === 'pending') return message.createdAt
  }
  return undefined
})

function submit(message: string, wantsDocx: boolean) {
  emit('submit', message, wantsDocx)
}

async function quickStart(message: string) {
  if (props.inFlight) return
  draft.value = message
  await nextTick()
  composer.value?.focus()
}
</script>

<template>
  <section class="conversation-pane" aria-label="对话">
    <div ref="list" class="conversation-scroll">
      <div v-if="!messages.length" class="conversation-empty">
        <div class="empty-orbit"><Bot :size="30" aria-hidden="true" /></div>
        <p class="eyebrow">READY FOR INQUIRY</p>
        <h1>从一个问题开始。</h1>
        <p>询问半导体知识，或描述应用场景、约束和候选技术以启动技术选型。</p>
        <div class="prompt-hints">
          <button
            v-for="prompt in quickStartPrompts"
            :key="prompt.label"
            class="prompt-hint"
            type="button"
            :disabled="inFlight"
            :data-quick-start="prompt.label"
            @click="quickStart(prompt.message)"
          >
            <Sparkles v-if="prompt.icon" :size="14" aria-hidden="true" />
            {{ prompt.label }}
          </button>
        </div>
      </div>
      <div v-else class="message-list">
        <MessageBubble v-for="message in messages" :key="message.id" :message="message" />
      </div>
      <ClarificationCard
        v-if="response?.status === 'awaiting_user'"
        :questions="response.clarification_questions"
      />
      <RequestNotice
        v-if="status === 'needs_human_review'"
        tone="warning"
        title="需要人工复核"
        message="这是待复核的技术判断，界面不会把它标记为批准结果。"
      />
      <RequestNotice
        v-if="notice"
        :tone="notice.status >= 500 || notice.status === 0 ? 'error' : 'warning'"
        :title="notice.code"
        :message="notice.message"
      />
      <WaitingState
        v-if="isWaiting"
        :can-exit="canExit"
        :cancelling="status === 'cancelling'"
        :started-at="waitingStartedAt"
        @exit="emit('exit')"
      />
    </div>
    <div class="composer-shell">
      <MessageComposer
        ref="composer"
        v-model="draft"
        :disabled="inFlight"
        :awaiting-user="status === 'awaiting_user'"
        @submit="submit"
      />
    </div>
  </section>
</template>
