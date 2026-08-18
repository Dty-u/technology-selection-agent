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
  canStop: boolean
}>()
const emit = defineEmits<{
  submit: [message: string, wantsDocx: boolean]
  stop: []
  sync: []
}>()
const list = ref<HTMLElement | null>(null)

watch(
  () => props.messages.length,
  async () => {
    await nextTick()
    list.value?.scrollTo({ top: list.value.scrollHeight, behavior: 'smooth' })
  },
)

const isWaiting = computed(() => ['submitting', 'running'].includes(props.status))

function submit(message: string, wantsDocx: boolean) {
  emit('submit', message, wantsDocx)
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
          <span><Sparkles :size="14" /> 比较 FinFET 与 GAA</span><span>解释先进封装中的 CoWoS</span>
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
        v-if="isWaiting || status === 'detached'"
        :can-stop="canStop"
        :detached="status === 'detached'"
        @stop="emit('stop')"
        @sync="emit('sync')"
      />
    </div>
    <div class="composer-shell">
      <MessageComposer
        :disabled="inFlight"
        :awaiting-user="status === 'awaiting_user'"
        @submit="submit"
      />
    </div>
  </section>
</template>
