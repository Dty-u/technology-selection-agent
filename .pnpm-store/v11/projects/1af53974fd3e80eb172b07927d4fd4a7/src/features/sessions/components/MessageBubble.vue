<script setup lang="ts">
import { computed } from 'vue'
import { Bot, CircleAlert, UserRound } from '@lucide/vue'

import MarkdownContent from '@/shared/components/MarkdownContent.vue'
import type { ChatMessage } from '../model/sessionTypes'
import { compactEvidenceIndex } from './compactEvidenceIndex'

const props = defineProps<{ message: ChatMessage }>()
const visibleContent = computed(() =>
  props.message.role === 'assistant'
    ? compactEvidenceIndex(props.message.content)
    : props.message.content,
)
</script>

<template>
  <article class="message" :class="`message--${message.role}`">
    <div class="message__avatar" aria-hidden="true">
      <UserRound v-if="message.role === 'user'" :size="17" />
      <Bot v-else :size="18" />
    </div>
    <div class="message__body">
      <div class="message__meta">
        <strong>{{ message.role === 'user' ? '你' : '晶圆智选' }}</strong>
        <span>{{
          new Date(message.createdAt).toLocaleTimeString('zh-CN', {
            hour: '2-digit',
            minute: '2-digit',
          })
        }}</span>
      </div>
      <MarkdownContent :content="visibleContent" />
      <p v-if="message.delivery === 'pending'" class="message__delivery" role="status">正在发送…</p>
      <p v-else-if="message.delivery === 'cancelled'" class="message__delivery" role="status">
        本次请求已退出
      </p>
      <p
        v-else-if="message.delivery === 'failed'"
        class="message__delivery message__delivery--failed"
        role="alert"
      >
        <CircleAlert :size="14" aria-hidden="true" /> 本条消息发送失败
      </p>
    </div>
  </article>
</template>
