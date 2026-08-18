<script setup lang="ts">
import { Bot, CircleAlert, UserRound } from '@lucide/vue'

import MarkdownContent from '@/shared/components/MarkdownContent.vue'
import type { ChatMessage } from '../model/sessionTypes'

defineProps<{ message: ChatMessage }>()
</script>

<template>
  <article class="message" :class="`message--${message.role}`">
    <div class="message__avatar" aria-hidden="true">
      <UserRound v-if="message.role === 'user'" :size="17" />
      <Bot v-else :size="18" />
    </div>
    <div class="message__body">
      <div class="message__meta">
        <strong>{{ message.role === 'user' ? '你' : 'Rag Agent' }}</strong>
        <span>{{
          new Date(message.createdAt).toLocaleTimeString('zh-CN', {
            hour: '2-digit',
            minute: '2-digit',
          })
        }}</span>
      </div>
      <MarkdownContent :content="message.content" />
      <p v-if="message.delivery === 'pending'" class="message__delivery">正在发送…</p>
      <p
        v-else-if="message.delivery === 'failed'"
        class="message__delivery message__delivery--failed"
      >
        <CircleAlert :size="14" aria-hidden="true" /> 本条消息发送失败
      </p>
    </div>
  </article>
</template>
