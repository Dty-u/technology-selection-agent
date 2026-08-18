<script setup lang="ts">
import { HelpCircle } from '@lucide/vue'

defineProps<{ questions: string[] }>()
</script>

<template>
  <section class="clarification-card" aria-labelledby="clarification-title">
    <div class="clarification-card__icon"><HelpCircle :size="20" aria-hidden="true" /></div>
    <div>
      <p class="eyebrow">需要你的输入</p>
      <h2 id="clarification-title">补充信息后继续同一会话</h2>
      <ol>
        <li v-for="question in questions" :key="question">{{ question }}</li>
      </ol>
      <p>请在下方一次性回答；服务端将从当前 checkpoint 恢复。</p>
    </div>
  </section>
</template>
