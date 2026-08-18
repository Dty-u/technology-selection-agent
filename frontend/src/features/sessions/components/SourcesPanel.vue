<script setup lang="ts">
import { ExternalLink, FileSearch } from '@lucide/vue'

import type { SessionSource } from '../api/sessionSchemas'

defineProps<{ sources: SessionSource[] }>()

function safeUrl(url?: string): string | null {
  if (!url) return null
  try {
    const parsed = new URL(url)
    return parsed.protocol === 'http:' || parsed.protocol === 'https:' ? parsed.href : null
  } catch {
    return null
  }
}
</script>

<template>
  <section class="inspector-section" aria-labelledby="sources-title">
    <div class="section-heading">
      <FileSearch :size="16" aria-hidden="true" />
      <h3 id="sources-title">来源</h3>
      <span>{{ sources.length }}</span>
    </div>
    <div v-if="sources.length" class="source-list">
      <article
        v-for="(source, index) in sources"
        :key="`${source.url}-${index}`"
        class="source-card"
      >
        <div class="source-card__index">{{ String(index + 1).padStart(2, '0') }}</div>
        <div>
          <strong>{{ source.title || source.source || '未命名来源' }}</strong>
          <p>{{ source.content || '该来源未提供摘录。' }}</p>
          <a
            v-if="safeUrl(source.url)"
            :href="safeUrl(source.url) ?? undefined"
            target="_blank"
            rel="noopener noreferrer"
          >
            查看来源 <ExternalLink :size="13" aria-hidden="true" />
          </a>
          <span v-else-if="source.url" class="unsafe-link-note">链接协议不可用</span>
        </div>
      </article>
    </div>
    <p v-else class="empty-copy">本次响应未返回来源。</p>
  </section>
</template>
