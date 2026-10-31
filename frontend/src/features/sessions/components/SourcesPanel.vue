<script setup lang="ts">
import { nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { ChevronDown, ExternalLink, FileSearch, FileText } from '@lucide/vue'

import type { SessionSource } from '../api/sessionSchemas'
import { isLocalSource, sourceLocation, sourceTitle } from './citationSource'

const props = withDefaults(
  defineProps<{
    sources: SessionSource[]
    pending?: boolean
    focus?: { index: number; key: number } | null
  }>(),
  { pending: false, focus: null },
)

const FOCUS_HIGHLIGHT_MS = 2_000
const panel = ref<HTMLDetailsElement | null>(null)
const highlightedIndex = ref<number | null>(null)
let highlightTimer: number | undefined

function safeUrl(url?: string): string | null {
  if (!url) return null
  try {
    const parsed = new URL(url)
    return parsed.protocol === 'http:' || parsed.protocol === 'https:' ? parsed.href : null
  } catch {
    return null
  }
}

async function revealFocus() {
  const focus = props.focus
  if (!focus || !panel.value) return
  panel.value.open = true
  highlightedIndex.value = focus.index
  await nextTick()
  panel.value
    .querySelector(`[data-source-index="${focus.index}"]`)
    ?.scrollIntoView?.({ block: 'center', behavior: 'smooth' })
  window.clearTimeout(highlightTimer)
  highlightTimer = window.setTimeout(() => (highlightedIndex.value = null), FOCUS_HIGHLIGHT_MS)
}

watch(() => props.focus, revealFocus, { flush: 'post' })
// The mobile drawer mounts this panel only after a citation has already been chosen.
onMounted(revealFocus)
onBeforeUnmount(() => window.clearTimeout(highlightTimer))
</script>

<template>
  <details ref="panel" class="inspector-section" aria-labelledby="sources-title" open>
    <summary class="section-heading">
      <FileSearch :size="16" aria-hidden="true" />
      <h3 id="sources-title">来源</h3>
      <span>{{ sources.length }}</span>
      <ChevronDown class="section-heading__chevron" :size="15" aria-hidden="true" />
    </summary>
    <div class="inspector-section__body">
      <div v-if="sources.length" class="source-list">
        <article
          v-for="(source, index) in sources"
          :key="`${source.url}-${index}`"
          class="source-card"
          :class="{ 'is-focused': highlightedIndex === index }"
          :data-source-index="index"
        >
          <div class="source-card__index">{{ String(index + 1).padStart(2, '0') }}</div>
          <div>
            <strong :title="sourceLocation(source) || undefined">{{ sourceTitle(source) }}</strong>
            <p v-if="source.content">{{ source.content }}</p>
            <a
              v-if="safeUrl(source.url)"
              :href="safeUrl(source.url) ?? undefined"
              target="_blank"
              rel="noopener noreferrer"
            >
              查看来源 <ExternalLink :size="13" aria-hidden="true" />
            </a>
            <span v-else-if="isLocalSource(source)" class="source-card__local">
              <FileText :size="13" aria-hidden="true" /> 本地文档
            </span>
            <span v-else-if="source.url" class="unsafe-link-note">链接不可用</span>
          </div>
        </article>
      </div>
      <p v-else class="empty-copy">
        {{ pending ? '正在等待工作流返回来源…' : '本次响应未返回来源。' }}
      </p>
    </div>
  </details>
</template>
