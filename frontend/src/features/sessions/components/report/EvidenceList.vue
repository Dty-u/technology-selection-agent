<script setup lang="ts">
import { ExternalLink, LibraryBig } from '@lucide/vue'

import type { ReportData } from '../../api/sessionSchemas'

defineProps<{ evidence: ReportData['evidence'] }>()

function safeUrl(value: string): string | null {
  try {
    const url = new URL(value)
    return ['http:', 'https:'].includes(url.protocol) ? url.toString() : null
  } catch {
    return null
  }
}
</script>

<template>
  <section class="report-card" aria-labelledby="evidence-title">
    <div class="report-card__eyebrow"><LibraryBig :size="15" aria-hidden="true" /> 证据记录</div>
    <h3 id="evidence-title">来源与支持关系</h3>
    <div v-if="evidence.length" class="evidence-list">
      <article v-for="item in evidence" :key="item.evidence_id" class="evidence-item">
        <div class="evidence-item__meta">
          <span>{{ item.source_label }}</span>
          <code>{{ item.evidence_id }}</code>
        </div>
        <h4>{{ item.title }}</h4>
        <p>{{ item.excerpt }}</p>
        <ul v-if="item.supports.length">
          <li
            v-for="support in item.supports"
            :key="`${support.candidate_id}-${support.criterion_id}`"
          >
            {{ support.candidate_name }} / {{ support.criterion_label }} · {{ support.level_label }}
          </li>
        </ul>
        <a
          v-if="safeUrl(item.url)"
          :href="safeUrl(item.url) ?? undefined"
          target="_blank"
          rel="noopener noreferrer"
        >
          查看外部来源 <ExternalLink :size="14" aria-hidden="true" />
        </a>
        <span v-else class="evidence-item__internal">{{ item.display_source }}</span>
      </article>
    </div>
    <p v-else class="empty-note">当前报告没有公开证据记录。</p>
  </section>
</template>
