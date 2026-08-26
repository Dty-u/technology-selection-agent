<script setup lang="ts">
import { ChevronDown, Download, FileText } from '@lucide/vue'
import { computed } from 'vue'

import type { ReportSummary } from '../api/sessionSchemas'

const props = defineProps<{ reports: ReportSummary[] }>()
const emit = defineEmits<{ download: [report: ReportSummary] }>()

const newestFirst = computed(() => [...props.reports].reverse())

function displayTime(value: string): string {
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) return '时间未知'
  return new Intl.DateTimeFormat('zh-CN', {
    month: 'numeric',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  }).format(parsed)
}
</script>

<template>
  <details class="inspector-section report-history" open>
    <summary class="section-heading">
      <FileText :size="16" aria-hidden="true" />
      <h3>选型报告</h3>
      <span>{{ reports.length }}</span>
      <ChevronDown class="section-heading__chevron" :size="15" aria-hidden="true" />
    </summary>
    <div class="inspector-section__body report-history__list">
      <article
        v-for="report in newestFirst"
        :key="report.report_id"
        class="report-history__item"
        data-report-item
      >
        <div class="report-history__copy">
          <strong>{{ report.title }}</strong>
          <span>{{ displayTime(report.created_at) }} · {{ report.format.toUpperCase() }}</span>
        </div>
        <button
          type="button"
          class="report-history__download"
          :aria-label="`下载 ${report.title}`"
          title="下载这份报告"
          @click="emit('download', report)"
        >
          <Download :size="15" aria-hidden="true" />
        </button>
      </article>
    </div>
  </details>
</template>
