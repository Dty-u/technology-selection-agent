<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { ArrowLeft, Download, ExternalLink, FileText, Search, Trash2 } from '@lucide/vue'

import { errorTitle } from '@/shared/api/errors'
import MarkdownContent from '@/shared/components/MarkdownContent.vue'
import RequestNotice from '@/shared/components/RequestNotice.vue'
import type { ConfirmRequest } from '@/shared/components/ConfirmDialog.vue'
import type { ReportLibraryItem } from '../api/sessionSchemas'
import { saveDownloadedFile } from '../model/saveDownloadedFile'
import { formatMessageTime } from '../model/sessionTime'
import { useReportLibraryStore } from '../store/reportLibraryStore'

const SEARCH_THRESHOLD = 8

const props = defineProps<{
  currentProjectId: string | null
  confirm: (request: ConfirmRequest) => Promise<boolean>
}>()
const emit = defineEmits<{
  openSession: [threadId: string]
  deleted: [report: ReportLibraryItem]
}>()

const store = useReportLibraryStore()
const query = ref('')
const selected = ref<ReportLibraryItem | null>(null)
const content = ref('')
const contentLoading = ref(false)

const showSearch = computed(() => store.reports.length > SEARCH_THRESHOLD)
const visibleReports = computed(() => {
  const keyword = showSearch.value ? query.value.trim().toLowerCase() : ''
  return store.reports.filter((report) =>
    `${report.title} ${report.session_title}`.toLowerCase().includes(keyword),
  )
})

onMounted(() => void store.load())

async function view(report: ReportLibraryItem) {
  selected.value = report
  content.value = ''
  contentLoading.value = true
  const loaded = await store.readContent(report)
  if (selected.value !== report) return
  contentLoading.value = false
  content.value = loaded?.content ?? ''
}

function backToList() {
  selected.value = null
  store.error = null
}

async function download(report: ReportLibraryItem) {
  const file = await store.download(report)
  if (file) saveDownloadedFile(file)
}

async function remove(report: ReportLibraryItem) {
  const confirmed = await props.confirm({
    title: '删除这份报告？',
    message: `“${report.title}”删除后无法恢复。`,
    details: [
      '会删除：报告库里的这条记录和已导出的文件。',
      '不会删除：所在会话，以及对话里已经显示的回答文字。',
    ],
    confirmLabel: '删除报告',
  })
  if (!confirmed || !(await store.remove(report))) return
  if (selected.value === report) selected.value = null
  emit('deleted', report)
}
</script>

<template>
  <div class="report-library">
    <RequestNotice
      v-if="store.error"
      tone="error"
      :title="errorTitle(store.error.code)"
      :message="store.error.message"
    />

    <article v-if="selected" class="report-library__detail">
      <div class="report-library__detail-bar">
        <button type="button" class="report-library__back" @click="backToList">
          <ArrowLeft :size="15" aria-hidden="true" /> 全部报告
        </button>
        <span class="report-library__actions">
          <button
            type="button"
            class="icon-button"
            aria-label="下载这份报告"
            title="下载这份报告"
            @click="download(selected)"
          >
            <Download :size="16" aria-hidden="true" />
          </button>
          <button
            type="button"
            class="icon-button danger-action"
            aria-label="删除这份报告"
            title="删除这份报告"
            @click="remove(selected)"
          >
            <Trash2 :size="16" aria-hidden="true" />
          </button>
        </span>
      </div>
      <h3>{{ selected.title }}</h3>
      <p class="report-library__meta">
        {{ formatMessageTime(selected.created_at) }} · {{ selected.format.toUpperCase() }} · 来自
        {{ selected.session_title }}
      </p>
      <p v-if="contentLoading" class="empty-copy">正在读取报告…</p>
      <MarkdownContent v-else-if="content" class="report-library__content" :content="content" />
    </article>

    <template v-else>
      <label v-if="showSearch" class="session-search report-library__search">
        <Search :size="14" aria-hidden="true" />
        <span class="sr-only">搜索报告</span>
        <input v-model="query" type="search" placeholder="搜索报告或会话标题" />
      </label>
      <p v-if="store.loading && !store.loaded" class="empty-copy">正在读取报告…</p>
      <p v-else-if="store.loaded && !store.reports.length" class="empty-copy">
        还没有报告。完成一次技术选型后，报告会出现在这里。
      </p>
      <p v-else-if="store.loaded && !visibleReports.length" class="empty-copy">没有匹配的报告。</p>
      <ul v-else class="report-library__list">
        <li
          v-for="report in visibleReports"
          :key="`${report.thread_id}/${report.report_id}`"
          class="report-library__item"
          data-report-item
        >
          <button type="button" class="report-library__open" @click="view(report)">
            <FileText :size="16" aria-hidden="true" />
            <span class="report-library__copy">
              <strong>{{ report.title }}</strong>
              <small>
                {{ formatMessageTime(report.created_at) }} · {{ report.format.toUpperCase() }}
              </small>
              <small class="report-library__origin">
                {{ report.session_title }}
                <template v-if="report.project_id !== currentProjectId">
                  · 项目 {{ report.project_id }}
                </template>
              </small>
            </span>
          </button>
          <span class="report-library__actions">
            <button
              v-if="report.project_id === currentProjectId"
              type="button"
              class="icon-button"
              :aria-label="`打开所在会话：${report.title}`"
              title="打开所在会话"
              @click="emit('openSession', report.thread_id)"
            >
              <ExternalLink :size="15" aria-hidden="true" />
            </button>
            <button
              type="button"
              class="icon-button"
              :aria-label="`下载 ${report.title}`"
              title="下载这份报告"
              @click="download(report)"
            >
              <Download :size="15" aria-hidden="true" />
            </button>
            <button
              type="button"
              class="icon-button danger-action"
              :aria-label="`删除 ${report.title}`"
              title="删除这份报告"
              @click="remove(report)"
            >
              <Trash2 :size="15" aria-hidden="true" />
            </button>
          </span>
        </li>
      </ul>
    </template>
  </div>
</template>
