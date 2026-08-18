<script setup lang="ts">
import { Download, FlaskConical, TriangleAlert } from '@lucide/vue'

import AppButton from '@/shared/components/AppButton.vue'
import MarkdownContent from '@/shared/components/MarkdownContent.vue'
import RequestNotice from '@/shared/components/RequestNotice.vue'
import StatusBadge from '@/shared/components/StatusBadge.vue'
import type { ReportData, SessionResponse } from '../api/sessionSchemas'
import type { SessionUiStatus } from '../model/sessionTypes'
import ReportPanel from './report/ReportPanel.vue'
import SourcesPanel from './SourcesPanel.vue'

defineProps<{
  response: SessionResponse | null
  reportData: ReportData | null
  status: SessionUiStatus
}>()
const emit = defineEmits<{ download: [] }>()
</script>

<template>
  <aside class="inspector-panel">
    <header class="inspector-header">
      <div>
        <span class="eyebrow">EVIDENCE INSPECTOR</span>
        <h2>证据与结论</h2>
      </div>
      <StatusBadge :status="status" />
    </header>

    <div v-if="!response" class="inspector-empty">
      <FlaskConical :size="26" aria-hidden="true" />
      <strong>等待可检查的响应</strong>
      <p>来源、证据缺口和结构化选型报告会出现在这里。</p>
    </div>

    <div v-else class="inspector-scroll">
      <RequestNotice
        v-for="warning in response.warnings"
        :key="warning"
        tone="warning"
        :message="warning"
      />
      <template v-if="response.query_type === 'technology_selection'">
        <ReportPanel v-if="reportData" :report="reportData" />
        <section v-else class="report-fallback">
          <RequestNotice
            tone="warning"
            title="结构化报告不可用"
            message="响应未满足 report_data V1 契约，已回退显示 Markdown 报告。"
          />
          <MarkdownContent :content="response.report || response.answer" />
        </section>
      </template>
      <SourcesPanel v-else :sources="response.sources" />
      <RequestNotice
        v-if="response.status === 'needs_human_review'"
        tone="warning"
        title="复核边界"
        message="当前结果需要人工复核，本工作台不提供不存在的批准操作。"
      />
      <AppButton v-if="response.report_url" variant="secondary" @click="emit('download')"
        ><Download :size="16" /> 下载 DOCX 报告</AppButton
      >
      <div v-if="response.status === 'error'" class="contract-alert">
        <TriangleAlert :size="16" /> 服务端返回了错误状态。
      </div>
    </div>
  </aside>
</template>
