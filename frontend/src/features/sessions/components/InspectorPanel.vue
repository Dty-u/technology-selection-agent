<script setup lang="ts">
import { Download, FlaskConical, TriangleAlert } from '@lucide/vue'

import AppButton from '@/shared/components/AppButton.vue'
import MarkdownContent from '@/shared/components/MarkdownContent.vue'
import RequestNotice from '@/shared/components/RequestNotice.vue'
import StatusBadge from '@/shared/components/StatusBadge.vue'
import type { ReportData, ReportSummary, SessionResponse } from '../api/sessionSchemas'
import type { SessionUiStatus } from '../model/sessionTypes'
import ReportPanel from './report/ReportPanel.vue'
import ReportHistoryPanel from './ReportHistoryPanel.vue'
import SourcesPanel from './SourcesPanel.vue'
import WorkflowTracePanel from './WorkflowTracePanel.vue'

defineProps<{
  response: SessionResponse | null
  reportData: ReportData | null
  status: SessionUiStatus
  sourceFocus?: { index: number; key: number } | null
}>()
const emit = defineEmits<{ download: [report?: ReportSummary] }>()
</script>

<template>
  <aside class="inspector-panel">
    <header class="inspector-header">
      <div>
        <span class="eyebrow">运行详情</span>
        <h2>执行、证据与结论</h2>
      </div>
      <StatusBadge :status="status" />
    </header>

    <div v-if="!response" class="inspector-empty">
      <FlaskConical :size="26" aria-hidden="true" />
      <strong>等待可检查的响应</strong>
      <p>真实执行轨迹、来源和结构化选型报告会出现在这里。</p>
    </div>

    <div v-else class="inspector-scroll">
      <WorkflowTracePanel :progress="response.progress" />
      <ReportHistoryPanel
        v-if="response.reports.length"
        :reports="response.reports"
        @download="emit('download', $event)"
      />
      <SourcesPanel v-if="response.progress?.active" :sources="response.sources" pending />
      <template v-else>
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
              message="报告数据格式不符合预期，已改为显示 Markdown 版本。"
            />
            <MarkdownContent :content="response.report || response.answer" />
          </section>
        </template>
        <SourcesPanel v-else :sources="response.sources" :focus="sourceFocus" />
        <RequestNotice
          v-if="response.status === 'needs_human_review'"
          tone="warning"
          title="复核边界"
          message="当前结果需要人工复核，本工作台不提供不存在的批准操作。"
        />
        <AppButton
          v-if="response.report_url && !response.reports.length"
          variant="secondary"
          @click="emit('download')"
          ><Download :size="16" /> 下载 DOCX 报告</AppButton
        >
        <div v-if="response.status === 'error'" class="contract-alert">
          <TriangleAlert :size="16" /> 服务端返回了错误状态。
        </div>
      </template>
    </div>
  </aside>
</template>
