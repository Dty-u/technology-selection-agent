<script setup lang="ts">
import type { ReportData } from '../../api/sessionSchemas'
import AuditPanel from './AuditPanel.vue'
import ComparisonMatrix from './ComparisonMatrix.vue'
import DecisionSummary from './DecisionSummary.vue'
import EvidenceList from './EvidenceList.vue'
import RiskGapPanel from './RiskGapPanel.vue'

defineProps<{ report: ReportData }>()
</script>

<template>
  <div class="report-panel">
    <DecisionSummary :decision="report.decision" />
    <section class="report-card requirements-card" aria-labelledby="requirements-title">
      <div class="report-card__eyebrow">需求基线</div>
      <h3 id="requirements-title">{{ report.requirements.business_goal || '未命名技术选型' }}</h3>
      <p>{{ report.requirements.application_scenario || '未提供应用场景' }}</p>
      <div class="criterion-chips">
        <span
          v-for="criterion in report.requirements.active_criteria"
          :key="criterion.criterion_id"
        >
          {{ criterion.label }} {{ Math.round(criterion.weight * 100) }}%
        </span>
      </div>
    </section>
    <ComparisonMatrix :comparison="report.comparison" />
    <RiskGapPanel :risks="report.risks" :gaps="report.evidence_gaps" />
    <EvidenceList :evidence="report.evidence" />
    <AuditPanel :audit="report.audit" />
  </div>
</template>
