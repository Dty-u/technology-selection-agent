<script setup lang="ts">
import { AlertOctagon, ScanSearch } from '@lucide/vue'

import type { ReportData } from '../../api/sessionSchemas'

defineProps<{
  risks: ReportData['risks']
  gaps: ReportData['evidence_gaps']
}>()
</script>

<template>
  <section class="report-card" aria-labelledby="risk-title">
    <div class="report-card__eyebrow">不确定性清单</div>
    <h3 id="risk-title">风险与证据缺口</h3>
    <div class="risk-list">
      <article
        v-for="risk in risks"
        :key="`${risk.candidate_id}-${risk.category}`"
        class="risk-item"
      >
        <AlertOctagon :size="17" aria-hidden="true" />
        <div>
          <div class="risk-item__title">
            <strong>{{ risk.candidate_name }} · {{ risk.category }}</strong>
            <span :data-level="risk.level">{{ risk.level }}</span>
          </div>
          <p>{{ risk.impact }}</p>
          <small>缓解：{{ risk.mitigation }} · {{ risk.basis_label }}</small>
        </div>
      </article>
    </div>
    <div v-if="gaps.length" class="gap-list">
      <h4><ScanSearch :size="16" aria-hidden="true" /> 待补证</h4>
      <ul>
        <li v-for="gap in gaps" :key="gap.description">
          <span>{{ gap.priority }}</span
          >{{ gap.description }}
        </li>
      </ul>
    </div>
  </section>
</template>
