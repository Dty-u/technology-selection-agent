<script setup lang="ts">
import { AlertTriangle, CheckCircle2, Gauge } from '@lucide/vue'

import type { ReportData } from '../../api/sessionSchemas'

defineProps<{ decision: ReportData['decision'] }>()
</script>

<template>
  <section class="report-card decision-card" aria-labelledby="decision-title">
    <div class="report-card__eyebrow">
      <Gauge :size="15" aria-hidden="true" />
      决策摘要
    </div>
    <div class="decision-card__headline">
      <div>
        <p class="decision-card__status">{{ decision.status_label }}</p>
        <h3 id="decision-title">{{ decision.primary_candidate_name || '暂不推荐候选' }}</h3>
      </div>
      <div
        class="coverage-dial"
        :style="{ '--coverage': `${decision.evidence_coverage * 360}deg` }"
      >
        <strong>{{ Math.round(decision.evidence_coverage * 100) }}%</strong>
        <span>证据覆盖</span>
      </div>
    </div>
    <p class="decision-card__summary">{{ decision.summary }}</p>
    <div v-if="decision.requires_human_review" class="review-alert" role="status">
      <AlertTriangle :size="17" aria-hidden="true" />
      <span><strong>需要人工复核</strong> — 当前结论仅供项目负责人审查。</span>
    </div>
    <div v-else class="review-pass">
      <CheckCircle2 :size="16" aria-hidden="true" />
      自动审查已通过，仍建议由项目负责人确认。
    </div>
    <dl class="decision-metrics">
      <div>
        <dt>可信度</dt>
        <dd>{{ decision.confidence_label }}</dd>
      </div>
      <div>
        <dt>判断依据</dt>
        <dd>{{ decision.decision_basis }}</dd>
      </div>
    </dl>
  </section>
</template>
