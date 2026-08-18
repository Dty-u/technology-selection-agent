<script setup lang="ts">
import type { ReportData } from '../../api/sessionSchemas'

defineProps<{ audit: ReportData['audit'] }>()
</script>

<template>
  <section class="report-card audit-card" aria-labelledby="audit-title">
    <div class="report-card__eyebrow">审计轨迹</div>
    <h3 id="audit-title">复核与迭代</h3>
    <dl>
      <div>
        <dt>工作流状态</dt>
        <dd>{{ audit.status }}</dd>
      </div>
      <div>
        <dt>审查结论</dt>
        <dd>{{ audit.review_label }}</dd>
      </div>
      <div>
        <dt>证据轮次</dt>
        <dd>{{ audit.evidence_round }}</dd>
      </div>
      <div>
        <dt>审查轮次</dt>
        <dd>{{ audit.review_round }}</dd>
      </div>
    </dl>
    <ul v-if="audit.review_reasons.length">
      <li v-for="reason in audit.review_reasons" :key="reason">{{ reason }}</li>
    </ul>
  </section>
</template>
