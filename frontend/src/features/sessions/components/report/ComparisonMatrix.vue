<script setup lang="ts">
import type { ReportData } from '../../api/sessionSchemas'

defineProps<{ comparison: ReportData['comparison'] }>()
</script>

<template>
  <section class="report-card" aria-labelledby="comparison-title">
    <div class="report-card__eyebrow">候选矩阵</div>
    <h3 id="comparison-title">指标对比</h3>
    <div class="comparison-scroll" tabindex="0" aria-label="候选指标对比表，可横向滚动">
      <table class="comparison-table">
        <thead>
          <tr>
            <th scope="col">指标</th>
            <th
              v-for="candidate in comparison.candidates"
              :key="candidate.candidate_id"
              scope="col"
            >
              {{ candidate.name }}
            </th>
            <th scope="col">判断</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="criterion in comparison.criteria" :key="criterion.criterion_id">
            <th scope="row">
              {{ criterion.label }}
              <span>{{ criterion.weight }}</span>
            </th>
            <td v-for="cell in criterion.cells" :key="cell.candidate_id">
              <strong>{{ cell.score ?? '—' }}</strong>
              <small>{{ cell.rationale || '暂无说明' }}</small>
              <span class="confidence-chip">可信度 {{ cell.confidence_label }}</span>
            </td>
            <td>
              {{ criterion.judgment }}
              <small>{{ criterion.evidence_status }}</small>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </section>
</template>
