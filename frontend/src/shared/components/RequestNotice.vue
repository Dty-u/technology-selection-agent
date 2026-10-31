<script setup lang="ts">
import { AlertCircle, Info, TriangleAlert } from '@lucide/vue'

withDefaults(
  defineProps<{
    tone?: 'info' | 'warning' | 'error'
    title?: string
    message: string
    detail?: string
  }>(),
  { tone: 'info', title: '', detail: '' },
)
</script>

<template>
  <div class="request-notice" :class="`request-notice--${tone}`" role="status">
    <AlertCircle v-if="tone === 'error'" :size="18" aria-hidden="true" />
    <TriangleAlert v-else-if="tone === 'warning'" :size="18" aria-hidden="true" />
    <Info v-else :size="18" aria-hidden="true" />
    <div>
      <strong v-if="title">{{ title }}</strong>
      <p>{{ message }}</p>
      <small v-if="detail" class="request-notice__detail">{{ detail }}</small>
    </div>
  </div>
</template>
