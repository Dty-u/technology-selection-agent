<script setup lang="ts">
import { X } from '@lucide/vue'

defineProps<{ open: boolean; title: string; side?: 'left' | 'right' }>()
const emit = defineEmits<{ close: [] }>()

function onKeydown(event: KeyboardEvent) {
  if (event.key === 'Escape') emit('close')
}
</script>

<template>
  <Transition name="drawer-fade">
    <div v-if="open" class="drawer-layer">
      <button class="drawer-backdrop" type="button" aria-label="关闭抽屉" @click="emit('close')" />
      <aside
        class="app-drawer"
        :class="`app-drawer--${side ?? 'left'}`"
        role="dialog"
        aria-modal="true"
        :aria-label="title"
        tabindex="-1"
        @keydown="onKeydown"
      >
        <header class="app-drawer__header">
          <strong>{{ title }}</strong>
          <button class="icon-button" type="button" aria-label="关闭" @click="emit('close')">
            <X :size="19" aria-hidden="true" />
          </button>
        </header>
        <div class="app-drawer__body"><slot /></div>
      </aside>
    </div>
  </Transition>
</template>
