<script setup lang="ts">
import { nextTick, onBeforeUnmount, ref, watch } from 'vue'
import { X } from '@lucide/vue'

const props = defineProps<{ open: boolean; title: string; side?: 'left' | 'right' }>()
const emit = defineEmits<{ close: [] }>()
const closeButton = ref<HTMLButtonElement | null>(null)
let previouslyFocused: HTMLElement | null = null

watch(
  () => props.open,
  async (open, wasOpen) => {
    if (open) {
      previouslyFocused =
        document.activeElement instanceof HTMLElement ? document.activeElement : null
      await nextTick()
      closeButton.value?.focus()
      return
    }
    if (wasOpen) previouslyFocused?.focus()
    previouslyFocused = null
  },
)

onBeforeUnmount(() => previouslyFocused?.focus())

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
          <button
            ref="closeButton"
            class="icon-button"
            type="button"
            aria-label="关闭"
            @click="emit('close')"
          >
            <X :size="19" aria-hidden="true" />
          </button>
        </header>
        <div class="app-drawer__body"><slot /></div>
      </aside>
    </div>
  </Transition>
</template>
