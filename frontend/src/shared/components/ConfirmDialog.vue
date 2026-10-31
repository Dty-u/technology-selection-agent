<script setup lang="ts">
import { nextTick, onBeforeUnmount, ref, watch } from 'vue'

export type ConfirmRequest = {
  title: string
  message: string
  details?: string[]
  confirmLabel: string
}

const props = defineProps<{ request: ConfirmRequest | null }>()
const emit = defineEmits<{ settle: [confirmed: boolean] }>()

const cancelButton = ref<HTMLButtonElement | null>(null)
const confirmButton = ref<HTMLButtonElement | null>(null)
let previouslyFocused: HTMLElement | null = null

watch(
  () => props.request,
  async (request, previous) => {
    if (request && !previous) {
      previouslyFocused =
        document.activeElement instanceof HTMLElement ? document.activeElement : null
      await nextTick()
      cancelButton.value?.focus()
      return
    }
    if (!request && previous) {
      previouslyFocused?.focus()
      previouslyFocused = null
    }
  },
)

onBeforeUnmount(() => previouslyFocused?.focus())

function onKeydown(event: KeyboardEvent) {
  if (event.key === 'Escape') {
    emit('settle', false)
    return
  }
  if (event.key !== 'Tab') return
  event.preventDefault()
  const next =
    document.activeElement === cancelButton.value ? confirmButton.value : cancelButton.value
  next?.focus()
}
</script>

<template>
  <Transition name="drawer-fade">
    <div v-if="request" class="confirm-layer">
      <button
        class="drawer-backdrop"
        type="button"
        aria-label="取消"
        tabindex="-1"
        @click="emit('settle', false)"
      />
      <div
        class="confirm-dialog"
        role="alertdialog"
        aria-modal="true"
        aria-labelledby="confirm-dialog-title"
        aria-describedby="confirm-dialog-message"
        @keydown="onKeydown"
      >
        <h2 id="confirm-dialog-title">{{ request.title }}</h2>
        <p id="confirm-dialog-message">{{ request.message }}</p>
        <ul v-if="request.details?.length">
          <li v-for="detail in request.details" :key="detail">{{ detail }}</li>
        </ul>
        <div class="confirm-dialog__actions">
          <button
            ref="cancelButton"
            type="button"
            class="confirm-dialog__cancel"
            @click="emit('settle', false)"
          >
            取消
          </button>
          <button
            ref="confirmButton"
            type="button"
            class="confirm-dialog__confirm"
            @click="emit('settle', true)"
          >
            {{ request.confirmLabel }}
          </button>
        </div>
      </div>
    </div>
  </Transition>
</template>
