<script setup lang="ts">
import { reactive, ref } from 'vue'
import { Save } from '@lucide/vue'

import AppButton from '@/shared/components/AppButton.vue'
import RequestNotice from '@/shared/components/RequestNotice.vue'
import { appSettingsSchema } from './settingsSchema'
import { useSettingsStore } from './settingsStore'

const emit = defineEmits<{ saved: [] }>()
const store = useSettingsStore()
const form = reactive({
  apiBaseUrl: store.settings?.apiBaseUrl ?? '',
  userId: store.settings?.userId ?? '',
  projectId: store.settings?.projectId ?? '',
})
const error = ref('')

function submit() {
  const result = appSettingsSchema.safeParse(form)
  if (!result.success) {
    error.value = result.error.issues[0]?.message ?? '请检查设置。'
    return
  }
  store.save(result.data)
  emit('saved')
}
</script>

<template>
  <div class="settings-drawer-content">
    <p>更改身份或项目后会切换到独立的本机会话空间，不会迁移现有记录。</p>
    <RequestNotice v-if="error" tone="error" :message="error" />
    <form class="settings-form" @submit.prevent="submit">
      <label for="drawer-api-url">API Base URL</label>
      <input id="drawer-api-url" v-model="form.apiBaseUrl" type="url" required />
      <label for="drawer-user-id">User ID</label>
      <input id="drawer-user-id" v-model="form.userId" required />
      <label for="drawer-project-id">Project ID</label>
      <input id="drawer-project-id" v-model="form.projectId" required />
      <AppButton type="submit"><Save :size="16" aria-hidden="true" /> 保存并切换</AppButton>
    </form>
  </div>
</template>
