<script setup lang="ts">
import { onMounted, reactive, ref } from 'vue'
import { Save } from '@lucide/vue'

import { createSessionApi } from '@/features/sessions/api/sessionApi'
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
const projectIds = ref<string[]>([])

// Suggestions only: a new Project ID typed here is created for this user on save.
async function loadProjects() {
  const userId = form.userId.trim()
  projectIds.value = []
  if (!userId || !form.apiBaseUrl) return
  try {
    const projects = await createSessionApi({
      apiBaseUrl: form.apiBaseUrl,
      userId,
      projectId: form.projectId,
    }).listProjects()
    if (form.userId.trim() === userId) {
      projectIds.value = projects.map((project) => project.project_id)
    }
  } catch {
    projectIds.value = []
  }
}

onMounted(loadProjects)

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
    <p>
      会话按 User ID 和 Project ID 保存在后端。每个 User ID 有自己的项目，切换后显示对应的会话。
    </p>
    <RequestNotice v-if="error" tone="error" :message="error" />
    <form class="settings-form" @submit.prevent="submit">
      <label for="drawer-api-url">API Base URL</label>
      <input id="drawer-api-url" v-model="form.apiBaseUrl" type="url" required />
      <label for="drawer-user-id">User ID</label>
      <input id="drawer-user-id" v-model="form.userId" required @change="loadProjects" />
      <label for="drawer-project-id">Project ID</label>
      <input
        id="drawer-project-id"
        v-model="form.projectId"
        list="drawer-project-options"
        placeholder="选择已有项目，或输入新的 Project ID"
        required
      />
      <datalist id="drawer-project-options">
        <option v-for="projectId in projectIds" :key="projectId" :value="projectId" />
      </datalist>
      <AppButton type="submit"><Save :size="16" aria-hidden="true" /> 保存并切换</AppButton>
    </form>
  </div>
</template>
