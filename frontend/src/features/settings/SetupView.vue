<script setup lang="ts">
import { computed, reactive, ref } from 'vue'
import { ArrowRight, CircuitBoard, Database, ShieldCheck } from '@lucide/vue'
import { useRouter } from 'vue-router'

import AppButton from '@/shared/components/AppButton.vue'
import RequestNotice from '@/shared/components/RequestNotice.vue'
import { appSettingsSchema } from './settingsSchema'
import { useSettingsStore } from './settingsStore'

const router = useRouter()
const store = useSettingsStore()
store.load()

const form = reactive({
  apiBaseUrl:
    store.settings?.apiBaseUrl ?? import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000',
  userId: store.settings?.userId ?? '',
  projectId: store.settings?.projectId ?? '',
})
const error = ref('')
const hasExistingSettings = computed(() => store.isConfigured)

function submit() {
  const result = appSettingsSchema.safeParse(form)
  if (!result.success) {
    error.value = result.error.issues[0]?.message ?? '请检查工作区设置。'
    return
  }
  error.value = ''
  store.save(result.data)
  void router.push('/workspace')
}
</script>

<template>
  <main class="setup-view">
    <section class="setup-intro" aria-labelledby="setup-title">
      <div class="brand-mark brand-mark--large"><CircuitBoard :size="26" aria-hidden="true" /></div>
      <p class="eyebrow">SEMICONDUCTOR DECISION LAB</p>
      <h1 id="setup-title">把技术问题放到<br /><em>证据工作台</em>上。</h1>
      <p class="setup-intro__lead">
        连接已有 Rag_Agent API，在一个清晰的界面里完成问答、澄清与可审计的技术选型。
      </p>
      <div class="setup-principles">
        <span><Database :size="17" aria-hidden="true" /> 服务端状态保持权威</span>
        <span><ShieldCheck :size="17" aria-hidden="true" /> 本机只保存会话索引</span>
      </div>
    </section>

    <section class="setup-card" aria-label="工作区设置">
      <div>
        <p class="step-label">01 / 本机配置</p>
        <h2>{{ hasExistingSettings ? '更新工作区连接' : '建立工作区连接' }}</h2>
        <p>这些信息只保存在当前浏览器，不包含 API Key 或数据库凭据。</p>
      </div>
      <RequestNotice v-if="error" tone="error" title="设置未保存" :message="error" />
      <RequestNotice v-if="store.warning" tone="warning" :message="store.warning" />
      <form class="settings-form" @submit.prevent="submit">
        <label for="api-base-url">API Base URL</label>
        <input
          id="api-base-url"
          v-model="form.apiBaseUrl"
          type="url"
          required
          spellcheck="false"
          placeholder="http://localhost:8000"
        />
        <p class="field-help">仅支持 HTTP(S)，末尾斜杠会自动整理。</p>

        <div class="form-row">
          <div>
            <label for="user-id">User ID</label>
            <input
              id="user-id"
              v-model="form.userId"
              required
              autocomplete="off"
              placeholder="user-001"
            />
          </div>
          <div>
            <label for="project-id">Project ID</label>
            <input
              id="project-id"
              v-model="form.projectId"
              required
              autocomplete="off"
              placeholder="project-001"
            />
          </div>
        </div>
        <AppButton type="submit">进入工作台 <ArrowRight :size="17" aria-hidden="true" /></AppButton>
      </form>
    </section>
  </main>
</template>
