import { createRouter, createWebHistory } from 'vue-router'

import { useSettingsStore } from '@/features/settings/settingsStore'
import SetupView from '@/features/settings/SetupView.vue'
import WorkspaceView from '@/features/sessions/views/WorkspaceView.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    { path: '/', redirect: '/workspace' },
    { path: '/setup', name: 'setup', component: SetupView },
    {
      path: '/workspace',
      name: 'workspace',
      component: WorkspaceView,
      meta: { requiresSettings: true },
    },
    {
      path: '/workspace/:threadId',
      name: 'session',
      component: WorkspaceView,
      meta: { requiresSettings: true },
    },
    { path: '/:pathMatch(.*)*', redirect: '/workspace' },
  ],
})

router.beforeEach((to) => {
  const settings = useSettingsStore()
  if (!settings.settings) settings.load()
  if (to.meta.requiresSettings && !settings.isConfigured) return { name: 'setup' }
  if (to.name === 'setup' && settings.isConfigured) return { name: 'workspace' }
  return true
})

export default router
