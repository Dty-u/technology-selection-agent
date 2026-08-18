import { computed, ref } from 'vue'
import { defineStore } from 'pinia'

import { readJson, writeJson } from '@/shared/storage/localStorage'
import { normalizeSettings, type AppSettings } from './settingsSchema'

export const SETTINGS_STORAGE_KEY = 'rag-agent.settings.v1'

export const useSettingsStore = defineStore('settings', () => {
  const settings = ref<AppSettings | null>(null)
  const warning = ref<string | null>(null)
  const isConfigured = computed(() => settings.value !== null)

  function load() {
    const result = readJson(localStorage, SETTINGS_STORAGE_KEY, normalizeSettings)
    settings.value = result.value
    warning.value = result.warning ?? null
  }

  function save(value: unknown) {
    const normalized = normalizeSettings(value)
    settings.value = normalized
    const result = writeJson(localStorage, SETTINGS_STORAGE_KEY, normalized)
    warning.value = result.warning ?? null
    return result.ok
  }

  function clear() {
    localStorage.removeItem(SETTINGS_STORAGE_KEY)
    settings.value = null
    warning.value = null
  }

  return { settings, warning, isConfigured, load, save, clear }
})
