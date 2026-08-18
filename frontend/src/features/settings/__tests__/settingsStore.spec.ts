import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it } from 'vitest'

import { useSettingsStore } from '../settingsStore'

describe('settingsStore', () => {
  beforeEach(() => {
    localStorage.clear()
    setActivePinia(createPinia())
  })

  it('persists normalized settings and restores them in a new Pinia instance', () => {
    const store = useSettingsStore()
    store.save({
      apiBaseUrl: 'http://localhost:8000/',
      userId: ' user-a ',
      projectId: ' project-a ',
    })

    setActivePinia(createPinia())
    const restored = useSettingsStore()
    restored.load()

    expect(restored.settings).toEqual({
      apiBaseUrl: 'http://localhost:8000',
      userId: 'user-a',
      projectId: 'project-a',
    })
    expect(restored.isConfigured).toBe(true)
  })
})
