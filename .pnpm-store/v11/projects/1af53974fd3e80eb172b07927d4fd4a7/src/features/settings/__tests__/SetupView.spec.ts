import { createPinia, setActivePinia } from 'pinia'
import { mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import SetupView from '../SetupView.vue'

const push = vi.fn<(location: string) => void>()
vi.mock('vue-router', () => ({ useRouter: () => ({ push }) }))

describe('SetupView', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    localStorage.clear()
    push.mockReset()
  })

  it('defaults a new local workspace to the available backend port', () => {
    const wrapper = mount(SetupView)

    expect(wrapper.get<HTMLInputElement>('#api-base-url').element.value).toBe(
      'http://localhost:8100',
    )
  })

  it('saves a valid local workspace and opens the conversation area', async () => {
    const wrapper = mount(SetupView)
    await wrapper.get('#api-base-url').setValue('http://localhost:8000/')
    await wrapper.get('#user-id').setValue('user-001')
    await wrapper.get('#project-id').setValue('project-001')
    await wrapper.get('form').trigger('submit')

    expect(JSON.parse(localStorage.getItem('rag-agent.settings.v1') ?? '{}')).toEqual({
      apiBaseUrl: 'http://localhost:8000',
      userId: 'user-001',
      projectId: 'project-001',
    })
    expect(push).toHaveBeenCalledWith('/workspace')
  })
})
