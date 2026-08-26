import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'

import SessionSidebar from '../SessionSidebar.vue'

const session = {
  threadId: 'thread-001',
  userId: 'user-001',
  projectId: 'project-001',
  title: '车规级功率器件选型',
  createdAt: '2026-08-24T08:00:00Z',
  updatedAt: '2026-08-24T09:00:00Z',
  status: 'completed' as const,
  messages: [],
}

describe('SessionSidebar', () => {
  it('separates selecting a session from removing its local record', async () => {
    const wrapper = mount(SessionSidebar, {
      props: { sessions: [session], activeThreadId: session.threadId, apiReachable: true },
    })

    const selectButton = wrapper.get('.session-item__select')
    expect(selectButton.attributes('aria-current')).toBe('page')
    await selectButton.trigger('click')
    expect(wrapper.emitted('select')).toEqual([[session.threadId]])

    const removeButton = wrapper.get('button[aria-label="仅移除本机记录：车规级功率器件选型"]')
    await removeButton.trigger('click')
    expect(wrapper.emitted('removeLocal')).toEqual([[session.threadId]])
  })
})
