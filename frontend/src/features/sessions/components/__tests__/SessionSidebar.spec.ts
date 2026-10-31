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

    const removeButton = wrapper.get('button[aria-label="从列表中移除：车规级功率器件选型"]')
    await removeButton.trigger('click')
    expect(wrapper.emitted('removeLocal')).toEqual([[session.threadId]])
  })

  it('groups sessions by recency, newest first', () => {
    const now = new Date()
    const yesterday = new Date(now)
    yesterday.setDate(now.getDate() - 1)
    const wrapper = mount(SessionSidebar, {
      props: {
        sessions: [
          session,
          { ...session, threadId: 'y', title: '昨天的', updatedAt: yesterday.toISOString() },
          { ...session, threadId: 't', title: '今天的', updatedAt: now.toISOString() },
        ],
        activeThreadId: null,
        apiReachable: true,
      },
    })

    expect(wrapper.findAll('.session-group-label').map((label) => label.text())).toEqual([
      '今天',
      '昨天',
      '更早',
    ])
    expect(wrapper.findAll('.session-item__title').map((title) => title.text())).toEqual([
      '今天的',
      '昨天的',
      '车规级功率器件选型',
    ])
  })

  it('offers search only once the list is long enough to need it', async () => {
    const sessions = Array.from({ length: 9 }, (_, index) => ({
      ...session,
      threadId: `thread-${index}`,
      title: index === 0 ? '光刻机是啥' : `其他会话 ${index}`,
    }))
    const short = mount(SessionSidebar, {
      props: { sessions: sessions.slice(0, 8), activeThreadId: null, apiReachable: true },
    })
    const long = mount(SessionSidebar, {
      props: { sessions, activeThreadId: null, apiReachable: true },
    })

    expect(short.find('input[type="search"]').exists()).toBe(false)
    await long.get('input[type="search"]').setValue('光刻')
    expect(long.findAll('.session-item__title').map((title) => title.text())).toEqual(['光刻机是啥'])
    await long.get('input[type="search"]').setValue('不存在')
    expect(long.text()).toContain('没有匹配的会话')
  })

  it('shows an unknown status instead of a stale one', () => {
    const wrapper = mount(SessionSidebar, {
      props: {
        sessions: [{ ...session, status: 'running' as const }],
        activeThreadId: null,
        apiReachable: true,
        unknownStatusIds: new Set([session.threadId]),
      },
    })

    expect(wrapper.get('.session-item__meta').text()).toContain('状态未知')
    expect(wrapper.get('.session-item__meta').text()).not.toContain('处理中')
  })
})
