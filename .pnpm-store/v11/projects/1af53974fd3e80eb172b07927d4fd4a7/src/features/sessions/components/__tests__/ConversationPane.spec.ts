import { mount } from '@vue/test-utils'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { ChatMessage } from '../../model/sessionTypes'
import ConversationPane from '../ConversationPane.vue'

const pendingMessages: ChatMessage[] = [
  {
    id: 'running-request',
    role: 'user',
    content: '继续执行技术选型',
    createdAt: '2026-08-23T12:00:00.000Z',
    delivery: 'pending',
    requestId: 'request-running',
  },
]

const completedMessages: ChatMessage[] = [
  {
    id: 'completed-request',
    role: 'user',
    content: '另一个会话',
    createdAt: '2026-08-23T11:00:00.000Z',
    delivery: 'sent',
  },
]

afterEach(() => vi.useRealTimers())

describe('ConversationPane', () => {
  it('fills and focuses the composer with the selected detailed prompt', async () => {
    const wrapper = mount(ConversationPane, {
      attachTo: document.body,
      props: {
        messages: [],
        response: null,
        status: 'completed',
        notice: null,
        inFlight: false,
        canExit: false,
      },
    })

    await wrapper.get('[data-quick-start="比较 FinFET 与 GAA"]').trigger('click')

    expect((wrapper.get('textarea').element as HTMLTextAreaElement).value).toBe(
      '请系统比较 FinFET 与 GAA 的结构与工作原理，并从性能、功耗、制造复杂度、工艺成熟度、成本、设计生态和典型适用场景说明差异；最后总结各自的优势、局限与适用条件。',
    )
    expect(document.activeElement).toBe(wrapper.get('textarea').element)
    expect(wrapper.emitted('submit')).toBeUndefined()
    wrapper.unmount()
  })

  it('keeps elapsed waiting time when leaving and returning to a running session', async () => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date('2026-08-23T12:00:20.000Z'))

    const wrapper = mount(ConversationPane, {
      props: {
        messages: pendingMessages,
        response: null,
        status: 'running',
        notice: null,
        inFlight: true,
        canExit: true,
      },
    })

    expect(wrapper.text()).toContain('已等待 20 秒')
    await vi.advanceTimersByTimeAsync(5_000)

    await wrapper.setProps({
      messages: completedMessages,
      status: 'completed',
      inFlight: false,
      canExit: false,
    })
    expect(wrapper.text()).not.toContain('服务端正在处理')

    await wrapper.setProps({
      messages: pendingMessages,
      status: 'running',
      inFlight: true,
      canExit: true,
    })

    expect(wrapper.text()).toContain('已等待 25 秒')
  })

  it('emits exit for a running request and explains cooperative cancellation', async () => {
    const wrapper = mount(ConversationPane, {
      props: {
        messages: pendingMessages,
        response: null,
        status: 'running',
        notice: null,
        inFlight: true,
        canExit: true,
      },
    })

    await wrapper.get('button.text-button').trigger('click')

    expect(wrapper.emitted('exit')).toHaveLength(1)
    await wrapper.setProps({ status: 'cancelling', canExit: false })
    expect(wrapper.text()).toContain('正在结束本次请求')
    expect(wrapper.text()).toContain('当前节点结束后将停止后续工作流步骤')
  })
})
