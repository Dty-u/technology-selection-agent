import { mount } from '@vue/test-utils'
import { nextTick } from 'vue'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { ApiError } from '@/shared/api/errors'
import { completedRagResponse } from '@/tests/fixtures/sessionResponses'
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
  const longAnswer: ChatMessage = {
    id: 'long-answer',
    role: 'assistant',
    content: '很长的回答',
    createdAt: '2026-08-23T11:00:01.000Z',
    delivery: 'sent',
  }

  function mountScrollable(props: Partial<InstanceType<typeof ConversationPane>['$props']>) {
    const wrapper = mount(ConversationPane, {
      props: {
        messages: [],
        response: null,
        status: 'completed',
        notice: null,
        inFlight: false,
        canExit: false,
        sessionKey: null,
        ...props,
      },
    })
    const scroll = wrapper.get('.conversation-scroll').element as HTMLElement
    const scrollTo = vi.fn<(options?: ScrollToOptions) => void>()
    scroll.scrollTo = scrollTo as typeof scroll.scrollTo
    // An empty list has no overflow, so it reads as "at the bottom" until messages render.
    Object.defineProperties(scroll, {
      scrollHeight: {
        configurable: true,
        get: () => (wrapper.find('.message').exists() ? 4_000 : 600),
      },
      clientHeight: { configurable: true, value: 600 },
    })
    return { wrapper, scroll, scrollTo }
  }

  it('opens a session at its latest answer instead of the bottom', async () => {
    const { wrapper, scroll, scrollTo } = mountScrollable({})
    vi.spyOn(Element.prototype, 'getBoundingClientRect').mockImplementation(function (this: Element) {
      const top = this.classList.contains('message--assistant') ? 500 : 100
      return { top } as DOMRect
    })

    await wrapper.setProps({ sessionKey: 'thread-a', messages: [...completedMessages, longAnswer] })
    await nextTick()

    expect(scroll.scrollTop).toBe(384)
    expect(scrollTo).not.toHaveBeenCalled()
    vi.restoreAllMocks()
  })

  it('still follows a streaming answer when the reader is at the bottom', async () => {
    const { wrapper, scroll } = mountScrollable({
      sessionKey: 'thread-a',
      messages: [...completedMessages, longAnswer],
    })
    const streaming: ChatMessage = { ...longAnswer, delivery: 'streaming', content: '正在' }
    await wrapper.setProps({ messages: [...completedMessages, streaming] })
    scroll.scrollTop = 3_400

    await wrapper.setProps({
      messages: [...completedMessages, { ...streaming, content: '正在生成更多内容' }],
    })
    await nextTick()

    expect(scroll.scrollTop).toBe(4_000)
  })

  it('follows to the bottom when a new message is sent', async () => {
    const { wrapper, scrollTo } = mountScrollable({
      sessionKey: 'thread-a',
      messages: [...completedMessages, longAnswer],
    })

    await wrapper.setProps({ messages: [...completedMessages, longAnswer, pendingMessages[0]!] })
    await nextTick()

    expect(scrollTo).toHaveBeenCalledWith({ top: 4_000, behavior: 'smooth' })
  })

  it('offers a way back to the bottom only when the reader has scrolled away', async () => {
    const { wrapper, scroll, scrollTo } = mountScrollable({
      sessionKey: 'thread-a',
      messages: [...completedMessages, longAnswer],
    })

    scroll.scrollTop = 0
    await wrapper.get('.conversation-scroll').trigger('scroll')
    await wrapper.get('.scroll-to-bottom').trigger('click')
    expect(scrollTo).toHaveBeenCalledWith({ top: 4_000, behavior: 'smooth' })

    scroll.scrollTop = 3_400
    await wrapper.get('.conversation-scroll').trigger('scroll')
    expect(wrapper.find('.scroll-to-bottom').exists()).toBe(false)
  })

  it('moves a failed message back into the composer for editing', async () => {
    const failed: ChatMessage = { ...completedMessages[0]!, id: 'failed', delivery: 'failed' }
    const { wrapper } = mountScrollable({ sessionKey: 'thread-a', messages: [failed] })

    await wrapper.findAll('.message__failure-actions button')[1]?.trigger('click')
    await nextTick()

    expect((wrapper.get('textarea').element as HTMLTextAreaElement).value).toBe(failed.content)
    expect(wrapper.emitted('discardFailed')).toEqual([['failed']])
  })

  it('asks for clarification only when there are questions to answer', async () => {
    const wrapper = mount(ConversationPane, {
      props: {
        messages: completedMessages,
        response: { ...completedRagResponse, status: 'awaiting_user' },
        status: 'awaiting_user',
        notice: null,
        inFlight: false,
        canExit: false,
      },
    })

    expect(wrapper.find('.clarification-card').exists()).toBe(false)
    expect(wrapper.get('textarea').attributes('placeholder')).not.toContain('上方问题')

    await wrapper.setProps({
      response: {
        ...completedRagResponse,
        status: 'awaiting_user',
        clarification_questions: ['目标工艺节点是什么？'],
      },
    })

    expect(wrapper.get('.clarification-card').text()).toContain('目标工艺节点是什么？')
    expect(wrapper.get('textarea').attributes('placeholder')).toContain('上方问题')
  })

  it('shows a readable error title and keeps the code as secondary detail', () => {
    const wrapper = mount(ConversationPane, {
      props: {
        messages: [],
        response: null,
        status: 'error',
        notice: new ApiError(0, 'network_error', '无法连接 Agent API'),
        inFlight: false,
        canExit: false,
      },
    })

    expect(wrapper.get('.request-notice strong').text()).toBe('无法连接服务')
    expect(wrapper.get('.request-notice__detail').text()).toContain('network_error')
  })

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
