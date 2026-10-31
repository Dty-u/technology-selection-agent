import { mount } from '@vue/test-utils'
import { nextTick } from 'vue'
import { describe, expect, it, vi } from 'vitest'

import MessageBubble from '../MessageBubble.vue'

const evidenceReport = `## 8. 证据索引

### [E1783F42F82AC] 2nm 手机芯片已开始量产
- 来源类型：有来源网页
- 来源：[查看来源](https://www.samsung.com/semiconductor/2nm-chip)
- 支持关系：候选方案/生产效率/直接证据
- 原文片段：这是一段不应在聊天正文展示的原文。

### [E1783F42F82AD] Samsung 量产进展
- 来源：[查看来源](https://samsung.com/news/foundry)

### [E1783F42F82AE] 行业市场分析
- 来源：[查看来源](https://trendforce.com/research/advanced-node)

### [E1783F42F82AF] 供应链观察
- 来源：[查看来源](https://www.pchome.com.tw/article/chip)

### [E1783F42F82AG] 其他行业资料
- 来源：[查看来源](https://news.example.net/chip)

## 9. 审核说明

- 审核结果：通过
`

describe('MessageBubble', () => {
  const answer = {
    id: 'assistant-answer',
    role: 'assistant' as const,
    createdAt: '2026-08-23T14:00:00.000Z',
    delivery: 'sent' as const,
    content: `GAA 优势明显[S1][S4]，来源未知[S9]。

## 来源引用

- [S1] 全环绕栅极晶体管，\`D:\\docs\\a.md\`
- [S4] 技术改进，\`D:\\docs\\b.md\`
- [S5] 驱动电流，\`D:\\docs\\b.md\``,
  }
  const sources = [
    { title: '', url: 'D:\\docs\\a.md', source: 'D:\\docs\\a.md' },
    { title: '', url: 'D:\\docs\\b.md', source: 'D:\\docs\\b.md' },
  ]

  it('links citations by document rather than by their number', async () => {
    const wrapper = mount(MessageBubble, { props: { message: answer, sources } })

    await wrapper.get('[data-citation="4"]').trigger('click')
    await wrapper.get('[data-citation="1"]').trigger('click')

    expect(wrapper.emitted('focusSource')).toEqual([[1], [0]])
  })

  it('leaves citations it cannot resolve as plain text', () => {
    const wrapper = mount(MessageBubble, { props: { message: answer, sources } })

    expect(wrapper.find('[data-citation="9"]').exists()).toBe(false)
    expect(wrapper.text()).toContain('[S9]')
  })

  it('does not turn citation-like text in code or user messages into links', () => {
    const inCode = mount(MessageBubble, {
      props: {
        message: { ...answer, content: '`[S1]`\n\n- [S1] `D:\\docs\\a.md`' },
        sources,
      },
    })
    const fromUser = mount(MessageBubble, {
      props: {
        message: { ...answer, role: 'user', content: '[S1]\n\n- [S1] D:\\docs\\a.md' },
        sources,
      },
    })

    expect(inCode.findAll('[data-citation]')).toHaveLength(1)
    expect(inCode.get('code').text()).toBe('[S1]')
    expect(fromUser.find('[data-citation]').exists()).toBe(false)
  })

  it('offers retry and edit only on failed messages', async () => {
    const failed = { ...answer, role: 'user' as const, delivery: 'failed' as const, content: '黑' }
    const wrapper = mount(MessageBubble, { props: { message: failed } })

    const [retry, edit] = wrapper.findAll('.message__failure-actions button')
    await retry?.trigger('click')
    await edit?.trigger('click')
    expect(wrapper.emitted('retry')).toEqual([[failed.id]])
    expect(wrapper.emitted('edit')).toEqual([[failed]])

    await wrapper.setProps({ actionsDisabled: true })
    expect(retry?.attributes('disabled')).toBeDefined()

    const sent = mount(MessageBubble, { props: { message: answer } })
    expect(sent.find('.message__failure-actions').exists()).toBe(false)
  })

  it('copies a finished answer but not a streaming one', async () => {
    const writeText = vi.fn<(text: string) => Promise<void>>().mockResolvedValue(undefined)
    vi.stubGlobal('navigator', { clipboard: { writeText } })
    const wrapper = mount(MessageBubble, { props: { message: answer } })

    await wrapper.get('.message__actions button').trigger('click')
    await nextTick()
    await nextTick()

    expect(writeText).toHaveBeenCalledWith(answer.content)
    expect(wrapper.get('.message__actions').text()).toContain('已复制')

    await wrapper.setProps({ message: { ...answer, delivery: 'streaming' } })
    expect(wrapper.find('.message__actions').exists()).toBe(false)
    vi.unstubAllGlobals()
  })

  it('summarizes evidence count and the first three unique source hosts', () => {
    const wrapper = mount(MessageBubble, {
      props: {
        message: {
          id: 'assistant-report',
          role: 'assistant',
          content: evidenceReport,
          createdAt: '2026-08-23T14:00:00.000Z',
          delivery: 'sent',
        },
      },
    })

    expect(wrapper.text()).toContain('8. 证据来源概览')
    expect(wrapper.text()).toContain('本报告共参考 5 条证据，来自 4 个来源')
    expect(wrapper.text()).toContain('samsung.com、trendforce.com、pchome.com.tw 等。')
    expect(wrapper.findAll('a').map((link) => link.attributes('href'))).toEqual([
      'https://www.samsung.com/semiconductor/2nm-chip',
      'https://trendforce.com/research/advanced-node',
      'https://www.pchome.com.tw/article/chip',
    ])
    expect(wrapper.text()).toContain('9. 审核说明')
    expect(wrapper.text()).not.toContain('2nm 手机芯片已开始量产')
    expect(wrapper.text()).not.toContain('news.example.net')
    expect(wrapper.text()).not.toContain('E1783F42F82AC')
    expect(wrapper.text()).not.toContain('来源类型')
    expect(wrapper.text()).not.toContain('支持关系')
    expect(wrapper.text()).not.toContain('原文片段')
    expect(wrapper.text()).not.toContain('这是一段不应在聊天正文展示的原文')
  })

  it('uses a compact notice when evidence has no valid external source', () => {
    const wrapper = mount(MessageBubble, {
      props: {
        message: {
          id: 'assistant-internal-report',
          role: 'assistant',
          content: `## 8. 证据索引

### [E1] 内部资料
- 来源：内部知识库记录

### [E2] 无效外部资料
- 来源：[查看来源](javascript:alert('unsafe'))

## 9. 审核说明`,
          createdAt: '2026-08-23T14:00:00.000Z',
          delivery: 'sent',
        },
      },
    })

    expect(wrapper.text()).toContain('本报告共参考 2 条证据，未发现可展示的外部来源。')
    expect(wrapper.findAll('a')).toHaveLength(0)
    expect(wrapper.text()).not.toContain('内部资料')
    expect(wrapper.text()).not.toContain('javascript')
  })

  it('replaces an unrecognized evidence section with a compact fallback', () => {
    const wrapper = mount(MessageBubble, {
      props: {
        message: {
          id: 'assistant-malformed-report',
          role: 'assistant',
          content: `## 8. 证据索引

这里是一段无法识别且不应继续展开的证据详情。

## 9. 审核说明`,
          createdAt: '2026-08-23T14:00:00.000Z',
          delivery: 'sent',
        },
      },
    })

    expect(wrapper.text()).toContain('证据来源暂无法概括，请在右侧证据面板查看。')
    expect(wrapper.text()).toContain('9. 审核说明')
    expect(wrapper.text()).not.toContain('无法识别且不应继续展开')
  })

  it('does not rewrite evidence-like text authored by the user', () => {
    const wrapper = mount(MessageBubble, {
      props: {
        message: {
          id: 'user-content',
          role: 'user',
          content: evidenceReport,
          createdAt: '2026-08-23T14:00:00.000Z',
          delivery: 'sent',
        },
      },
    })

    expect(wrapper.text()).toContain('支持关系')
    expect(wrapper.text()).toContain('原文片段')
  })
})
