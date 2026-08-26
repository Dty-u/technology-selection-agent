import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'

import SourcesPanel from '../SourcesPanel.vue'

describe('SourcesPanel', () => {
  it('shows an honest waiting state while the workflow is still running', () => {
    const wrapper = mount(SourcesPanel, {
      props: {
        sources: [],
        pending: true,
      },
    })

    expect(wrapper.get('summary').text()).toContain('来源')
    expect(wrapper.text()).toContain('正在等待工作流返回来源')
    expect(wrapper.text()).not.toContain('本次响应未返回来源')
  })

  it('only turns HTTP(S) source URLs into safe external links', () => {
    const wrapper = mount(SourcesPanel, {
      props: {
        sources: [
          { title: '安全来源', url: 'https://example.com/evidence', content: '内容' },
          { title: '危险来源', url: 'javascript:alert(1)', content: '不可点击' },
        ],
      },
    })

    const links = wrapper.findAll('a')
    expect(links).toHaveLength(1)
    expect(links[0]?.attributes()).toMatchObject({
      href: 'https://example.com/evidence',
      target: '_blank',
      rel: 'noopener noreferrer',
    })
    expect(wrapper.text()).toContain('危险来源')
  })

  it('toggles the source list from its heading while keeping the source count', async () => {
    const wrapper = mount(SourcesPanel, {
      props: {
        sources: [{ title: '晶圆制造来源', content: '来源摘要' }],
      },
    })
    const details = wrapper.get('details')
    const summary = wrapper.get('summary')

    expect((details.element as HTMLDetailsElement).open).toBe(true)
    await summary.trigger('click')
    expect((details.element as HTMLDetailsElement).open).toBe(false)
    expect(summary.text()).toContain('来源')
    expect(summary.text()).toContain('1')
    expect(wrapper.text()).toContain('晶圆制造来源')
  })
})
