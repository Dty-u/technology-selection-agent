import { mount } from '@vue/test-utils'
import { nextTick } from 'vue'
import { describe, expect, it, vi } from 'vitest'

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

  it('shows a local document by file name and hides empty excerpts', () => {
    const path = 'D:\\Mydocs\\rag_md\\md\\tech_report_99duun4t.md'
    const wrapper = mount(SourcesPanel, {
      props: { sources: [{ title: '', url: path, source: path }] },
    })

    expect(wrapper.get('.source-card strong').text()).toBe('tech_report_99duun4t.md')
    expect(wrapper.get('.source-card strong').attributes('title')).toBe(path)
    expect(wrapper.text()).toContain('本地文档')
    expect(wrapper.find('.source-card p').exists()).toBe(false)
    expect(wrapper.find('.unsafe-link-note').exists()).toBe(false)
  })

  it('highlights and reveals the source a citation points to', async () => {
    const scrollIntoView = vi.fn<() => void>()
    Element.prototype.scrollIntoView = scrollIntoView
    const wrapper = mount(SourcesPanel, {
      props: { sources: [{ title: '甲' }, { title: '乙' }], focus: null },
    })
    ;(wrapper.get('details').element as HTMLDetailsElement).open = false

    await wrapper.setProps({ focus: { index: 1, key: 1 } })
    await nextTick()

    expect((wrapper.get('details').element as HTMLDetailsElement).open).toBe(true)
    expect(wrapper.findAll('.source-card')[1]?.classes()).toContain('is-focused')
    expect(wrapper.findAll('.source-card')[0]?.classes()).not.toContain('is-focused')
    expect(scrollIntoView).toHaveBeenCalled()
  })

  it('reveals a citation chosen before the panel mounted, as in the mobile drawer', async () => {
    const wrapper = mount(SourcesPanel, {
      props: { sources: [{ title: '甲' }, { title: '乙' }], focus: { index: 1, key: 1 } },
    })
    await nextTick()

    expect(wrapper.findAll('.source-card')[1]?.classes()).toContain('is-focused')
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
