import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'

import SourcesPanel from '../SourcesPanel.vue'

describe('SourcesPanel', () => {
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
})
