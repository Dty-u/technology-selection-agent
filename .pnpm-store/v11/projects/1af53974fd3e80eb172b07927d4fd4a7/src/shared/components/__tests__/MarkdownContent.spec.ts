import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'

import MarkdownContent from '../MarkdownContent.vue'

describe('MarkdownContent', () => {
  it('escapes raw HTML and removes javascript links', () => {
    const wrapper = mount(MarkdownContent, {
      props: {
        content: '<img src=x onerror=alert(1)> [危险](javascript:alert(1))',
      },
    })

    expect(wrapper.find('img').exists()).toBe(false)
    expect(wrapper.html()).not.toContain('javascript:')
    expect(wrapper.text()).toContain('<img src=x onerror=alert(1)>')
  })

  it('opens https sources with isolation attributes', () => {
    const wrapper = mount(MarkdownContent, {
      props: { content: '[证据](https://example.com/evidence)' },
    })

    const link = wrapper.get('a')
    expect(link.attributes('target')).toBe('_blank')
    expect(link.attributes('rel')).toContain('noopener')
    expect(link.attributes('rel')).toContain('noreferrer')
  })
})
