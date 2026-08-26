import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'

import MessageComposer from '../MessageComposer.vue'

describe('MessageComposer', () => {
  it('updates its draft from a parent v-model value', async () => {
    const wrapper = mount(MessageComposer, { props: { modelValue: '初始问题' } })

    await wrapper.setProps({ modelValue: '补充后的完整问题' })

    expect((wrapper.get('textarea').element as HTMLTextAreaElement).value).toBe('补充后的完整问题')
  })

  it('submits with Enter and preserves Shift+Enter for multiline input', async () => {
    const wrapper = mount(MessageComposer)
    const textarea = wrapper.get('textarea')
    await textarea.setValue('比较 FinFET 和 GAA')
    await textarea.trigger('keydown', { key: 'Enter', shiftKey: true })
    expect(wrapper.emitted('submit')).toBeUndefined()

    await textarea.trigger('keydown', { key: 'Enter', shiftKey: false })

    expect(wrapper.emitted('submit')?.[0]).toEqual(['比较 FinFET 和 GAA', false])
  })

  it('does not submit blank messages or messages longer than the API limit', async () => {
    const wrapper = mount(MessageComposer)
    const textarea = wrapper.get('textarea')
    await textarea.setValue('   ')
    await wrapper.get('form').trigger('submit')
    await textarea.setValue('x'.repeat(10_001))
    await wrapper.get('form').trigger('submit')

    expect(wrapper.emitted('submit')).toBeUndefined()
    expect(wrapper.text()).toContain('10,000')
  })
})
