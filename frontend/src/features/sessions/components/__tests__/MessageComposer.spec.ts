import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'

import MessageComposer from '../MessageComposer.vue'

describe('MessageComposer', () => {
  it('updates its draft from a parent v-model value', async () => {
    const wrapper = mount(MessageComposer, { props: { modelValue: '初始问题' } })

    await wrapper.setProps({ modelValue: '补充后的完整问题' })

    expect((wrapper.get('textarea').element as HTMLTextAreaElement).value).toBe('补充后的完整问题')
  })

  it('starts compact and grows with its content up to a limit', async () => {
    const wrapper = mount(MessageComposer)
    const element = wrapper.get('textarea').element as HTMLTextAreaElement
    expect(element.getAttribute('rows')).toBe('1')

    Object.defineProperty(element, 'scrollHeight', { configurable: true, value: 120 })
    await wrapper.get('textarea').setValue('第一行\n第二行\n第三行')
    expect(element.style.height).toBe('120px')
    expect(element.style.overflowY).toBe('hidden')

    Object.defineProperty(element, 'scrollHeight', { configurable: true, value: 900 })
    await wrapper.get('textarea').setValue('很多行'.repeat(100))
    expect(element.style.height).toBe('220px')
    expect(element.style.overflowY).toBe('auto')
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
