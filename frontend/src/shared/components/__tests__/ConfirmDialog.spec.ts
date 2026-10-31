import { mount } from '@vue/test-utils'
import { nextTick } from 'vue'
import { describe, expect, it } from 'vitest'

import ConfirmDialog from '../ConfirmDialog.vue'

const request = {
  title: '结束这个会话？',
  message: '结束后无法继续。',
  details: ['会删除：服务端的会话进度。'],
  confirmLabel: '结束并删除',
}

describe('ConfirmDialog', () => {
  it('renders nothing until a request arrives', () => {
    const wrapper = mount(ConfirmDialog, { props: { request: null } })

    expect(wrapper.find('[role="alertdialog"]').exists()).toBe(false)
  })

  it('states what will be affected and defaults focus to cancel', async () => {
    const wrapper = mount(ConfirmDialog, {
      attachTo: document.body,
      props: { request: null },
    })

    await wrapper.setProps({ request })
    await nextTick()

    const dialog = wrapper.get('[role="alertdialog"]')
    expect(dialog.text()).toContain('会删除：服务端的会话进度。')
    expect(document.activeElement).toBe(wrapper.get('.confirm-dialog__cancel').element)
    wrapper.unmount()
  })

  it('settles true only from the confirm button', async () => {
    const wrapper = mount(ConfirmDialog, { props: { request } })

    await wrapper.get('.confirm-dialog__cancel').trigger('click')
    await wrapper.get('.drawer-backdrop').trigger('click')
    await wrapper.get('[role="alertdialog"]').trigger('keydown', { key: 'Escape' })
    await wrapper.get('.confirm-dialog__confirm').trigger('click')

    expect(wrapper.emitted('settle')).toEqual([[false], [false], [false], [true]])
  })

  it('keeps keyboard focus inside the dialog', async () => {
    const wrapper = mount(ConfirmDialog, {
      attachTo: document.body,
      props: { request: null },
    })
    await wrapper.setProps({ request })
    await nextTick()

    await wrapper.get('[role="alertdialog"]').trigger('keydown', { key: 'Tab' })
    expect(document.activeElement).toBe(wrapper.get('.confirm-dialog__confirm').element)
    await wrapper.get('[role="alertdialog"]').trigger('keydown', { key: 'Tab' })
    expect(document.activeElement).toBe(wrapper.get('.confirm-dialog__cancel').element)
    wrapper.unmount()
  })
})
