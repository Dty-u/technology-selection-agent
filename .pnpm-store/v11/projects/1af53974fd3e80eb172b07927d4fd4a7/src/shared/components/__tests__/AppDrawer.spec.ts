import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'

import AppDrawer from '../AppDrawer.vue'

describe('AppDrawer', () => {
  it('closes with Escape and exposes a labelled dialog', async () => {
    const wrapper = mount(AppDrawer, {
      props: { open: true, title: '会话列表' },
      slots: { default: '<button>内容</button>' },
      attachTo: document.body,
    })

    const dialog = wrapper.get('[role="dialog"]')
    expect(dialog.attributes('aria-label')).toBe('会话列表')
    await dialog.trigger('keydown', { key: 'Escape' })
    expect(wrapper.emitted('close')).toHaveLength(1)
    wrapper.unmount()
  })

  it('moves focus into the drawer and restores it after closing', async () => {
    const trigger = document.createElement('button')
    trigger.textContent = '打开抽屉'
    document.body.append(trigger)
    trigger.focus()

    const wrapper = mount(AppDrawer, {
      props: { open: false, title: '会话列表' },
      slots: { default: '<button>内容</button>' },
      attachTo: document.body,
    })

    await wrapper.setProps({ open: true })
    await wrapper.vm.$nextTick()
    expect(document.activeElement).toBe(wrapper.get('button[aria-label="关闭"]').element)

    await wrapper.setProps({ open: false })
    expect(document.activeElement).toBe(trigger)
    wrapper.unmount()
    trigger.remove()
  })
})
