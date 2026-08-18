import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'

import { validReportData } from '@/tests/fixtures/sessionResponses'
import ReportPanel from '../ReportPanel.vue'

describe('ReportPanel', () => {
  it('renders conditional decisions as review-required instead of a final approval', () => {
    const wrapper = mount(ReportPanel, { props: { report: validReportData } })

    expect(wrapper.text()).toContain('有条件推荐')
    expect(wrapper.text()).toContain('需要人工复核')
    expect(wrapper.text()).toContain('75%')
    expect(wrapper.text()).toContain('缺少量产良率数据')
    expect(wrapper.text()).not.toContain('已批准')
  })
})
