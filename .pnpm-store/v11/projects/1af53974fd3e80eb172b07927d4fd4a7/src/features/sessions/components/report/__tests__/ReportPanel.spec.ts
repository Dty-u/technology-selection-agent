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

  it('shows a concise evidence preview until the user asks for the remaining sources', async () => {
    const evidence = Array.from({ length: 5 }, (_, index) => ({
      ...validReportData.evidence[0]!,
      evidence_id: `ev-${index + 1}`,
      title: `证据来源 ${index + 1}`,
    }))
    const wrapper = mount(ReportPanel, {
      props: { report: { ...validReportData, evidence } },
    })

    expect(wrapper.text()).toContain('证据来源 1')
    expect(wrapper.text()).toContain('证据来源 3')
    expect(wrapper.text()).not.toContain('证据来源 4')
    expect(wrapper.get('button').text()).toContain('显示其余 2 条')

    await wrapper.get('button').trigger('click')

    expect(wrapper.text()).toContain('证据来源 4')
    expect(wrapper.text()).toContain('证据来源 5')
    expect(wrapper.get('button').text()).toContain('收起来源')
  })
})
