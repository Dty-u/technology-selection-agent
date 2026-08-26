import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'

import ReportHistoryPanel from '../ReportHistoryPanel.vue'

describe('ReportHistoryPanel', () => {
  const reports = [
    {
      report_id: 'report-1',
      title: '移动 SoC 技术选型',
      created_at: '2026-08-23T09:00:00+00:00',
      format: 'markdown' as const,
      download_url: '/api/v1/projects/project-a/sessions/thread-a/reports/report-1',
    },
    {
      report_id: 'report-2',
      title: '功率器件技术选型',
      created_at: '2026-08-23T10:00:00+00:00',
      format: 'docx' as const,
      download_url: '/api/v1/projects/project-a/sessions/thread-a/reports/report-2',
    },
  ]

  it('lists every report newest first and emits the selected download url', async () => {
    const wrapper = mount(ReportHistoryPanel, { props: { reports } })

    expect(wrapper.text()).toContain('选型报告')
    expect(wrapper.text()).toContain('移动 SoC 技术选型')
    expect(wrapper.text()).toContain('功率器件技术选型')
    expect(wrapper.findAll('[data-report-item]')[0]?.text()).toContain('功率器件技术选型')

    await wrapper.findAll('button')[0]!.trigger('click')
    expect(wrapper.emitted('download')?.[0]).toEqual([reports[1]])
  })
})
