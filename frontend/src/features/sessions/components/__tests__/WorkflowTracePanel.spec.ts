import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'

import WorkflowTracePanel from '../WorkflowTracePanel.vue'

describe('WorkflowTracePanel', () => {
  it('renders the backend-reported current stage and completed trace', () => {
    const wrapper = mount(WorkflowTracePanel, {
      props: {
        progress: {
          request_id: '11111111-1111-4111-8111-111111111111',
          active: true,
          cancel_requested: false,
          cancelled: false,
          current_stage: {
            scope: '普通RAG',
            stage_id: 'generate',
            label: '生成回答',
          },
          events: [
            {
              scope: '普通RAG',
              stage_id: 'vector_retrieve',
              label: '向量检索',
              status: 'completed' as const,
              started_at: '2026-08-23T00:00:00+00:00',
              finished_at: '2026-08-23T00:00:00.250000+00:00',
              duration_ms: 250,
            },
            {
              scope: '普通RAG',
              stage_id: 'generate',
              label: '生成回答',
              status: 'running' as const,
              started_at: '2026-08-23T00:00:01+00:00',
              finished_at: null,
              duration_ms: null,
            },
          ],
        },
      },
    })

    expect(wrapper.get('[role="status"]').text()).toContain('当前阶段 · 普通RAG')
    expect(wrapper.get('[role="status"]').text()).toContain('生成回答')
    expect(wrapper.text()).toContain('向量检索')
    expect(wrapper.text()).toContain('250 ms')
  })

  it('does not invent stages when the backend has no events', () => {
    const wrapper = mount(WorkflowTracePanel)

    expect(wrapper.text()).toContain('此会话没有保留执行轨迹')
    expect(wrapper.findAll('.workflow-event')).toHaveLength(0)
  })

  it('does not tell a finished session to submit a request first', () => {
    const progress = {
      request_id: null,
      active: false,
      cancel_requested: false,
      cancelled: false,
      current_stage: null,
      events: [],
    }
    const finished = mount(WorkflowTracePanel, { props: { progress } })
    const active = mount(WorkflowTracePanel, { props: { progress: { ...progress, active: true } } })

    expect(finished.text()).toContain('此会话没有保留执行轨迹')
    expect(finished.text()).not.toContain('提交请求后')
    expect(active.text()).toContain('正在等待后端上报执行阶段')
  })

  it('toggles the execution trace from its heading while keeping the section data', async () => {
    const wrapper = mount(WorkflowTracePanel)
    const details = wrapper.get('details')
    const summary = wrapper.get('summary')

    expect((details.element as HTMLDetailsElement).open).toBe(true)
    await summary.trigger('click')
    expect((details.element as HTMLDetailsElement).open).toBe(false)
    expect(wrapper.text()).toContain('执行轨迹')
    expect(wrapper.text()).toContain('此会话没有保留执行轨迹')
  })
})
