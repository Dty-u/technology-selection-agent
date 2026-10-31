import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { useSettingsStore } from '@/features/settings/settingsStore'
import { jsonResponse } from '@/tests/fixtures/sessionResponses'
import ReportLibrary from '../ReportLibrary.vue'

const THREAD = '11111111-1111-4111-8111-111111111111'

function report(id: string, overrides: Record<string, string> = {}) {
  return {
    report_id: id,
    title: `${id} 报告`,
    created_at: '2026-01-01T00:00:00+00:00',
    format: 'markdown',
    download_url: `/api/v1/projects/project-a/sessions/${THREAD}/reports/${id}`,
    project_id: 'project-a',
    thread_id: THREAD,
    session_title: '会话一',
    ...overrides,
  }
}

function mockFetch(reports: unknown[], overrides: Record<string, () => Response> = {}) {
  const fetcher = vi.fn<typeof fetch>(async (input, init) => {
    const url = String(input)
    const method = init?.method ?? 'GET'
    const override = overrides[`${method} ${new URL(url).pathname}`]
    if (override) return override()
    if (url.endsWith('/api/v1/reports')) return jsonResponse({ reports })
    if (method === 'DELETE') return new Response(null, { status: 204 })
    return jsonResponse({
      report_id: 'one',
      title: 'one 报告',
      created_at: '2026-01-01T00:00:00+00:00',
      format: 'markdown',
      content: '# 报告正文',
    })
  })
  vi.stubGlobal('fetch', fetcher)
  return fetcher
}

function mountLibrary(confirm = vi.fn<() => Promise<boolean>>(async () => true)) {
  const wrapper = mount(ReportLibrary, { props: { currentProjectId: 'project-a', confirm } })
  return { wrapper, confirm }
}

describe('ReportLibrary', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    useSettingsStore().settings = {
      apiBaseUrl: 'http://localhost:8100',
      userId: 'user-a',
      projectId: 'project-a',
    }
  })

  afterEach(() => vi.unstubAllGlobals())

  it('lists every report for the user with its identity header', async () => {
    const fetcher = mockFetch([
      report('one'),
      report('two', { project_id: 'project-b', session_title: '会话二' }),
    ])
    const { wrapper } = mountLibrary()
    await flushPromises()

    expect(fetcher.mock.calls[0]?.[0]).toBe('http://localhost:8100/api/v1/reports')
    expect(fetcher.mock.calls[0]?.[1]?.headers).toMatchObject({ 'X-User-ID': 'user-a' })
    expect(wrapper.findAll('[data-report-item]')).toHaveLength(2)
    expect(wrapper.text()).toContain('项目 project-b')
  })

  it('explains an empty library', async () => {
    mockFetch([])
    const { wrapper } = mountLibrary()
    await flushPromises()

    expect(wrapper.text()).toContain('还没有报告')
  })

  it('opens a session only for reports in the current project', async () => {
    mockFetch([report('one'), report('two', { project_id: 'project-b' })])
    const { wrapper } = mountLibrary()
    await flushPromises()

    expect(wrapper.find('button[aria-label="打开所在会话：two 报告"]').exists()).toBe(false)
    await wrapper.get('button[aria-label="打开所在会话：one 报告"]').trigger('click')
    expect(wrapper.emitted('openSession')).toEqual([[THREAD]])
  })

  it('shows the report text when one is opened', async () => {
    mockFetch([report('one')])
    const { wrapper } = mountLibrary()
    await flushPromises()

    await wrapper.get('.report-library__open').trigger('click')
    await flushPromises()

    expect(wrapper.get('.report-library__content').text()).toContain('报告正文')
    await wrapper.get('.report-library__back').trigger('click')
    expect(wrapper.findAll('[data-report-item]')).toHaveLength(1)
  })

  it('deletes through the report project after confirmation and drops the row', async () => {
    const fetcher = mockFetch([report('one', { project_id: 'project b' }), report('two')])
    const { wrapper, confirm } = mountLibrary()
    await flushPromises()

    await wrapper.get('button[aria-label="删除 one 报告"]').trigger('click')
    await flushPromises()

    expect(confirm).toHaveBeenCalledOnce()
    const call = fetcher.mock.calls.find(([, init]) => init?.method === 'DELETE')
    expect(call?.[0]).toBe(
      `http://localhost:8100/api/v1/projects/project%20b/sessions/${THREAD}/reports/one`,
    )
    expect(wrapper.findAll('[data-report-item]')).toHaveLength(1)
    expect(wrapper.emitted('deleted')).toHaveLength(1)
  })

  it('does nothing when deletion is declined', async () => {
    const fetcher = mockFetch([report('one')])
    const { wrapper } = mountLibrary(vi.fn<() => Promise<boolean>>(async () => false))
    await flushPromises()

    await wrapper.get('button[aria-label="删除 one 报告"]').trigger('click')
    await flushPromises()

    expect(fetcher.mock.calls.some(([, init]) => init?.method === 'DELETE')).toBe(false)
    expect(wrapper.findAll('[data-report-item]')).toHaveLength(1)
  })

  it('keeps the row and explains why when the session is busy', async () => {
    mockFetch([report('one')], {
      [`DELETE /api/v1/projects/project-a/sessions/${THREAD}/reports/one`]: () =>
        jsonResponse({ detail: { code: 'session_busy', message: 'busy' } }, 409),
    })
    const { wrapper } = mountLibrary()
    await flushPromises()

    await wrapper.get('button[aria-label="删除 one 报告"]').trigger('click')
    await flushPromises()

    expect(wrapper.findAll('[data-report-item]')).toHaveLength(1)
    expect(wrapper.text()).toContain('完成后才能删除这份报告')
    expect(wrapper.emitted('deleted')).toBeUndefined()
  })
})
