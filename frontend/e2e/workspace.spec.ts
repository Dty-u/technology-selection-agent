import { expect, test, type Page, type Route } from '@playwright/test'

import {
  awaitingSelectionResponse,
  completedRagResponse,
  validReportData,
} from '../src/tests/fixtures/sessionResponses'

const apiBaseUrl = 'http://localhost:8000'
const threadId = completedRagResponse.thread_id

async function installSettings(page: Page) {
  await page.addInitScript(
    ({ baseUrl }) => {
      localStorage.setItem(
        'rag-agent.settings.v1',
        JSON.stringify({ apiBaseUrl: baseUrl, userId: 'user-001', projectId: 'project-001' }),
      )
    },
    { baseUrl: apiBaseUrl },
  )
}

async function mockHealth(page: Page) {
  await page.route(`${apiBaseUrl}/health/live`, (route) =>
    route.fulfill({ json: { status: 'ok' } }),
  )
}

function sessionRecord() {
  return {
    threadId,
    userId: 'user-001',
    projectId: 'project-001',
    title: 'FinFET 基础问题',
    createdAt: '2026-08-18T00:00:00.000Z',
    updatedAt: '2026-08-18T00:00:01.000Z',
    status: 'completed',
    messages: [
      {
        id: 'local-message-1',
        role: 'user',
        content: '什么是 FinFET？',
        createdAt: '2026-08-18T00:00:00.000Z',
        delivery: 'sent',
      },
    ],
  }
}

test('首次设置后创建 RAG 会话并安全查看来源', async ({ page }) => {
  await mockHealth(page)
  await page.route(`${apiBaseUrl}/api/v1/projects/project-001/sessions`, async (route) => {
    const request = route.request()
    expect(request.headers()['x-user-id']).toBe('user-001')
    expect(request.headers()['x-request-id']).toBeTruthy()
    expect(await request.postDataJSON()).toEqual({
      message: '什么是 FinFET？',
      report_format: 'markdown',
    })
    await route.fulfill({
      status: 201,
      json: { ...completedRagResponse, project_id: 'project-001' },
    })
  })

  await page.goto('/')
  await page.getByLabel('API Base URL').fill(apiBaseUrl)
  await page.getByLabel('User ID').fill('user-001')
  await page.getByLabel('Project ID').fill('project-001')
  await page.getByRole('button', { name: '进入工作台' }).click()
  await page.getByRole('textbox', { name: '消息', exact: true }).fill('什么是 FinFET？')
  await page.getByRole('button', { name: '发送' }).click()

  await expect(page).toHaveURL(new RegExp(`/workspace/${threadId}$`))
  await expect(page.getByText('FinFET 使用鳍式沟道增强栅控能力。')).toBeVisible()
  await expect(page.getByRole('heading', { name: '来源' })).toBeVisible()
  const source = page.getByRole('link', { name: '查看来源' })
  await expect(source).toHaveAttribute('rel', 'noopener noreferrer')
})

test('技术选型在同一会话完成澄清并渲染待复核报告', async ({ page }) => {
  await installSettings(page)
  await mockHealth(page)
  let createCount = 0
  await page.route(`${apiBaseUrl}/api/v1/projects/project-001/sessions`, async (route) => {
    createCount += 1
    await route.fulfill({
      status: 201,
      json: { ...awaitingSelectionResponse, project_id: 'project-001' },
    })
  })
  await page.route(
    `${apiBaseUrl}/api/v1/projects/project-001/sessions/${threadId}/messages`,
    async (route) => {
      expect((await route.request().postDataJSON()).message).toBe('3nm，功耗优先')
      await route.fulfill({
        json: {
          ...completedRagResponse,
          project_id: 'project-001',
          status: 'needs_human_review',
          query_type: 'technology_selection',
          response_type: 'report',
          answer: '',
          report: '# GAA 技术选型报告',
          report_data: validReportData,
        },
      })
    },
  )

  await page.goto('/workspace')
  await page.getByRole('textbox', { name: '消息', exact: true }).fill('比较 FinFET 与 GAA')
  await page.getByRole('button', { name: '发送' }).click()
  await expect(page.getByRole('heading', { name: '补充信息后继续同一会话' })).toBeVisible()
  await expect(page.getByText('目标制程节点是什么？')).toBeVisible()

  await page.getByRole('textbox', { name: '消息', exact: true }).fill('3nm，功耗优先')
  await page.getByRole('button', { name: '发送' }).click()
  await expect(page.getByRole('heading', { name: 'GAA', exact: true })).toBeVisible()
  await expect(page.getByText('缺少量产良率数据')).toBeVisible()
  await expect(page.getByText('需要人工复核', { exact: true }).first()).toBeVisible()
  expect(createCount).toBe(1)
})

test('刷新已知 thread_id 后通过 GET 恢复服务端状态', async ({ page }) => {
  await installSettings(page)
  await page.addInitScript(
    ({ record }) => {
      localStorage.setItem(
        'rag-agent.sessions.v1:user-001:project-001',
        JSON.stringify({ version: 1, sessions: [record] }),
      )
    },
    { record: sessionRecord() },
  )
  await mockHealth(page)
  await page.route(
    `${apiBaseUrl}/api/v1/projects/project-001/sessions/${threadId}`,
    async (route) => {
      expect(route.request().headers()['x-user-id']).toBe('user-001')
      await route.fulfill({ json: { ...completedRagResponse, project_id: 'project-001' } })
    },
  )

  await page.goto(`/workspace/${threadId}`)
  await expect(page.getByText('FinFET 使用鳍式沟道增强栅控能力。')).toBeVisible()
  await page.reload()
  await expect(page.getByText('FinFET 使用鳍式沟道增强栅控能力。')).toBeVisible()
})

test('DOCX 使用带身份请求头的 fetch 下载', async ({ page }) => {
  await installSettings(page)
  await mockHealth(page)
  await page.route(`${apiBaseUrl}/api/v1/projects/project-001/sessions`, (route) =>
    route.fulfill({
      status: 201,
      json: {
        ...completedRagResponse,
        project_id: 'project-001',
        query_type: 'technology_selection',
        response_type: 'report',
        report: '# 报告',
        report_data: validReportData,
        report_url: `/api/v1/projects/project-001/sessions/${threadId}/report`,
      },
    }),
  )
  let reportRequest: Route | null = null
  await page.route(
    `${apiBaseUrl}/api/v1/projects/project-001/sessions/${threadId}/report`,
    async (route) => {
      reportRequest = route
      await route.fulfill({
        body: 'fake-docx',
        headers: {
          'Content-Disposition': 'attachment; filename="selection.docx"',
          'Content-Type': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
          'Access-Control-Expose-Headers': 'Content-Disposition',
        },
      })
    },
  )

  await page.goto('/workspace')
  await page.getByRole('textbox', { name: '消息', exact: true }).fill('生成技术选型报告')
  await page.getByLabel('需要 DOCX').check()
  await page.getByRole('button', { name: '发送' }).click()
  const downloadPromise = page.waitForEvent('download')
  await page.getByRole('button', { name: '下载 DOCX 报告' }).click()
  const download = await downloadPromise
  expect(download.suggestedFilename()).toBe('selection.docx')
  expect(reportRequest?.request().headers()['x-user-id']).toBe('user-001')
})

test('窄屏保留核心对话并以抽屉打开会话和检查器', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await installSettings(page)
  await mockHealth(page)
  await page.goto('/workspace')
  await expect(page.getByRole('textbox', { name: '消息', exact: true })).toBeVisible()
  await page.getByRole('button', { name: '打开会话列表' }).click()
  await expect(page.getByRole('dialog', { name: '本机会话' })).toBeVisible()
  await page.getByRole('button', { name: '关闭', exact: true }).click()
  await page.getByRole('button', { name: '打开证据检查器' }).click()
  await expect(page.getByRole('dialog', { name: '证据与结论' })).toBeVisible()
})

const errorCases = [
  [404, '服务端找不到这个会话'],
  [409, '该会话仍在处理上一条消息'],
  [422, '提交内容不符合接口要求'],
  [500, '内部细节已隐藏'],
  [503, '依赖的服务暂时不可用'],
] as const

for (const [status, message] of errorCases) {
  test(`将 ${status} 显示为可理解且不泄露内部信息的错误`, async ({ page }) => {
    await installSettings(page)
    await mockHealth(page)
    await page.route(`${apiBaseUrl}/api/v1/projects/project-001/sessions`, (route) =>
      route.fulfill({
        status,
        json: { detail: { code: `mock_${status}`, message: 'PRIVATE TRACEBACK' } },
      }),
    )
    await page.goto('/workspace')
    await page.getByRole('textbox', { name: '消息', exact: true }).fill('触发错误')
    await page.getByRole('button', { name: '发送' }).click()
    await expect(page.getByText(new RegExp(message))).toBeVisible()
    await expect(page.getByText('PRIVATE TRACEBACK')).toHaveCount(0)
  })
}

test('网络中断不会自动重试 POST', async ({ page }) => {
  await installSettings(page)
  await mockHealth(page)
  let requestCount = 0
  await page.route(`${apiBaseUrl}/api/v1/projects/project-001/sessions`, async (route) => {
    requestCount += 1
    await route.abort('connectionfailed')
  })
  await page.goto('/workspace')
  await page.getByRole('textbox', { name: '消息', exact: true }).fill('网络中断测试')
  await page.getByRole('button', { name: '发送' }).click()
  await expect(page.getByText('无法连接 Agent API')).toBeVisible()
  expect(requestCount).toBe(1)
})
