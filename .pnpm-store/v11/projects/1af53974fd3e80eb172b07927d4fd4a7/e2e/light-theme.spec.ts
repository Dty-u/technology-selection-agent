import { expect, test } from '@playwright/test'

const apiBaseUrl = 'http://localhost:8000'

test('工作台使用浅暖色编辑台主题', async ({ page }) => {
  const browserErrors: string[] = []
  page.on('pageerror', (error) => browserErrors.push(error.message))
  page.on('console', (message) => {
    if (message.type() === 'error') browserErrors.push(message.text())
  })

  await page.addInitScript(
    ({ baseUrl }) => {
      localStorage.setItem(
        'rag-agent.settings.v1',
        JSON.stringify({ apiBaseUrl: baseUrl, userId: 'user-001', projectId: 'project-001' }),
      )
    },
    { baseUrl: apiBaseUrl },
  )
  await page.route(`${apiBaseUrl}/health/live`, (route) =>
    route.fulfill({ json: { status: 'ok' } }),
  )

  await page.goto('/workspace')

  const theme = await page.evaluate(() => ({
    colorScheme: getComputedStyle(document.documentElement).colorScheme,
    bodyBackground: getComputedStyle(document.body).backgroundColor,
    sidebarBackground: getComputedStyle(document.querySelector('.session-sidebar')!)
      .backgroundColor,
    primaryBackground: getComputedStyle(document.querySelector('.app-button--primary')!)
      .backgroundColor,
  }))

  expect(theme).toEqual({
    colorScheme: 'light',
    bodyBackground: 'rgb(247, 245, 239)',
    sidebarBackground: 'rgb(251, 250, 247)',
    primaryBackground: 'rgb(11, 143, 121)',
  })
  expect(browserErrors).toEqual([])
})
