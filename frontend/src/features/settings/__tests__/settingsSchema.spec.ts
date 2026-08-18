import { describe, expect, it } from 'vitest'

import { normalizeSettings } from '../settingsSchema'

describe('normalizeSettings', () => {
  it('trims identifiers and removes trailing slashes from the API URL', () => {
    expect(
      normalizeSettings({
        apiBaseUrl: ' http://localhost:8000/// ',
        userId: ' user-a ',
        projectId: ' project-a ',
      }),
    ).toEqual({
      apiBaseUrl: 'http://localhost:8000',
      userId: 'user-a',
      projectId: 'project-a',
    })
  })

  it('rejects empty scopes and API URLs containing credentials', () => {
    expect(() =>
      normalizeSettings({
        apiBaseUrl: 'http://user:secret@localhost:8000',
        userId: ' ',
        projectId: 'project-a',
      }),
    ).toThrow(/API 地址/)
  })
})
