import { describe, expect, it, vi } from 'vitest'

import { readJson, writeJson, type StorageLike } from '../localStorage'

describe('safe local storage', () => {
  it('returns a warning instead of throwing for corrupted JSON', () => {
    const storage: StorageLike = {
      getItem: () => '{broken',
      setItem: vi.fn<(key: string, value: string) => void>(),
      removeItem: vi.fn<(key: string) => void>(),
    }

    const result = readJson(storage, 'key', (value) => value as object)

    expect(result.value).toBeNull()
    expect(result.warning).toMatch(/损坏/)
  })

  it('reports quota failures so the caller can keep in-memory state visible', () => {
    const storage: StorageLike = {
      getItem: () => null,
      setItem: () => {
        throw new DOMException('quota', 'QuotaExceededError')
      },
      removeItem: vi.fn<(key: string) => void>(),
    }

    const result = writeJson(storage, 'key', { value: 1 })

    expect(result.ok).toBe(false)
    expect(result.warning).toMatch(/浏览器/)
  })
})
