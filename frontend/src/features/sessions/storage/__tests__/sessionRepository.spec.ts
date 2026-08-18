import { describe, expect, it } from 'vitest'

import type { StorageLike } from '@/shared/storage/localStorage'
import { loadSessionRecords, saveSessionRecords, sessionStorageKey } from '../sessionRepository'
import type { LocalSession } from '../../model/sessionTypes'

class MemoryStorage implements StorageLike {
  readonly values = new Map<string, string>()

  getItem(key: string) {
    return this.values.get(key) ?? null
  }

  setItem(key: string, value: string) {
    this.values.set(key, value)
  }

  removeItem(key: string) {
    this.values.delete(key)
  }
}

function session(index: number): LocalSession {
  const timestamp = new Date(Date.UTC(2026, 7, 18, 0, index)).toISOString()
  return {
    threadId: `00000000-0000-4000-8000-${String(index).padStart(12, '0')}`,
    userId: 'user-a',
    projectId: 'project-a',
    title: `Session ${index}`,
    createdAt: timestamp,
    updatedAt: timestamp,
    status: 'completed',
    messages: [],
  }
}

describe('sessionRepository', () => {
  it('uses a different storage key for every user and project scope', () => {
    expect(sessionStorageKey('user-a', 'project-a')).not.toBe(
      sessionStorageKey('user-a', 'project-b'),
    )
    expect(sessionStorageKey('user-a', 'project-a')).not.toBe(
      sessionStorageKey('user-b', 'project-a'),
    )
  })

  it('keeps only the 20 most recently updated sessions', () => {
    const storage = new MemoryStorage()
    const records = Array.from({ length: 22 }, (_, index) => session(index))

    const result = saveSessionRecords(storage, 'user-a', 'project-a', records)
    const loaded = loadSessionRecords(storage, 'user-a', 'project-a')

    expect(result.ok).toBe(true)
    expect(loaded.value).toHaveLength(20)
    expect(loaded.value[0]?.title).toBe('Session 21')
    expect(loaded.value[loaded.value.length - 1]?.title).toBe('Session 2')
  })

  it('does not load records whose embedded scope differs from the selected workspace', () => {
    const storage = new MemoryStorage()
    const wrongScope = { ...session(1), projectId: 'project-b' }
    storage.setItem(
      sessionStorageKey('user-a', 'project-a'),
      JSON.stringify({ version: 1, sessions: [wrongScope] }),
    )

    const loaded = loadSessionRecords(storage, 'user-a', 'project-a')

    expect(loaded.value).toEqual([])
    expect(loaded.warning).toMatch(/作用域/)
  })
})
