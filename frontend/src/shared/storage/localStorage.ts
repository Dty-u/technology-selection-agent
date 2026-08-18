export type StorageLike = Pick<Storage, 'getItem' | 'setItem' | 'removeItem'>

export type StorageResult<T> = {
  value: T
  warning?: string
}

export function readJson<T>(
  storage: StorageLike,
  key: string,
  parse: (value: unknown) => T,
): StorageResult<T | null> {
  const raw = storage.getItem(key)
  if (raw === null) return { value: null }
  try {
    return { value: parse(JSON.parse(raw) as unknown) }
  } catch {
    return { value: null, warning: '浏览器中的本地数据已损坏，本次已安全忽略。' }
  }
}

export function writeJson(
  storage: StorageLike,
  key: string,
  value: unknown,
): { ok: boolean; warning?: string } {
  try {
    storage.setItem(key, JSON.stringify(value))
    return { ok: true }
  } catch {
    return {
      ok: false,
      warning: '浏览器无法保存本地记录；当前内容仍保留在本页面内存中。',
    }
  }
}
