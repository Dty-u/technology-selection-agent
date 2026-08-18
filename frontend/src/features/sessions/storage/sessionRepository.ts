import { z } from 'zod'

import {
  readJson,
  writeJson,
  type StorageLike,
  type StorageResult,
} from '@/shared/storage/localStorage'
import type { LocalSession } from '../model/sessionTypes'

const messageSchema = z.object({
  id: z.string(),
  role: z.enum(['user', 'assistant', 'system']),
  content: z.string(),
  createdAt: z.string(),
  delivery: z.enum(['pending', 'sent', 'failed']),
  requestId: z.string().optional(),
})

const localSessionSchema = z.object({
  threadId: z.string(),
  userId: z.string(),
  projectId: z.string(),
  title: z.string(),
  createdAt: z.string(),
  updatedAt: z.string(),
  status: z.enum(['running', 'completed', 'awaiting_user', 'needs_human_review', 'error']),
  messages: z.array(messageSchema),
})

const envelopeSchema = z.object({
  version: z.literal(1),
  sessions: z.array(localSessionSchema),
})

export function sessionStorageKey(userId: string, projectId: string): string {
  return `rag-agent.sessions.v1:${encodeURIComponent(userId)}:${encodeURIComponent(projectId)}`
}

export function loadSessionRecords(
  storage: StorageLike,
  userId: string,
  projectId: string,
): StorageResult<LocalSession[]> {
  const result = readJson(storage, sessionStorageKey(userId, projectId), (value) =>
    envelopeSchema.parse(value),
  )
  if (!result.value) return { value: [], warning: result.warning }

  const matching = result.value.sessions.filter(
    (record) => record.userId === userId && record.projectId === projectId,
  )
  return {
    value: matching,
    warning:
      matching.length === result.value.sessions.length
        ? result.warning
        : '检测到不属于当前作用域的本地会话，已安全忽略。',
  }
}

export function saveSessionRecords(
  storage: StorageLike,
  userId: string,
  projectId: string,
  records: LocalSession[],
): { ok: boolean; warning?: string } {
  const recent = records
    .filter((record) => record.userId === userId && record.projectId === projectId)
    .slice()
    .sort((left: LocalSession, right: LocalSession) =>
      right.updatedAt.localeCompare(left.updatedAt),
    )
    .slice(0, 20)
  return writeJson(storage, sessionStorageKey(userId, projectId), {
    version: 1,
    sessions: recent,
  })
}
