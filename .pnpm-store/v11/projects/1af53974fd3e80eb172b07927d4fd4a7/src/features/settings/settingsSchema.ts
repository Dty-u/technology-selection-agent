import { z } from 'zod'

export type AppSettings = {
  apiBaseUrl: string
  userId: string
  projectId: string
}

const apiUrlSchema = z
  .string()
  .trim()
  .url('请输入完整的 API 地址')
  .transform((value, context) => {
    const url = new URL(value)
    if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password) {
      context.addIssue({
        code: 'custom',
        message: 'API 地址只允许 http/https，且不能包含用户名或密码',
      })
      return z.NEVER
    }
    return value.replace(/\/+$/, '')
  })

export const appSettingsSchema = z.object({
  apiBaseUrl: apiUrlSchema,
  userId: z.string().trim().min(1, '请输入用户 ID').max(128, '用户 ID 不能超过 128 个字符'),
  projectId: z.string().trim().min(1, '请输入项目 ID').max(128, '项目 ID 不能超过 128 个字符'),
})

export function normalizeSettings(value: unknown): AppSettings {
  return appSettingsSchema.parse(value)
}
