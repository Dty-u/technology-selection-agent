export type ApiErrorCode =
  | 'session_not_found'
  | 'session_busy'
  | 'service_unavailable'
  | 'validation_error'
  | 'invalid_response'
  | 'network_error'
  | 'request_aborted'
  | 'unsafe_report_url'
  | 'internal_error'

export class ApiError extends Error {
  readonly status: number
  readonly code: ApiErrorCode | string

  constructor(status: number, code: ApiErrorCode | string, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.code = code
  }
}

const titlesByCode: Record<string, string> = {
  session_not_found: '会话不存在',
  session_busy: '会话正忙',
  service_unavailable: '服务暂不可用',
  validation_error: '输入有误',
  invalid_response: '响应格式异常',
  network_error: '无法连接服务',
  request_aborted: '已停止等待',
  unsafe_report_url: '报告地址不安全',
  internal_error: '服务异常',
  thread_id_mismatch: '会话标识不一致',
}

export function errorTitle(code: string): string {
  return titlesByCode[code] ?? '请求失败'
}

const messagesByStatus: Record<number, string> = {
  404: '服务端找不到这个会话，它可能已关闭或不属于当前项目。',
  409: '该会话仍在处理上一条消息，请稍后同步最新状态。',
  422: '提交内容不符合接口要求，请检查输入。',
  500: '服务暂时无法完成请求，内部细节已隐藏。',
  503: 'Agent 依赖的服务暂时不可用，请稍后再试。',
}

export async function apiErrorFromResponse(response: Response): Promise<ApiError> {
  let code: string | undefined
  try {
    const body = (await response.json()) as { detail?: { code?: unknown } }
    if (typeof body.detail?.code === 'string') code = body.detail.code
  } catch {
    // Error bodies are untrusted and optional; never expose the raw text.
  }
  return new ApiError(
    response.status,
    code ?? (response.status === 422 ? 'validation_error' : 'internal_error'),
    messagesByStatus[response.status] ?? '请求失败，请稍后重试。',
  )
}

export function apiErrorFromUnknown(error: unknown): ApiError {
  if (error instanceof ApiError) return error
  if (error instanceof DOMException && error.name === 'AbortError') {
    return new ApiError(0, 'request_aborted', '已停止在当前页面等待，服务端任务可能仍在继续。')
  }
  return new ApiError(0, 'network_error', '无法连接 Agent API，请检查地址和服务状态。')
}

const statusByStreamCode: Record<string, number> = {
  session_not_found: 404,
  session_busy: 409,
  service_unavailable: 503,
  internal_error: 500,
}

export function apiErrorFromStreamEvent(value: unknown): ApiError {
  const code =
    typeof (value as { code?: unknown } | null)?.code === 'string'
      ? (value as { code: string }).code
      : 'internal_error'
  const status = statusByStreamCode[code] ?? 500
  return new ApiError(status, code, messagesByStatus[status] ?? '请求失败，请稍后重试。')
}
